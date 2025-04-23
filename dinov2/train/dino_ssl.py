import torch
from torch import nn

from dinov2.models import build_model
from dinov2.train.train_utils import get_checkpoint_statedict
from .io_utils import printit
from dinov2.layers import DINOHead
from functools import partial
from dinov2.loss import DINOLoss, iBOTPatchLoss
from dinov2.fsdp import reshard_fsdp_model, get_fsdp_modules
try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


class DINO_SSL(nn.Module):
    def __init__(self, cfg, local_rank=None, rank=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.local_rank = local_rank if local_rank else 0
        if not rank:
            self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        else:
            self.rank = rank
        self.do_dino = self.cfg.dino.loss_weight > 0
        self.do_ibot = self.cfg.ibot.loss_weight > 0
        self.ibot_separate_head = self.cfg.ibot.separate_head
        self.n_global_crops = self.cfg.crops.global_crops_number
        self.n_local_crops = self.cfg.crops.local_crops_number

        assert (
            self.do_dino and self.do_ibot and self.ibot_separate_head
        ), "We only support both dino and ibot with sep head for ibot"

        self.dino_out_dim = self.cfg.dino.head_n_prototypes
        self.dino_loss = DINOLoss(self.dino_out_dim)
        
        self.ibot_out_dim = self.cfg.ibot.head_n_prototypes
        self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)

        self.student_backbone, self.teacher_backbone, self.embed_dim = build_model(
            self.cfg.student, only_teacher=False, img_size=self.cfg.crops.global_crops_size
        )

        if self.cfg.student.pretrained_weights:
            # pretrained student statedict
            pt_st_sd = get_checkpoint_statedict(self.cfg.student.pretrained_weights)
            model_load_response = self.student_backbone.load_state_dict(pt_st_sd, strict=False)
            printit(f"Loading pretrained checkpoint into student backbone: {model_load_response}")

        if self.cfg.teacher.pretrained_weights:
            # pretrained student statedict
            pt_st_sd = get_checkpoint_statedict(self.cfg.teacher.pretrained_weights)
            model_load_response = self.teacher_backbone.load_state_dict(pt_st_sd, strict=False)
            printit(f"Loading pretrained checkpoint into teacher backbone: {model_load_response}")

        dino_head = partial(
            DINOHead,
            in_dim=self.embed_dim,
            out_dim=self.cfg.dino.head_n_prototypes,
            hidden_dim = self.cfg.dino.head_hidden_dim,
            bottleneck_dim=self.cfg.dino.head_bottleneck_dim,
            nlayers=self.cfg.dino.head_nlayers,
        )
        ibot_head = partial(
            DINOHead,
            in_dim=self.embed_dim,
            out_dim=self.cfg.ibot.head_n_prototypes,
            hidden_dim=self.cfg.ibot.head_hidden_dim,
            bottleneck_dim=self.cfg.ibot.head_bottleneck_dim,
            nlayers=self.cfg.ibot.head_nlayers,
        )

        self.student_dino_head = dino_head()
        self.student_ibot_head = ibot_head()

        self.teacher_dino_head = dino_head()
        self.teacher_ibot_head = ibot_head()

        # load dino and ibot heads
        if self.cfg.student.pretrained_weights:
            # pretrained student statedict
            pt_st_sd = get_checkpoint_statedict(self.cfg.student.pretrained_weights)
            heads = {k.replace('dino_head.',''): v for k, v in pt_st_sd.items() if "dino_head" in k}
            model_load_response = self.student_dino_head.load_state_dict(heads, strict=False)
            printit(f"Loading pretrained checkpoint into student dino head: {model_load_response}")

            heads = {k.replace('ibot_head.',''): v for k, v in pt_st_sd.items() if "ibot_head" in k}
            model_load_response = self.student_ibot_head.load_state_dict(heads, strict=False)
            printit(f"Loading pretrained checkpoint into student ibot head: {model_load_response}")
        
        if self.cfg.teacher.pretrained_weights:
            pt_tch_sd = get_checkpoint_statedict(self.cfg.teacher.pretrained_weights)
            heads = {k.replace('dino_head.',''): v for k, v in pt_tch_sd.items() if "dino_head" in k}
            model_load_response = self.teacher_dino_head.load_state_dict(heads, strict=False)
            printit(f"Loading pretrained checkpoint into teacher dino head: {model_load_response}")

            heads = {k.replace('ibot_head.',''): v for k, v in pt_tch_sd.items() if "ibot_head" in k}
            model_load_response = self.teacher_ibot_head.load_state_dict(heads, strict=False)
            printit(f"Loading pretrained checkpoint into teacher ibot head: {model_load_response}")

        # zero the teacher model params
        self.teacher_params = []
        self.teacher_params.extend(list(self.teacher_backbone.parameters()))
        self.teacher_params.extend(list(self.teacher_dino_head.parameters()))
        self.teacher_params.extend(list(self.teacher_ibot_head.parameters()))

        self.remove_grad_teacher()

    def remove_grad_teacher(self):
        for p in self.teacher_params:
            p.requires_grad = False

    @torch.no_grad()
    def teacher_forward(self, x):

        ## STEP1: Forward pass over teacher backbone
        global_crops = x["collated_global_crops"]
        n_masked_patches = x["mask_indices_list"].shape[0]
        teacher_backbone_output = self.teacher_backbone(global_crops, is_training=True)
        teacher_cls_tokens = teacher_backbone_output["x_norm_clstoken"].chunk(self.n_global_crops)

        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))

        ibot_teacher_patch_tokens = teacher_backbone_output["x_norm_patchtokens"]
        _dim = ibot_teacher_patch_tokens.shape[-1]
        n_cls_tokens = teacher_cls_tokens.shape[0]
        buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(x["upperbound"], _dim)
        torch.index_select(
            ibot_teacher_patch_tokens.flatten(0, 1),
            dim=0,
            index=x["mask_indices_list"],
            out=buffer_tensor_teacher[:n_masked_patches],
        )

        ## STEP2: FORWARD PASS OVER DINO & IBOT HEAD
        teacher_cls_tokens_after_head = self.teacher_dino_head(teacher_cls_tokens)
        masked_teacher_patch_tokens_after_head = self.teacher_ibot_head(buffer_tensor_teacher)[:n_masked_patches]

        ## STEP3: CENTER THE TEACHER OUTPUTS
        assert self.cfg.train.centering == "centering"

        teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(teacher_cls_tokens_after_head, teacher_temp=x['teacher_temp'])
        teacher_dino_softmaxed_centered_list = teacher_dino_softmaxed_centered_list.view(self.n_global_crops, -1, *teacher_cls_tokens_after_head.shape[1:])
        self.dino_loss.update_center(teacher_cls_tokens_after_head)

        masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
        masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
            masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=x['teacher_temp']
        )
        masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
        self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])


        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered


    @torch.no_grad()
    def teacher_eval_forward(self, x):

        ## STEP1: Forward pass over teacher backbone
        global_crops = x["collated_global_crops"]
        
        assert global_crops.shape[0] <= self.cfg.train.batch_size_per_gpu, "Only one global crop is supported for evaluation"

        n_masked_patches = x["mask_indices_list"].shape[0]

        teacher_backbone_output = self.teacher_backbone(global_crops, is_training=True)

        teacher_cls_tokens = teacher_backbone_output["x_norm_clstoken"]
        

        ibot_teacher_patch_tokens = teacher_backbone_output["x_norm_patchtokens"] # B, no. of tokens (- CLS), D_vit
        _dim = ibot_teacher_patch_tokens.shape[-1]
        n_cls_tokens = teacher_cls_tokens.shape[0]
        buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(x["upperbound"], _dim)
        torch.index_select(
            ibot_teacher_patch_tokens.flatten(0, 1),
            dim=0,
            index=x["mask_indices_list"],
            out=buffer_tensor_teacher[:n_masked_patches],
        )

        ## STEP2: FORWARD PASS OVER DINO & IBOT HEAD
        
        #   (a) DINO
        teacher_cls_tokens_after_head = self.teacher_dino_head(teacher_cls_tokens)
        
        #   (b) IBOT
        # masked_teacher_patch_tokens_after_head = self.teacher_ibot_head(buffer_tensor_teacher)[:n_masked_patches]
        ibot_teacher_patch_tokens_after_head = self.teacher_ibot_head(ibot_teacher_patch_tokens) # B, no. of tokens (- CLS), D_ibot


        ## STEP3: CENTER THE TEACHER OUTPUTS
        assert self.cfg.train.centering == "centering"

        teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(teacher_cls_tokens_after_head, teacher_temp=x['teacher_temp'])
        teacher_dino_softmaxed_centered_list = teacher_dino_softmaxed_centered_list.view(self.n_global_crops, -1, *teacher_cls_tokens_after_head.shape[1:])
        
        # masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
        teacher_ibot_softmaxed_centered_list = self.ibot_patch_loss.softmax_center_teacher(
            ibot_teacher_patch_tokens_after_head, teacher_temp=x['teacher_temp']
        )
        return teacher_dino_softmaxed_centered_list, teacher_ibot_softmaxed_centered_list


    def student_forward(self, x):
        n_masked_patches = x['mask_indices_list'].shape[0]
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student_backbone(
            [x['collated_global_crops'], x['collated_local_crops']], masks=[x['collated_masks'], None], is_training=True
        )

        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        
        inputs_for_student_head_list = []
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))        
        
        ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
        buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(x['upperbound'], student_global_cls_tokens.shape[-1])
        buffer_tensor_patch_tokens[:n_masked_patches].copy_(
            torch.index_select(ibot_student_patch_tokens.flatten(0,1), dim=0, index=x['mask_indices_list'])
        )
        student_global_masked_patch_tokens_after_head = self.student_ibot_head(buffer_tensor_patch_tokens)[:n_masked_patches]

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student_dino_head(cat_inputs))

        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)    


        return student_local_cls_tokens_after_head, student_global_cls_tokens_after_head, student_local_cls_tokens, student_global_masked_patch_tokens_after_head


    def forward(self, x):
        student_local_output, student_global_output, student_local_cls_tokens, student_global_masked_patch_tokens_after_head = self.student_forward(x)   
        teacher_dino_output, teacher_ibot_output = self.teacher_forward(x)
    
        teacher_output = {'dino_output': teacher_dino_output, 'ibot_output': teacher_ibot_output}
        student_output = {'local_output': student_local_output, 'global_output': student_global_output, 'student_local_cls_tokens': student_local_cls_tokens, 'student_global_masked_patch_tokens_after_head':student_global_masked_patch_tokens_after_head}
        # teacher_output = None
        return {"teacher_output": teacher_output, "student_output": student_output}

    def set_train(self):
        self.student_backbone.train()
        self.student_dino_head.train()
        self.student_ibot_head.train()

        self.remove_grad_teacher()
        self.teacher_backbone.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()

    def set_eval(self):
        self.student_backbone.eval()
        self.student_dino_head.eval()
        self.student_ibot_head.eval()

        self.remove_grad_teacher()
        self.teacher_backbone.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()

    def update_teacher(self, m):
        student_param_list = get_fsdp_modules(self.student_backbone)
        teacher_param_list = get_fsdp_modules(self.teacher_backbone)
        
        student_param_list.extend(get_fsdp_modules(self.student_dino_head))
        teacher_param_list.extend(get_fsdp_modules(self.teacher_dino_head))

        student_param_list.extend(get_fsdp_modules(self.student_ibot_head))
        teacher_param_list.extend(get_fsdp_modules(self.teacher_ibot_head))
        
        with torch.no_grad():
            for ms, mt in zip(student_param_list, teacher_param_list):
            # for ms, mt in zip(self.student_backbone[k], self.teacher_backbone[k]):
                student_param_list += ms.params
                teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

            