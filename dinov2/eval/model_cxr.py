from dinov2.utils.config import get_cfg_from_args
import argparse, os
from dinov2.train.ssl_meta_arch import SSLMetaArch


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    # parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.") 

    return parser

def get_param_counts(model):
    return sum([x.numel() for x in model.parameters() if x.requires_grad]) / 1000_000

if __name__ == "__main__":
    description = "DINOv2 CXR model param count"
    args = get_args_parser(add_help=True).parse_args()
    cfg = get_cfg_from_args(args)
    # cfg must also have a key "train" for train dataset
    model = SSLMetaArch(cfg)       
    # breakpoint() 
    if model.teacher.backbone.blocks[-1].mlp.__class__.__name__ == 'MixtureActivationMlp':
        # logger.info("Using MixtureActivationMlp")
        # logger.info("Activations: {}".format(model.teacher.backbone.blocks[-1].mlp.activations))
        raise NotImplementedError

    print(f"Student_model: {get_param_counts(model.student)} M")

    breakpoint()