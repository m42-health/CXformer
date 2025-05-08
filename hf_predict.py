from transformers import AutoModel, AutoImageProcessor
from PIL import Image

model_name = 'm42-health/CXformer-base'

image_processor = AutoImageProcessor.from_pretrained(model_name,trust_remote_code=True)
model = AutoModel.from_pretrained(model_name)

model.eval()

image = Image.open('sample_cxr.png')

image = image_processor(image, return_tensors='pt')
print(image['pixel_values'].shape) # [1,3,518,518]

print("Doing forward...!!")
output = model(**image).last_hidden_state  # [1, 1374, 768]

print(output.shape)