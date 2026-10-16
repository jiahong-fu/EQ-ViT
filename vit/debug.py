
from PIL import Image
import requests
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import ViTImageProcessor, ViTForImageClassification

# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
# pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
