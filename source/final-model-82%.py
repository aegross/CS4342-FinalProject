import numpy as np
import torch
import torch.nn as nn
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
from tqdm import tqdm

# Load Pretrained Model from Hugging Face
pipe = pipeline("image-classification", model="Falcom/animal-classifier")
processor = AutoImageProcessor.from_pretrained("Falcom/animal-classifier")
model = AutoModelForImageClassification.from_pretrained("Falcom/animal-classifier")
model.eval()

# Load Test Dataset
test_images = np.load('cleaned_animals_dataset/test_data_images.npy')
test_labels = np.load('cleaned_animals_dataset/test_data_labels.npy')

# Convert NumPy images to PIL format for processing
def preprocess_images(images):
    """Convert NumPy array images to a list of PIL images."""
    pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    return pil_images

test_pil_images = preprocess_images(test_images)

# Make Predictions
predictions = []
for img in tqdm(test_pil_images, desc="Classifying images"):
    outputs = pipe(img)
    predicted_label = outputs[0]['label']  # Take top-1 prediction
    predictions.append(predicted_label)

# Convert predictions to numerical labels
label_mapping = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3, 'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7, 'spider': 8, 'squirrel': 9}
pred_labels = np.array([label_mapping.get(pred, -1) for pred in predictions])

# Compute Accuracy
correct = np.sum(pred_labels == test_labels)
total = len(test_labels)
accuracy = correct / total * 100

print(f'Final Model Test Accuracy: {accuracy:.2f}%')
