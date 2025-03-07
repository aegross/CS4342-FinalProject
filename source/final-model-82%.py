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

# load training set (not actually used for training)
train_images = np.load('cleaned_animals_dataset/train_data_images.npy')
train_labels = np.load('cleaned_animals_dataset/train_data_labels.npy')

# Convert NumPy images to PIL format for processing
def preprocess_images(images):
    """Convert NumPy array images to a list of PIL images."""
    pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    return pil_images

test_pil_images = preprocess_images(test_images)
train_pil_images = preprocess_images(train_images)

# Make Predictions
test_predictions = []
for img in tqdm(test_pil_images, desc="Classifying Testing Images"):
    outputs = pipe(img)
    predicted_label = outputs[0]['label']  # Take top-1 prediction
    test_predictions.append(predicted_label)

train_predictions = []
for img in tqdm(train_pil_images, desc="Classifying Training Images"):
    outputs = pipe(img)
    predicted_label = outputs[0]['label']  # Take top-1 prediction
    train_predictions.append(predicted_label)

# convert predictions to numerical labels
label_mapping = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3, 'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7, 'spider': 8, 'squirrel': 9}
test_pred_labels = np.array([label_mapping.get(pred, -1) for pred in test_predictions])
train_pred_labels = np.array([label_mapping.get(pred, -1) for pred in train_predictions])

# compute accuracies
test_correct = np.sum(test_pred_labels == test_labels)
test_total = len(test_labels)
test_accuracy = test_correct / test_total * 100

train_correct = np.sum(train_pred_labels == train_labels)
train_total = len(train_labels)
train_accuracy = train_correct / train_total * 100

print(f'Final Model Training Accuracy: {train_accuracy:.2f}%')
print(f'Final Model Test Accuracy: {test_accuracy:.2f}%')