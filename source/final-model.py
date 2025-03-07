import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load Pretrained Model from Hugging Face
pipe = pipeline("image-classification", model="Falcom/animal-classifier")
processor = AutoImageProcessor.from_pretrained("Falcom/animal-classifier")
model = AutoModelForImageClassification.from_pretrained("Falcom/animal-classifier")
model.train()

# Load Training and Test Dataset
train_images = np.load('cleaned_animals_dataset/train_data_images.npy')
train_labels = np.load('cleaned_animals_dataset/train_data_labels.npy')
test_images = np.load('cleaned_animals_dataset/test_data_images.npy')
test_labels = np.load('cleaned_animals_dataset/test_data_labels.npy')

# Convert NumPy images to PIL format for processing
def preprocess_images(images):
    """Convert NumPy array images to a list of PIL images and ensure normalization."""
    return [processor(Image.fromarray((img * 255).astype(np.uint8)), return_tensors="pt")['pixel_values'].squeeze(0) for img in images]

def process_batch(images, labels, device):
    """Process images for model input."""
    inputs = torch.stack(images).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return inputs, labels

train_pil_images = preprocess_images(train_images)
test_pil_images = preprocess_images(test_images)

# Training Setup
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Loop
def train_model(model, images, labels, optimizer, criterion, batch_size=16, epochs=3):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i in tqdm(range(0, len(images), batch_size), desc=f"Epoch {epoch+1}"):
            batch_images = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            inputs, targets = process_batch(batch_images, batch_labels, device)
            
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {running_loss/(len(images)/batch_size):.4f}, Accuracy: {accuracy:.2f}%")

# Train the model
train_model(model, train_pil_images, train_labels, optimizer, criterion, batch_size=16, epochs=3)

# Save Fine-Tuned Model
torch.save(model.state_dict(), "fine_tuned_animal_classifier.pth")

# Evaluate Model
def evaluate_model(model, images, labels, batch_size=16):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            inputs, targets = process_batch(batch_images, batch_labels, device)
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Compute accuracy before and after fine-tuning
pretrained_accuracy = evaluate_model(model, test_pil_images, test_labels)
print(f'Pretrained Model Test Accuracy: {pretrained_accuracy:.2f}%')

# Load Fine-Tuned Model and Evaluate
model.load_state_dict(torch.load("fine_tuned_animal_classifier.pth"))
fine_tuned_accuracy = evaluate_model(model, test_pil_images, test_labels)
print(f'Fine-Tuned Model Test Accuracy: {fine_tuned_accuracy:.2f}%')

# Debugging: Print a few test predictions
def debug_predictions(model, images, labels, num_samples=5):
    model.eval()
    print("\nSample Predictions:")
    with torch.no_grad():
        for i in range(num_samples):
            image = images[i].unsqueeze(0).to(device)
            output = model(image).logits
            _, predicted = torch.max(output, 1)
            print(f"Actual: {labels[i]}, Predicted: {predicted.item()}")

debug_predictions(model, test_pil_images, test_labels)