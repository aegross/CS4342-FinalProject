import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
train_images = np.load('cleaned_animals_dataset/train_data_images.npy')
train_labels = np.load('cleaned_animals_dataset/train_data_labels.npy')
test_images = np.load('cleaned_animals_dataset/test_data_images.npy')
test_labels = np.load('cleaned_animals_dataset/test_data_labels.npy')

# Print unique labels and their counts
unique_labels, counts = np.unique(train_labels, return_counts=True)
print("\nClass distribution in training set:")
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

# Simple normalization to [0, 1]
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Calculate mean and std for normalization
train_mean = train_images.mean()
train_std = train_images.std()

# Normalize with mean and std
train_images = (train_images - train_mean) / train_std
test_images = (test_images - train_mean) / train_std

print(f"\nTraining data shape: {train_images.shape}")
print(f"Test data shape: {test_images.shape}")
print(f"Data range after normalization: [{train_images.min():.2f}, {train_images.max():.2f}]")

CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        
        # Simple convolutional layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Light dropout
            nn.Dropout2d(0.1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AnimalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create datasets
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

train_dataset = AnimalDataset(train_images, train_labels, transform=train_transform)
test_dataset = AnimalDataset(test_images, test_labels)

# Smaller batch size for better generalization
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer
num_classes = len(np.unique(train_labels))
model = SimpleNet(num_classes=num_classes).to(device)
print("\nModel architecture:")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Training parameters
num_epochs = 100
best_test_accuracy = 0
patience = 10
patience_counter = 0

# Lists to store metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def evaluate(model, dataloader, criterion):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # Per-class accuracy
            for i in range(num_classes):
                mask = (labels == i)
                class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                class_total[i] += mask.sum().item()
    
    # Print per-class accuracies
    print("\nPer-class accuracies:")
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{CLASS_NAMES[i]}: {accuracy:.2f}%")
    
    # Calculate overall accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_accuracy = 100 * total_correct / total_samples
    test_loss = running_loss / len(dataloader)
    
    return overall_accuracy, test_loss

print("\nStarting training...")

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training phase
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'accuracy': f'{100 * correct / total:.1f}%'
        })
    
    # Calculate training metrics
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Evaluation phase
    test_accuracy, test_loss = evaluate(model, test_loader, criterion)
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)
    
    print(f'\nEpoch [{epoch+1}/{num_epochs}]')
    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Overall Test Accuracy: {test_accuracy:.2f}%')
    
    # Update learning rate scheduler
    scheduler.step(test_accuracy)
    
    # Save best model and implement early stopping
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_accuracy,
        }, 'best_animal_cnn_model.pth')
        patience_counter = 0
        print(f'New best model saved with test accuracy: {test_accuracy:.2f}%')
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f'\nEarly stopping triggered after epoch {epoch+1}')
        break

# Plot training and testing metrics
plt.figure(figsize=(12, 4))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Testing Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Load best model for final evaluation
print("\nLoading best model for final evaluation...")
checkpoint = torch.load('best_animal_cnn_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Perform final evaluation on the full test set
print("\nFinal Evaluation on Test Set (Best Model):")
model.eval()
test_accuracy = evaluate(model, test_loader, criterion)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# Print best test accuracy achieved during training
best_test_accuracy = max(test_accuracies)
best_epoch = test_accuracies.index(best_test_accuracy) + 1
print(f"\nBest Test Accuracy: {best_test_accuracy:.2f}% (Epoch {best_epoch})")


#Per-class accuracies:
# butterfly: 72.34%
# cat: 54.49%
# chicken: 79.19%
# cow: 72.46%
# dog: 78.83%
# elephant: 73.45%
# horse: 75.24%
# sheep: 65.93%
# spider: 87.88%
# squirrel: 68.63%
# Final Test Accuracy: 75.73%

# Best Test Accuracy: 75.73% (Epoch 58)