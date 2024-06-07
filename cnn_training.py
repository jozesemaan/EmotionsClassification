import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt

# Define CNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes: Angry, Engaged, Happy, Neutral
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*12*12)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Load datasets
data_dir = r'C:\Users\jozes\Desktop\COMP472\Project\DataSetFinal'

def load_datasets(data_dir):
    train_image_paths = []
    train_labels = []
    test_image_paths = []
    test_labels = []
    classes = ['Angry', 'Engaged', 'Happy', 'Neutral']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        train_dir = os.path.join(data_dir, cls, cls, 'train')
        test_dir = os.path.join(data_dir, cls, cls, 'test')

        # Print the directory paths to verify
        print(f"Train directory for class '{cls}': {train_dir}")
        print(f"Test directory for class '{cls}': {test_dir}")

        # Check if directories exist
        if not os.path.exists(train_dir):
            print(f"Error: Train directory '{train_dir}' does not exist.")
        else:
            train_image_paths.extend([os.path.join(train_dir, img) for img in os.listdir(train_dir)])
            train_labels.extend([class_to_idx[cls]] * len(os.listdir(train_dir)))
        
        if not os.path.exists(test_dir):
            print(f"Error: Test directory '{test_dir}' does not exist.")
        else:
            test_image_paths.extend([os.path.join(test_dir, img) for img in os.listdir(test_dir)])
            test_labels.extend([class_to_idx[cls]] * len(os.listdir(test_dir)))

    train_dataset = CustomDataset(train_image_paths, train_labels, transform=transform)
    test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)

    return train_dataset, test_dataset

train_dataset, test_dataset = load_datasets(data_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mapping of class indices to class names
class_names = ['Angry', 'Engaged', 'Happy', 'Neutral']

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

        # Validation
        val_loss = evaluate_model(model, test_loader, criterion)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, test_loader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return total_loss / len(test_loader)

# Initialize model, optimizer, and loss function
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
