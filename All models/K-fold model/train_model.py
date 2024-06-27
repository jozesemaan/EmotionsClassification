import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
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
        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Load datasets
data_dir = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\fulldataset'

def load_datasets(data_dir):
    image_paths = []
    labels = []
    classes = ['Angry', 'Engaged', 'Happy', 'Neutral']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Error: Directory '{cls_dir}' does not exist.")
            continue
        image_paths.extend([os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.endswith(('png', 'jpg', 'jpeg'))])
        labels.extend([class_to_idx[cls]] * len(os.listdir(cls_dir)))

    dataset = CustomDataset(image_paths, labels, transform=transform)
    return dataset

dataset = load_datasets(data_dir)

# Class names
class_names = ['Angry', 'Engaged', 'Happy', 'Neutral']

# K-fold cross-validation
def cross_validate_model(dataset, model_class, criterion, optimizer_class, k=10, num_epochs=20, patience=3, min_epochs=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    all_metrics = {
        'accuracy': [],
        'macro_precision': [],
        'micro_precision': [],
        'macro_recall': [],
        'micro_recall': [],
        'macro_f1': [],
        'micro_f1': [],
        'loss': []
    }
    
    fold = 0
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        fold += 1
        print(f'Fold {fold}/{k}')
        train_val_subset = Subset(dataset, train_idx)
        
        # Further split into training and validation subsets
        train_size = int(0.85 * len(train_val_subset))
        val_size = len(train_val_subset) - train_size
        train_subset, val_subset = random_split(train_val_subset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        test_subset = Subset(dataset, val_idx)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        model = model_class()
        optimizer = optimizer_class(model.parameters(), lr=0.001)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, min_epochs)
        
        val_loss, metrics = evaluate_model(model, test_loader, criterion)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    avg_metrics = {key: np.mean(all_metrics[key]) for key in all_metrics}
    std_metrics = {key: np.std(all_metrics[key]) for key in all_metrics}
    
    for key in avg_metrics:
        print(f'{key.capitalize()} - Mean: {avg_metrics[key]:.4f}, Std: {std_metrics[key]:.4f}')

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=3, min_epochs=10):
    model.train()
    best_val_loss = float('inf')
    no_improvement_cnt = 0

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
        print(f'Epoch [{epoch+1}/{num_epochs}]: Training Loss: {avg_train_loss:.4f}')

        val_loss, _ = evaluate_model(model, val_loader, criterion, print_metrics=False)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_cnt = 0
        else:
            no_improvement_cnt += 1

        if no_improvement_cnt >= patience and epoch >= min_epochs:
            print(f'Early stopping after {epoch+1} epochs.')
            break

def evaluate_model(model, test_loader, criterion, print_metrics=True):
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

    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    val_loss = total_loss / len(test_loader)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'micro_precision': precision_micro,
        'macro_recall': recall_macro,
        'micro_recall': recall_micro,
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'loss': val_loss
    }

    if print_metrics:
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Macro Precision: {precision_macro:.4f}, Micro Precision: {precision_micro:.4f}')
        print(f'Macro Recall: {recall_macro:.4f}, Micro Recall: {recall_micro:.4f}')
        print(f'Macro F1-score: {f1_macro:.4f}, Micro F1-score: {f1_micro:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Annotate the confusion matrix with the counts
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    return val_loss, metrics

# Initialize criterion
criterion = nn.CrossEntropyLoss()

# Perform 10-fold cross-validation
cross_validate_model(dataset, SimpleCNN, criterion, optim.Adam, k=10, num_epochs=20, patience=3, min_epochs=10)
