import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the SimpleCNN architecture
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
data_dir = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\Bias\20-30_percent_bias'

def load_datasets(data_dir):
    test_image_paths = []
    test_labels = []
    classes = ['Angry', 'Engaged', 'Happy', 'Neutral']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)

        # Check if directory exists
        if not os.path.exists(cls_dir):
            print(f"Error: Directory '{cls_dir}' does not exist.")
        else:
            test_image_paths.extend([os.path.join(cls_dir, img) for img in os.listdir(cls_dir)])
            test_labels.extend([class_to_idx[cls]] * len(os.listdir(cls_dir)))

    test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)

    return test_dataset

# Initialize model, criterion, and dataset
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
test_dataset = load_datasets(data_dir)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mapping of class indices to class names
class_names = ['Angry', 'Engaged', 'Happy', 'Neutral']

# Function to load the best model
def load_best_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Error: Model weights not found at {model_path}")

# Evaluation function
def evaluate_model(model, loader, criterion, dataset_name="Test"):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    # Format metrics as a list
    metrics_list = [
        f'{dataset_name} Accuracy: {accuracy:.4f}',
        f'{dataset_name} Precision (Macro): {precision:.4f}',
        f'{dataset_name} Recall (Macro): {recall:.4f}',
        f'{dataset_name} F1-score (Macro): {f1:.4f}',
        f'{dataset_name} Precision (Micro): {micro_precision:.4f}',
        f'{dataset_name} Recall (Micro): {micro_recall:.4f}',
        f'{dataset_name} F1-score (Micro): {micro_f1:.4f}',
        f'{dataset_name} Loss: {total_loss / len(loader):.4f}'
    ]

    for metric in metrics_list:
        print(f'- {metric}')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Display the numbers in the confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return total_loss / len(loader)

# Path to the best model
best_model_path = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\All models-part3\main_model\best_model.pth'

# Load the best model
load_best_model(model, best_model_path)

# Evaluate on the test set
evaluate_model(model, test_loader, criterion, dataset_name="Test")
