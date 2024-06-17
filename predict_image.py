import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(64 * 12 * 12, 128)
    self.fc2 = nn.Linear(128, 4)  # 4 classes: Angry, Engaged, Happy, Neutral
    self.pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.view(-1, 64 * 12 * 12)
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

class_names = ['Angry', 'Engaged', 'Happy', 'Neutral']


def load_best_model(model_path):

  if os.path.exists(model_path):
    model = SimpleCNN()  # Define the model here
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model weights from {model_path}")
    return model
  else:
    print(f"Error: Model weights not found at {model_path}")
    return None  # Indicate model loading failure


def predict_image(model, image_path, transform):

  model.eval()
  image = Image.open(image_path).convert('L')  # Convert image to grayscale
  image = transform(image).unsqueeze(0)  # Add batch dimension

  with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

  return class_names[predicted.item()]


# Define paths (modify as needed)
best_model_path = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\All models\main_model\best_model.pth'
image_path = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\data\fulldataset_cleaned\Angry\test\Angry_test_01.jpg'

# Load the best model (assuming the function returns the model)
loaded_model = load_best_model(best_model_path)
if loaded_model is not None:  # Check if model loaded successfully
  # Predict the class of the image
  predicted_class = predict_image(loaded_model, image_path, transform)
  print(f'The predicted class for the image is: {predicted_class}')
else:
  print("Failed to load model. Exiting...")
