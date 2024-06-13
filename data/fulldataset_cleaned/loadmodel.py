import torch
import torch.nn as nn

# Step 1: Define Your Model Class matching the saved model's architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your layers here to match the saved model
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjust the input size to 1600 (64 * 5 * 5)
        self.fc2 = nn.Linear(128, 4)  # Adjust the output size based on your specific use case

    def forward(self, x):
        # Define the forward pass to match the saved model's forward pass
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)  # Flatten the tensor with 1600 (64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Load the Model
def load_model(model_path, use_state_dict=True):
    if use_state_dict:
        # Create an instance of your model
        model = MyModel()
        
        # Load the state dictionary from the .pth file
        model.load_state_dict(torch.load(model_path))
    else:
        # Load the entire model directly
        model = torch.load(model_path)

    # Set the model to evaluation mode (important for inference)
    model.eval()
    return model

# Step 3: Use the Model for Inference
def infer(model, input_tensor):
    # Get the model's prediction
    output = model(input_tensor)
    return output

# Example usage
if __name__ == "__main__":
    # Correctly format the model path
    model_path = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\best_model.pth'
    
    # Choose whether you're loading a state dictionary or an entire model
    use_state_dict = True  # Set to False if loading an entire model

    # Load the model
    model = load_model(model_path, use_state_dict)

    # Example input tensor (replace with your actual input)
    # Adjust the size to match the expected input size of your model
    input_tensor = torch.randn(1, 1, 28, 28)  # Example for a model expecting 28x28 images

    # Perform inference
    output = infer(model, input_tensor)

    print("Model output:", output)
