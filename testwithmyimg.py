from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Define the classes for CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to load and preprocess the custom image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])
    image = Image.open(image_path)
    image = transform(image).float()
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Define the neural network model (same as train.py neural configuration)
# Define a convolution neural network with increased complexity
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)  # Added dropout layer

    def forward(self, input):
        output = F.elu(self.bn1(self.conv1(input)))      
        output = F.elu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.elu(self.bn3(self.conv3(output)))     
        output = F.elu(self.bn4(self.conv4(output)))
        output = self.pool(output)
        output = output.view(-1, 128 * 8 * 8)
        output = F.elu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        return output

# Instantiate the neural network model 
model = Network()

# Load the trained model state dict
model_path = r"newmodel.pth"  # Adjust the path accordingly
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")

# Set the model to evaluation mode
model.eval()

# Function to test the model with a custom image
def test_custom_image(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the image
    image = load_image(image_path).to(device)

    # Predict the class
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Decode and print the prediction
    predicted_class = classes[predicted.item()]
    print('Predicted class:', predicted_class)

# Path to the custom image
image_path = r'D:\imgclassif\img1.jpeg'  # Adjust the path accordingly

# Test the model with the custom image
test_custom_image(image_path)
