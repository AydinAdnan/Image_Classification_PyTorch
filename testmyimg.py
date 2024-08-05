import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#without using try except block 
#if you trained your model in gpu env then running again required cpu so in the loading command
#make sure you set the map location to cpu if gpu not available
#model.load_state_dict(torch.load("newmodel.pth", map_location=torch.device('cpu')))


# Define a convolution neural network
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

# Define the classes
classes = ('plane', 'car', 'bird', 'poocha', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

# Function to test the model with a custom image
def test_custom_image(image_path):
    # Define the device
    # Load and preprocess the image
    image = load_image(image_path).to(device)

    # Load the trained model
    model = Network()
    model_path="newmodel.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model Successfully loaded")
    except Exception as e:
        print("Fix your code bro. Might still run tho")
    model.to(device)
    model.eval()

    # Predict the class
    with torch.no_grad():
        output = model(image)  # Use singular 'image' here
        _, predicted = torch.max(output, 1)

    # Decode and print the prediction
    predicted_class = classes[predicted.item()]
    print('Predicted class:', predicted_class)

# Path to the custom image
image_path = 'D:\imgclassif\c.jpeg'

# Test the model with the custom image
test_custom_image(image_path)
