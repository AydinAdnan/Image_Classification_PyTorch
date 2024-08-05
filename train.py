import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Instantiate a neural network model 
model = Network()

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Function to save the model
def saveModel():
    path = "./newmodel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy / total)
    return accuracy

# Training function with increased number of epochs
def train(num_epochs):
    best_accuracy = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        accuracy = testAccuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    model.eval()
    
    images, labels = next(iter(test_loader))

    imageshow(torchvision.utils.make_grid(images))
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),  # Added data augmentation
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                               shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    batch_size = 4

    # Let's build our model
    train(10)  # Increased number of epochs for better training
    print('Finished Training')

    # Test which classes performed well
    print('Test accuracy:', testAccuracy())
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "./newmodel.pth"
    model.load_state_dict(torch.load(path))
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Test with batch of images
    testBatch()
