## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ## Input Image Size = 224 x 224
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        # Output Size = ((W - F) / S) + 1 = (224 - 4) / 1 + 1 = 220 => (32, 221, 221)
        # After Pooling Output => (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 4)
        # Output Size = ((W - F) / S) + 1 = (110 - 3) /1 + 1 = 106 => (64, 108, 108)
        # After Pooling Output => (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Output Size = ((W - F) / S) + 1 = (54 - 2)/1 + 1 = 53 => (128, 53, 53)
        # After Pooling Output => (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 2)
        # Output Size = ((W - F) / S) + 1 = (26 - 1)/1 + 1 = 20 => (256, 26, 26)
        # After Pooling Output => (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # Fully-Connected Layers
        self.fc1 = nn.Linear(256*13*13, 3000)
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        # MaxPool Layer
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout Layer
        self.dropout = nn.Dropout(0.2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        ## Flattening
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        #x = F.log_softmax(x, dim=1)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
