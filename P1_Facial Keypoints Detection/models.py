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
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Output Size = ((W - F) / S) + 1 = (224 - 5) / 1 + 1 = 220 => (32, 220, 220)
        # After Pooling Output => (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Output Size = ((W - F) / S) + 1 = (110 - 5) /1 + 1 = 106 => (64, 106, 106)
        # After Pooling Output => (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # Output Size = ((W - F) / S) + 1 = (53 - 5)/1 + 1 = 49 => (128, 49, 49)
        # After Pooling Output => (128, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # Fully-Connected Layer
        self.fc = nn.Linear(128*24*24, 136)
        
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
        ## Flattening
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
