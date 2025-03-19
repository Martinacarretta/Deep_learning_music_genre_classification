import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

# class ConvNet(nn.Module):
#     def __init__(self, kernels, classes=8):
#         super(ConvNet, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, kernels[0], kernel_size=5, stride=1, padding=2),  #  3 input channels (RGB), kernels[0] output channels
#             nn.BatchNorm2d(kernels[0]),  # Bstch normalization to improve faster convergence and reducing overfitting
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.25)  # Add dropout with 25% probability
#         )
        
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),  # kernels[0] input channels, kernels[1] output channels
#             nn.BatchNorm2d(kernels[1]),  # Add batch normalization
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.25)  # Add dropout with 25% probability
#         )

#         # Fully connected layer
#         self.fc = nn.Linear(56 * 56 * kernels[1], classes)  # Flatten the output and apply fully connected layer
        
#     def forward(self, x):
#         out = self.layer1(x)  # Pass input through first layer
#         out = self.layer2(out)  # Pass output through second layer
#         out = out.reshape(out.size(0), -1)  # Flatten the output for the fully connected layer
#         out = self.fc(out)  # Pass through fully connected layer
#         return out  # Return the final output


# class ResNet50(nn.Module):
#     def __init__(self, num_classes=8):
#         super(ResNet50, self).__init__()
#         self.model = models.resnet50(pretrained=True)
#         self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.model.fc = nn.Linear(in_features=2048, out_features=num_classes)

#     def forward(self, x):
#         return self.model(x)
#         '''
        
# '''
# class MusicGenreCNN(nn.Module):
#     def __init__(self, num_classes=8, num_input_channels=3):
#         super(MusicGenreCNN, self).__init__()
#         self.conv1 = nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjusted input size for 224x224 spectrograms
#         self.fc2 = nn.Linear(512, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = self.pool(nn.functional.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 28 * 28)  # Adjusted input size for 224x224 spectrograms
#         x = self.dropout(x)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# class ImprovedMusicGenreCNN(nn.Module):
#     def __init__(self, num_classes=8, num_input_channels=3):
#         super(ImprovedMusicGenreCNN, self).__init__()
#         self.conv1 = nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(256 * 14 * 14, 512)
#         self.fc2 = nn.Linear(512, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = x.view(-1, 256 * 14 * 14)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class ConvNetIMPROV(nn.Module):
#     def __init__(self, kernels, classes=8):
#         super(ConvNetIMPROV, self).__init__()

#         # Ensure kernels list has enough elements
#         assert len(kernels) >= 3, "The kernels list must have at least 3 elements."

#         # Layer 1
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, kernels[0], kernel_size=3, stride=1, padding=1),  # Changed kernel size to 3
#             nn.BatchNorm2d(kernels[0]),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.25)
#         )

#         # Layer 2
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(kernels[0], kernels[1], kernel_size=3, stride=1, padding=1),  # Changed kernel size to 3
#             nn.BatchNorm2d(kernels[1]),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.25)
#         )

#         # Layer 3
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(kernels[1], kernels[2], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(kernels[2]),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.25)
#         )

#         # Calculate the size of the output from the last convolutional layer
#         self._to_linear = None
#         self.convs = nn.Sequential(self.layer1, self.layer2, self.layer3)
#         self._get_conv_output_size()

#         # Fully connected layers
#         self.fc1 = nn.Linear(self._to_linear, 1024)  # Adjusted for additional layer
#         self.fc2 = nn.Linear(1024, classes)
        
#     def _get_conv_output_size(self):
#         # Dummy forward pass to calculate the size of the output from conv layers
#         x = torch.randn(1, 3, 224, 224)  # Use a dummy input with the same dimensions as your actual input
#         x = self.convs(x)
#         self._to_linear = x.numel()

#     def forward(self, x):
#         out = self.convs(x)
#         out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
#         out = self.fc1(out)
#         out = F.relu(out)
#         out = self.fc2(out)
#         return out


import torch.nn.functional as F

class ImprovedMusicGenreCNNv2(nn.Module): 
    def __init__(self, num_classes=8, num_input_channels=3):
        super(ImprovedMusicGenreCNNv2, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024 * 7 * 7, 2048)
        self.dropout = nn.Dropout(0.8)  # Decreased dropout rate
        self.dropout2 = nn.Dropout(0.7)  # Decreased dropout rate
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128)
        )
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256)
        )
        self.shortcut5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512)
        )
        self.shortcut6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024)
        )

    def forward(self, x):
        residual = self.shortcut1(x)
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        residual = self.pool(residual)  # Match dimensions
        x += residual

        residual = self.shortcut2(x)
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        residual = self.pool(residual)  # Match dimensions
        x += residual

        residual = self.shortcut3(x)
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        residual = self.pool(residual)  # Match dimensions
        x += residual

        residual = self.shortcut4(x)
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        residual = self.pool(residual)  # Match dimensions
        x += residual

        residual = self.shortcut5(x)
        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))
        residual = self.pool(residual)  # Match dimensions
        x += residual

        residual = self.shortcut6(x)
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        x += residual

        x = x.view(-1, 1024 * 7 * 7)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
