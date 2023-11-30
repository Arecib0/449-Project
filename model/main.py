# Raw copilot output for using resnet50 below

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from load import load_and_preprocess_data, load_labels
from loss import combined_loss

# Load data
data = load_and_preprocess_data('Data\images_Y10_test_150.npy')
# Assumes the data is of the shape (num_images, height, width, channels)
# ResNet50 expects image that are 224x224

# Load labels
labels=load_labels(file_path)

# Load ResNet50 model without top layer
base_model = models.resnet50(pretrained=True)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, 3)  # replace 3 with your number of classes

# Define the loss function and optimizer
criterion = combined_loss
optimizer = optim.Adam(base_model.parameters())

# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    optimizer.zero_grad()  # zero the parameter gradients

    # forward + backward + optimize
    outputs = base_model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print('Finished Training')