# Raw copilot output for using resnet50 below

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from load import load_and_preprocess_data, load_labels
from loss import combined_loss
from test import test_model
from argument_parser import create_arg_parser

def main(args):
    # Load data
    train_data = load_and_preprocess_data(args.data_path)
    # Assumes the data is of the shape (num_images, height, width, channels)
    # ResNet50 expects image that are 224x224

    # Load labels
    train_labels = load_labels(args.labels_path)

    # Create a DataLoader for your training data
    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    # Load ResNet50 model without top layer
    base_model = models.resnet50(pretrained=True)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, args.num_classes)  # replace 3 with your number of classes

    # Define the loss function and optimizer
    criterion = combined_loss
    optimizer = optim.SGD(base_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    # Train the model
    for epoch in range(10):  # Assuming you want to train for 10 epochs
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Finished Training')
    # Save the model
    torch.save(base_model.state_dict(), 'trained_model.pt')

    # Test the model
    test_model(base_model)  # Uncomment this line to test your model

    if __name__ == 'main':
        parser=create_arg_parser()
        args=parser.parse_args()
        main(args)