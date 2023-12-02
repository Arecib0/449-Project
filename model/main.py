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
from plot import plotLoss
from torch.optim.lr_scheduler import StepLR
from feature_extractor import FeatureExtractor
from memory import MemoryBank


def main(args):
    # Load data
    train_data = load_and_preprocess_data(args.data_path)
    target_data = load_and_preprocess_data("Data\images_Y1_test_150.npy")
    # Assumes the data is of the shape (num_images, height, width, channels)
    # ResNet50 expects image that are 224x224

    # Load labels
    train_labels = load_labels(args.labels_path)

    validation_data = load_and_preprocess_data(args.validation_path)
    validation_labels = load_labels(args.validation_labels_path)


    # Create a DataLoader for your training data
    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    # Create a DataLoader for your validation data
    val_dataset = TensorDataset(validation_data, validation_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Create a DataLoader for your target training data
    target_train_dataset = TensorDataset(target_data, train_labels)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=32)

    # Load ResNet50 model without top layer
    # I'm setting pretrained to False because I believe that the paper did not use a pretained model
    # If we need to, we can re-enable this later
    base_model = models.resnet50(pretrained=False)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, args.num_classes)  # replace 3 with your number of classes

    # Define the loss function and optimizer
    criterion = lambda outputs, labels: combined_loss(labels, outputs, args.loss_weight, args.rho, args.m)
    optimizer = optim.SGD(base_model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    # Define the scheduler
    # This will decrease the learning rate by a factor of 0.1 every 10 epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize variables for early stopping
    best_accuracy = 0.0
    epochs_no_improve = 0

    # Initialize lists to store the losses
    ce_losses = []
    ac_losses = []
    es_losses = []

    # Initialize the feature extractor and memory bank
    feature_extractor = FeatureExtractor(base_model)
    memory_bank = MemoryBank(1000, 2048, device='cpu')

    # Train the model
    for epoch in range(args.epochs):  # Assuming you want to train for 10 epochs
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss, ce_loss, ac_loss, es_loss = criterion(outputs, labels)  
            
            # Save the losses
            ce_losses.append(ce_loss.item())
            ac_losses.append(ac_loss.item())
            es_losses.append(es_loss.item())

            loss.backward()
            optimizer.step()

            # Update the memory bank
            with torch.no_grad():
                features = feature_extractor(inputs)
                memory_bank.update(features, labels)

        scheduler.step() # Update the learning rate
        # Evaluate on the validation set
        base_model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_dataloader:
                outputs = base_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
        # Check for improvement
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == 12:
            print('Early stopping!')
            break

        # Switch back to training mode
        base_model.train()

    # Adapt the model to the target domain
    for epoch in range(args.epochs):  
        for inputs, labels in target_train_dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)

            # Compute similarity
            features = feature_extractor(inputs)
            similarity = memory_bank.compute_similarity(features)

            # Compute loss with similarity
            loss = criterion(outputs, labels, similarity)  
            loss.backward()
            optimizer.step()
    
    print('Finished Training')

    # Save Loss
    plotLoss(ce_losses, ac_losses, es_losses)

    # Save the model
    # torch.save(base_model.state_dict(), 'trained_model.pt')

    # Test the model
    ## test_model(base_model)  # Uncomment this line to test your model

if __name__ == 'main':
    parser=create_arg_parser()
    args=parser.parse_args()
    main(args)