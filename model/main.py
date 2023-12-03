# Raw copilot output for using resnet50 below

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from load import load_and_preprocess_data, load_labels
from loss import *
from test import test_model
from plot import plotLoss, plotAccuracy
from torch.optim.lr_scheduler import StepLR
from feature_extractor import FeatureExtractor
from memory import MemoryBank
import yaml


def main(args):
    # Load configuration
    with open('Data/Config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    batch_size = config['batch_size']
    num_classes = config['num_classes']
    # Load data
    train_data = load_and_preprocess_data(config['train_data_path'])
    target_data = load_and_preprocess_data(config['target_data_path'])
    # Assumes the data is of the shape (num_images, height, width, channels)
    # ResNet50 expects image that are 224x224

    # Load labels
    train_labels = load_labels(config['train_labels_path'])

    validation_data = load_and_preprocess_data(config['validation_data_path'])
    validation_labels = load_labels(config['validation_labels_path'])
    validation_target_data=load_and_preprocess_data('Data\images_Y1_valid.npy')
    validation_target_labels=load_labels('Data\labels_valid.npy')


    # Create a DataLoader for your training data
    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size)

    # Create a DataLoader for your validation data
    val_dataset = TensorDataset(validation_data, validation_labels)
    val_dataloader = DataLoader(val_dataset, batch_size)
    val_target_dataset=TensorDataset(validation_target_data,validation_target_labels)
    val_target_dataloader=DataLoader(val_target_dataset,batch_size=32)

    # Create a DataLoader for your target training data
    target_train_dataset = TensorDataset(target_data, train_labels)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size)

    # Load ResNet50 model without top layer
    # I'm setting pretrained to False because I believe that the paper did not use a pretained model
    # If we need to, we can re-enable this later
    base_model = models.resnet50(pretrained=False)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, num_classes)  # replace 3 with your number of classes

    # Define the loss function and optimizer
    criterion = lambda outputs, labels: combined_loss(labels, outputs, config['loss_weight'], config['rho'], config['m'])
    optimizer = optim.SGD(base_model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], nesterov=True)

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
    class_accuracies = [[] for _ in range(num_classes)]

    # Initialize the feature extractor and memory bank
    # feature_extractor = FeatureExtractor(base_model)
    # memory_bank = MemoryBank(1000, 2048, device='cpu')
    # The 1000 is the size of the memory bank, which is a hyperparameter
    # and determines how many images are stored in the memory bank.
    # Essentially, it's how much memory to allocate for storing the features.
    # I'm not sure what number would be best for your machine when you're
    # training the model, since I don't know how much memory you have.
    # If you run out of memory, you can try reducing this number.
    # But honestly, I don't think it will be a problem.
    # If anything, we'll more likely have to increase this number.

    # Create a separate memory bank for the output vectors
    output_memory_bank = MemoryBank(1000, num_classes, device='cpu')
    # label_memory_bank = MemoryBank(1000, args.num_classes, device='cpu')

    # Train the model
    for epoch in range(config['source_epochs']):  # Assuming you want to train for 10 epochs
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss, ce_loss, ac_loss, es_loss = criterion(outputs, labels)  
            
            # Save the losses

            # When the adaptive clustering loss is implemented,
            # for the labeled source data the class labels are
            # used to generate the similarity labels rather than
            # the output prediction vectors.
            ce_losses.append(ce_loss.item())
            ac_losses.append(ac_loss.item())
            es_losses.append(es_loss.item())

            loss.backward()
            optimizer.step()

            # Update the memory bank
            with torch.no_grad():
                # features = feature_extractor(inputs)
                # memory_bank.update(features, labels)
                output_memory_bank.update(outputs.detach, labels)

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
    for epoch in range(config['target_epochs']):
        for inputs, labels in target_train_dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            criterion1=adaptive_clustering()
            criterion2=entropy_separation()
            ac_loss=criterion1(output_memory_bank.bank, outputs, 3)
            es_loss=criterion2(outputs, args.rho, args.m)

            # Compute similarity
            # features = feature_extractor(inputs)
            similarity = output_memory_bank.compute_similarity(outputs.detach())

            # Update the memory bank
            with torch.no_grad():
                # features = feature_extractor(inputs)
                # memory_bank.update(features, labels)
                output_memory_bank.update(outputs)

            # Compute loss with similarity
            # The similarity is used in the adaptive clustering loss
            # which is at this point unwritten.
            # When Eric writes the adaptive clustering loss, he can
            # use this similarity to compute the loss.
            # I believe that this, combined with the entropy separation
            # loss, is the primary domain adaptation method used in the paper.
            loss = ac_loss + es_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct = [0]*num_classes
            total = [0]*num_classes
            for inputs, labels in val_dataloader:
                outputs = base_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
            for i in range(num_classes):
                correct[i] += (predicted[labels == i] == labels[labels == i]).sum().item()
                total[i] += (labels == i).sum().item()
        accuracies = [correct[i] / total[i] if total[i] > 0 else 0 for i in range(num_classes)]
        class_accuracies.append(accuracies)
    
    print('Finished Training')

    # Save Loss and Accuracy Plots
    plotLoss(ce_losses, ac_losses, es_losses)
    plotAccuracy(class_accuracies, num_classes)

    # Save the model
    # torch.save(base_model.state_dict(), 'trained_model.pt')

    # Test the model
    ## test_model(base_model)  # Uncomment this line to test your model
