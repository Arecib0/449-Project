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
from memory import MemoryBank
import yaml


def main():
    torch.autograd.set_detect_anomaly(True)
    
    # Bring in hyperparameters and pathes
    with open('Data/Config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    batch_size = config['batch_size']
    num_classes = config['num_classes']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data that can be fed into Resnet50
    train_data = load_and_preprocess_data(config['train_data_path'])
    target_data = load_and_preprocess_data(config['target_data_path'])
    # Assumes the data is of the shape (num_images, height, width, channels)
    # ResNet50 expects image that are 224x224

    # Load labels
    train_labels = load_labels(config['train_labels_path'])

    # Load validation data
    validation_data = load_and_preprocess_data(config['validation_data_path'])
    validation_labels = load_labels(config['validation_labels_path'])
    validation_target_data=load_and_preprocess_data(config['validation_target_data_path'])
    validation_target_labels=load_labels(config['validation_target_labels_path'])


    # Establish a training dataset and create a DataLoader for your training data
    train_dataset = TensorDataset(train_data.to(device), train_labels.to(device))
    train_dataloader = DataLoader(train_dataset, batch_size)

    # Establish validation dataset and create a DataLoader for your validation data
    val_dataset = TensorDataset(validation_data.to(device), validation_labels.to(device))
    val_dataloader = DataLoader(val_dataset, batch_size)
    val_target_dataset=TensorDataset(validation_target_data.to(device),validation_target_labels.to(device))
    val_target_dataloader=DataLoader(val_target_dataset, batch_size)

    # Create a DataLoader for your target training data
    target_train_dataset = TensorDataset(target_data.to(device), train_labels.to(device))
    target_train_dataloader = DataLoader(target_train_dataset, batch_size)
    
    # Load ResNet50 model without top layer
    # I'm setting pretrained to False because I believe that the paper did not use a pretained model
    # This section removes the fully connected layer in ResNet50 and replaces it with a new linear layer followed
    # by a softmax. The linear function now ouputs a tensor with k=#classes spaces.
    base_model = models.resnet50(weights=None)
    base_model = base_model.to(device)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),  # replace num_classes with your number of classes
        nn.Softmax(dim=1))

    # Define the loss function and optimizer
    criterion = lambda outputs, labels, memory: combined_loss(labels, outputs, config['loss_weight'], config['rho'], config['m'], memory, outputs, num_classes)
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
    class_accuracies = []

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

    # Train the model
    print("Starting Training")
    for epoch in range(config['source_epochs']):  
        print(epoch)
        epoch_ce_loss = []
        epoch_ac_loss = []
        epoch_es_loss = []
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss, ce_loss, ac_loss, es_loss = criterion(outputs, labels, memory=output_memory_bank.bank)  
            
            # Save the losses

            # When the adaptive clustering loss is implemented,
            # for the labeled source data the class labels are
            # used to generate the similarity labels rather than
            # the output prediction vectors.
            epoch_ce_loss.append(ce_loss.item())
            epoch_ac_loss.append(ac_loss.item())
            print(ac_loss.item())
            epoch_es_loss.append(es_loss.item())

            # backpropogation and gradient step
            loss.backward()
            optimizer.step()

            # Update the memory bank
            with torch.no_grad():
                output_memory_bank.update(outputs.detach(), labels)
        
        # append losses to their respective lists
        ce_losses.append(sum(epoch_ce_loss) / len(epoch_ce_loss))
        ac_losses.append(sum(epoch_ac_loss) / len(epoch_ac_loss))
        es_losses.append(sum(epoch_es_loss) / len(epoch_es_loss))
        
        scheduler.step() # Update the learning rate

    # Adapt the model to the target domain
    print("Begining target domain adaptation")
    for epoch in range(config['target_epochs']):
        print(epoch)
        # Established two lists of length num_classes which have all zeros.
        # Each entry in the list is for a different class. So if num_classes=3
        # and the first class is observed, total[0]+=1. If the prediction of 
        # the model is correct correct[0]+=1.
        correct = [0]*num_classes
        total = [0]*num_classes
        
        # cross entropy left out because now training on unlabelled data
        for inputs, labels in target_train_dataloader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            ac_loss=adaptive_clustering(output_memory_bank.bank, outputs, num_classes)
            es_loss=entropy_separation(outputs, config['rho'], config['m'])

            # Update the memory bank
            # no labels this time so only updating ouput_memory_bank
            with torch.no_grad():
                output_memory_bank.update(outputs)

            # Compute loss with similarity
            # The similarity is used in the adaptive clustering loss
            # which is at this point unwritten.
            # When Eric writes the adaptive clustering loss, he can
            # use this similarity to compute the loss.
            # I believe that this, combined with the entropy separation
            # loss, is the primary domain adaptation method used in the paper.
            loss = ac_loss + es_loss

            # backprop and gradient step
            loss.backward()
            optimizer.step()

            scheduler.step() # Update the learning rate
            _, predicted = torch.max(outputs.data, 1) # gives max val of predictor and its index
            # increments the correct entry of correct if the prediction was right
            # increments 
            for i in range(num_classes):
                correct[i] += (predicted[labels == i] == labels[labels == i]).sum().item()
                total[i] += (labels == i).sum().item()
            

        # add accuracies to class_accuracies list
        with torch.no_grad():
            accuracies = [correct[i] / total[i] if total[i] > 0 else 0 for i in range(num_classes)]
            if len(accuracies) < num_classes:
                accuracies += [0] * (num_classes - len(accuracies))
            class_accuracies.append(accuracies)
            print(class_accuracies)


    
    print('Finished Training')

    # Save Loss and Accuracy Plots
    plotLoss(ce_losses, ac_losses, es_losses)
    plotAccuracy(class_accuracies, num_classes)

    # Save the model
    torch.save(base_model.state_dict(), 'trained_model.pt')

    # Test the model
    ## test_model(base_model)  # Uncomment this line to test your model
if __name__ == '__main__':
    main()
