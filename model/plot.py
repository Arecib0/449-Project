import matplotlib.pyplot as plt
import numpy as np


def plotLoss(ce_losses, ac_losses, es_losses):
        '''
        Generates a bunch of loss plots and saves them to the ouput folder.
        ce_losses is the cross entropy losss, ac_losses is binary cross entropy loss,
        and es_losses is entropy separation loss. These are losses over all the epochs used by the model.
        The respective names of these plots are: ce_loss_plot.png, ac_loss_plot.png, and
        es_loss_plot.png. A plot of all of them in the same figure is also generated and
        called loss_plot.png.
        '''
        # Combined Loss chart
    plt.figure(figsize=(10, 5))
    plt.plot(ce_losses, label='CE')
    plt.plot(ac_losses, label='AC')
    plt.plot(es_losses, label='ES')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Output/loss_plot.png')
    plt.close()

    # Plot and save the CE loss
    plt.figure(figsize=(10, 5))
    plt.plot(ce_losses, label='CE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Output/ce_loss_plot.png')
    plt.close()

    # Plot and save the AC loss
    plt.figure(figsize=(10, 5))
    plt.plot(ac_losses, label='AC')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Output/ac_loss_plot.png')
    plt.close()

    # Plot and save the ES loss
    plt.figure(figsize=(10, 5))
    plt.plot(es_losses, label='ES')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Output/es_loss_plot.png')
    plt.close()

def plotAccuracy(class_acc, num_class):
        '''
        Takes two arguments: class_acc which is a list of sublists with length equal to the number of classes and 
        num_class which is the number of classes in the classification problem. The ith 
        entry of the jth sublist is the accuracy of the network on the jth epoch for the ith class. num_class is a
        natural number.
        The plot is saved to the Output folder as class_acc_plot.png.
        '''
    # Assuming class_accuracies is a list of lists where each inner list contains the accuracies for each class for a single epoch
    print(class_acc)
    class_acc_np = np.array(class_acc)
    class_names = ['Elliptical', 'Spiral', 'Merger']
    # Get the number of classes and epochs
    # num_classes = class_accuracies_np.shape[1]
    # num_epochs = class_accuracies_np.shape[0]

    # Create a list of epoch numbers
    # epochs = list(range(num_epochs))

    # Plot the accuracy for each class
    for i in range(num_class):
         plt.plot(class_acc_np[:, i], label=class_names[i])

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Output/class_acc_plot.png')
