import matplotlib.pyplot as plt
import numpy as np


def plotLoss(ce_losses, ac_losses, es_losses):
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