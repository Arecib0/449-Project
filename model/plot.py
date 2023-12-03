import matplotlib.pyplot as plt


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
    class_names = ['Eliptical', 'Spiral', 'Merger']
    for i in range(num_class):
        plt.plot(class_acc[i], label=class_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Output/class_acc_plot.png')