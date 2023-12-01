import torch
from torch.utils.data import DataLoader, TensorDataset
from load import load_and_preprocess_data, load_labels

# Load and preprocess test data and labels
test_data = load_and_preprocess_data('Data\images_Y10_test_150.npy')
test_labels = load_labels('Data\labels_test_150.npy')

# Create a DataLoader for your test data
test_dataset = TensorDataset(test_data, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=32)

def test_model(model):
    # Set the model to evaluation mode
    model.eval()

    # Initialize counters
    correct = 0
    total = 0

    # No need to track gradients for testing, so wrap in 
    # no_grad to save memory
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # Forward pass
            outputs = model(inputs)

            # Get prediction
            _, predicted = torch.max(outputs.data, 1)

            # Update total
            total += labels.size(0)

            # Update correct
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: {}%'.format(100 * correct / total))