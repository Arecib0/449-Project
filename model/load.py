import numpy as np
import torch

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses data from filepath
    :param filepath: Path to data
    :return: Preprocessed data in form of PyTorch tensor of shape (N, C, H, W)
    """
    data_list = np.load(filepath)
    data = np.stack(data_list, axis=0)  # Convert list of arrays to array of arrays
    data = np.transpose(data, (0, 3, 1, 2))  # Change data from (N, H, W, C) to (N, C, H, W)
    data = data / 255.0  # Normalize data
    data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor

    return data

def load_labels(filepath):
    """
    Loads labels from filepath in a one-hot encoded format (i.e. [0, 1, 0] for class 2)
    
    :param filepath: Path to labels
    :return: Labels in form of PyTorch tensor of shape (N, C)
    """
    labels = np.load(filepath) 
    labels = torch.tensor(labels, dtype=torch.long)
    
    return labels

