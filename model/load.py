import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses data from filepath
    :param filepath: Path to data
    :return: Preprocessed data in form of PyTorch tensor of shape (N, C, H, W)
    """
    data = np.load(filepath)
    data = data.transpose(0, 2, 3, 1)

    # Define the preprocessing steps
    preprocess = Compose([
        Resize((224, 224)),  # Resize images to 224x224
        ToTensor(),  # Convert to PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize data
    ])
        # Apply the preprocessing steps to each image
    data = torch.stack([preprocess(Image.fromarray(img.astype('uint8'))) for img in data])
    # Change data from (N, H, W, C) to (N, C, H, W)
    data = data.transpose(1, 3).transpose(1, 2)

    return data

def load_labels(filepath):
    """
    Loads labels from filepath in a one-hot encoded format (i.e. [0, 1, 0] for class 2)
    
    :param filepath: Path to labels
    :return: Labels in form of PyTorch tensor of shape (N,)
    """
    labels = np.load(filepath) 
    labels = torch.tensor(labels, dtype=torch.long)
    labels = torch.argmax(labels, dim=1)  # Convert one-hot encoding to class number
    
    return labels

