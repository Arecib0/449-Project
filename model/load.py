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
    data = np.load(filepath) # loaded data in form (N,C,H,W)=(#samples,colour chans,img height,img width)
    # dim0=N, dim1=C, dim2=H, dim3=W
    data = data.transpose(0, 2, 3, 1) # (N,C,H,W)->(N,H,W,C) to be correct format in preprocessing below
    #(dim0,dim1,dim2,dim3)->(dim0,dim2,dim3,dim1)
    
    # Define the preprocessing steps
    preprocess = Compose([
        Resize((224, 224)),  # Resize images to 224x224
        ToTensor(),  # Convert to PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize data
    ]) # normalize basically scales RGB channels using values determined from a large collection of images. Makes z-score
        # Apply the preprocessing steps to each image
    data = torch.stack([preprocess(Image.fromarray(img.astype('uint8'))) for img in data])
    # Change data from (N, H, W, C) to (N, C, H, W); the original format
    data = data.permute(0, 1, 3, 2)
    print(data.shape)

    return data

def load_labels(filepath):
    """
    Loads labels from filepath in a one-hot encoded format (i.e. [0, 1, 0] for class 2).
    Possible labels are [1,0,0],[0,1,0],[0,0,1] for the 3 class problem.
    
    :param filepath: Path to labels
    :return: Labels in form of PyTorch tensor of shape (1,N). The ith entry is the index
    in the ith sample with 1 as its entry.

    Ex.
    labels=np.load()=[[1,0,0],[0,1,0],[0,0,1]] -> torch([0,1,2])
    
    """
    labels = np.load(filepath) 
    labels = torch.tensor(labels, dtype=torch.long)
    labels = torch.argmax(labels, dim=1)  # Convert one-hot encoding to class number 0,1, or 2 in 3 class problem
    print(labels.shape)

    return labels

