import numpy as np
import torch

def load_and_preprocess_data(filepath):
    data = np.load(filepath)
    data = np.transpose(data, (0, 3, 1, 2))  # Change data from (N, H, W, C) to (N, C, H, W)
    data = data / 255.0  # Normalize data
    data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor

    return data