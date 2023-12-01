import torch
import torch.nn.functional as F

def cross_entropy(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true)
    # Note: F.cross_entropy combines log_softmax and nll_loss
    # Because of this, y_pred should be the raw output of the model
    # Do not apply softmax in the last layer of the model
    # It also expects y_true to be a 1D tensor of class indices


def adaptive_clustering(y_true, y_pred):
    # Replace this with your actual implementation
    return torch.mean(y_pred - y_true)

def entropy_separation(y_true, y_pred, rho, m):
    # Apply softmax to convert raw output to probabilities
    y_pred = torch.softmax(y_pred, dim=1)

    # Calculate entropy of the predictions
    p_log_p = y_pred * torch.log(y_pred + 1e-9)  # add a small constant to avoid log(0)
    entropy = -p_log_p.sum(dim=1)
    
    # Calculate mean entropy
    mean_entropy = entropy.mean()
    
    # Calculate absolute difference from rho
    diff = torch.abs(mean_entropy - rho)
    
    # Apply threshold
    loss = torch.where(diff > m, -diff, torch.zeros_like(diff))
    
    return loss.mean()

def combined_loss(y_true, y_pred):
    return cross_entropy(y_true, y_pred) + adaptive_clustering(y_true, y_pred) + entropy_separation(y_true, y_pred)