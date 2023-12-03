import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true)
    # Note: F.cross_entropy combines log_softmax and nll_loss
    # Because of this, y_pred should be the raw output of the model
    # Do not apply softmax in the last layer of the model
    # It also expects y_true to be a 1D tensor of class indices


# Takes a list of numbers and sorts them in descending order. The returned list
# is this sorted list but the elements are tuples of 2 where the first element is
# the index of the second in the original list
def sort_with_index(lst):
    # Enumerate the list to create tuples of (index, element)
    indexed_list = list(enumerate(lst))

    # Sort the indexed list based on the second element (the actual value)
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

    # index is the first element in each tuple
    return sorted_list

# returns the list of the indices for the top k elements
def top_k(lst,k):
  sorted_lst=sort_with_index(lst)
  rank_lst=[]
  for i in range(k):
    rank_lst.append(sorted_lst[i][0])
  return rank_lst
 

# Takes the memory of all previous vectors of the CNN and those in the current batch.
# it uses this these with the indices to be compared and returns the binary cross entropy
# loss.

def adaptive_clustering(B,bt,k):
  Loss=torch.zeros(1)
  

  # main loop to calculate loss
  for i in range(len(B)):
    top_ki=top_k(B[i],k)
    for j in range(len(bt)):## could use batch size instead of len(bt)
        top_kj=top_k(bt[j],k)
        if top_ki==top_kj:
            sij=1
        else:
            sij=0
      
        score=torch.dot(B[i].clone(),bt[j].clone())
        
        if score!=0:
            Loss-=(sij*torch.log(score)+(1-sij)*torch.log(1-score))
  
  return Loss



def entropy_separation(y_pred, rho, m):
    # Assumes that y_pred is the raw output of the model
    # and that y_pred is a 2D tensor of shape (N, num_classes)
    # In this case, the 2nd dimension corresponds to raw output for each class
    # These raw outputs are converted to probabilities using softmax 
    # in this function, so you don't need to apply softmax in the last layer of the model

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

def combined_loss(y_true, y_pred, weight, rho, m, B, bt, k):
    total_entropy = cross_entropy(y_true, y_pred) + weight*adaptive_clustering(B, bt, k) + weight*entropy_separation(y_pred, rho, m)
    return total_entropy, cross_entropy(y_true, y_pred), adaptive_clustering(B, bt, k), entropy_separation(y_pred, rho, m)
