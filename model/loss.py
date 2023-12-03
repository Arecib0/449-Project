import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy(y_true, y_pred):
    # Apply log to the predictions because nn.NLLLoss expects log probabilities
    y_pred_log = torch.log(y_pred)
    return F.nll_loss(y_pred_log, y_true)


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
  count=0
  eps=1e-9

  # main loop to calculate loss
  for i in range(len(B)):
    top_ki=top_k(B[i],k)
    for j in range(len(bt)):## could use batch size instead of len(bt)
        top_kj=top_k(bt[j],k)
        if top_ki==top_kj:
            sij=1
        else:
            sij=0
      
        # Calculate the dot product of the vectors
        score=torch.dot(B[i], bt[j])
        
        if score!=0:
            Loss-=(sij*torch.log(score+eps)+(1-sij)*torch.log(1-score+eps))
            count+=1
        
        if score<0:
           print('You have a negative score. In theory, this should not happen.')
           print("if you're seeing this, something has gone very wrong.")
           print('Score:', score)
  
  # Calculate the mean loss
  if count > 0:
      mean_loss = Loss / count
  else:
      mean_loss = Loss
  
  return mean_loss



def entropy_separation(y_pred, rho, m):
    # Assumes that y_pred is the raw output of the model
    # and that y_pred is a 2D tensor of shape (N, num_classes)
    # In this case, the 2nd dimension corresponds to raw output for each class
    # These raw outputs are converted to probabilities using softmax 
    # in this function, so you don't need to apply softmax in the last layer of the model

    # Apply softmax to convert raw output to probabilities
    # y_pred = torch.softmax(y_pred, dim=1)

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
