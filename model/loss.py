import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy(y_true, y_pred):
    '''
    Calculates the cross entropy between a known label ([1,0,0],[0,1,0],or [0,0,1] in 3 class case)
    and a label predicted by the model of the same dimension. Both are torch tensors and the return a tensor of loss.
    '''
    # Apply log to the predictions because nn.NLLLoss expects log probabilities
    y_pred_log = torch.log(y_pred)
    return F.nll_loss(y_pred_log, y_true)



def sort_with_index(lst):
    '''
    Takes a list of numbers and sorts them in descending order. The returned list
    is this sorted list but the elements are tuples of 2 where the first element is
    the index of the second in the original list

    Ex.
    x=[1,4,2,6,8]; indices are [0,1,2,3,4]

    sort_with_index(x) -> [(4,8),(3,6),(1,4),(2,2),(0,1)]
    '''
    # Enumerate the list to create tuples of (index, element)
    indexed_list = list(enumerate(lst))

    # Sort the indexed list based on the second element (the actual value)
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

    # index is the first element in each tuple
    return sorted_list


def top_k(lst,k):
    '''
    Returns a list of length k. These contains the indices of the largest k elements in lst (another list).

    Ex. 
    x=[1,4,2,6,8]; indices are [0,1,2,3,4]
    top_k(x,3) -> [4,3,1]
    '''
  sorted_lst=sort_with_index(lst)
  rank_lst=[]
  for i in range(k):
    rank_lst.append(sorted_lst[i][0])
  return rank_lst
 

def adaptive_clustering(B,bt,k):
    '''
    Takes an instance of the MemoryBank class B which is essentially a torch tensor of previous prediction 
    vectors (or tensors) and determines the adaptive_clustering loss with current batch of prediction vectors, bt.
    It also takes k in relation to top_k where each sample from B and bt have a similarity label assigned based on the 
    indices of the top k elements. If the top indices match then the similarity label is 1; otherwise 0. This is just an
    implementation of binary cross entropy between current batch and all previous predictions. Returns a torch tensor of loss.
    '''
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
        score=torch.dot(B[i].clone(), bt[j].clone())
        
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
    '''
    Calculates the entropy separation loss. Here rho is a target entropy and m is a threshold 
    value for the absolute value of the difference between the entropy calculated from one of the
    N predictors in y_pred and rho. y_pred is a torch tensor of size (N,#classes) #classes will be 3 for it's use.
    The return is a torch tensor of the loss.
    '''
    # Calculate entropy of the predictions
    p_log_p = y_pred * torch.log(y_pred + 1e-9)  # add a small constant to avoid log(0)
    entropy = -p_log_p.sum(dim=1)

    # Calculate absolute difference from rho
    diff = (entropy - rho).abs()

    # Apply threshold
    loss = torch.where(diff > m, -diff, torch.tensor(0.0, device=diff.device))

    return loss.mean()

def combined_loss(y_true, y_pred, weight, rho, m, B, bt, k):
    '''
    Combines CE, AC, and ES loss into one function with an additional argument weight.
    This weights the sum of ES and AC loss when added to CE loss.
    Return is still a torch tensor.
    '''
    total_entropy = cross_entropy(y_true, y_pred) + weight*adaptive_clustering(B, bt, k) + weight*entropy_separation(y_pred, rho, m)
    return total_entropy, cross_entropy(y_true, y_pred), adaptive_clustering(B, bt, k), entropy_separation(y_pred, rho, m)
