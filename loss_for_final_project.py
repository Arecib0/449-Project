import torch
import torch.nn as nn
import numpy as np

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
  Loss=0

  # main loop to calculate loss
  for i in range(len(B)):
    top_ki=top_k(B[i],k)
    for j in range(len(bt)):## could use batch size instead of len(bt)
      top_kj=top_k(bt[j],k)
      if top_ki==top_kj:
        sij=1
      else:
        sij=0
      
      score=np.dot(B[i],bt[j])
      Loss-=(sij*np.log(score)+(1-sij)*np.log(1-score))
  
  return Loss

