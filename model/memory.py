import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, size, feature_dim, device):
        '''
        Instantiates a MemoryBank object with parameters self, size, feature_dim, device.
        Throughout this project device was often set to 'cpu'. Takes self, size which is the
        number vectors it can hold, and the dimension of those vectors is feature_dim.
        These dimension should be natural numbers.

        self.bank provices a torch tensor of zeros with dimensions sizexfeature_dim
        self.labels provides associated labels to the size number of samples in a torch tensor
        self.ptr initiated at 0. Points to where in the bank or labels a space is free
        '''
        # Established a bank of zeros
        self.bank = torch.zeros(size, feature_dim, dtype=torch.float, device=device)
        self.labels = torch.empty(size, dtype=torch.long, device=device)
        self.ptr = 0

    def update(self, features, labels=None):
        '''
        Takes self, a tensor of new features(tensors) and adds these to the bank (from self.bank). If labels is not None
        then the labels can also be added to labels (from self.labels). If the whole bank is full, starting a the oldest entries,
        these will be replaced as more features are added.
        '''
        with torch.no_grad():
            batch_size = features.size(0) # uses the number of features
            self.bank[self.ptr:self.ptr+batch_size] = features # replaces entries in bank with new features
            if labels is not None:
                self.labels[self.ptr:self.ptr+batch_size] = labels # replaces entries in labels  with new labels
            self.ptr = (self.ptr + batch_size) % self.bank.size(0) # adjusts pointer position 

