import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, size, feature_dim, device):
        self.bank = torch.randn(size, feature_dim, device=device)
        self.labels = torch.empty(size, dtype=torch.long, device=device)
        self.ptr = 0

    def update(self, features, labels):
        with torch.no_grad():
            batch_size = features.size(0)
            self.bank[self.ptr:self.ptr+batch_size] = features
            self.labels[self.ptr:self.ptr+batch_size] = labels
            self.ptr = (self.ptr + batch_size) % self.bank.size(0)

    def compute_similarity(self, features):
        # Compute the dot product similarity
        similarity = torch.mm(features, self.bank.t())
        return similarity