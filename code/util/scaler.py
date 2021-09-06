import torch

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
