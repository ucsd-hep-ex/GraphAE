import torch

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """
        :param data: torch tensor
        """
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, log_pt=False):
        """
        :param data: torch tensor
        :param log_pt: undo log transformation on pt
        """
        inverse = (data * self.std) + self.mean
        if log_pt:
            inverse[:,0] = (10 ** inverse[:,0]) - 1
        return inverse
