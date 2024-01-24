import torch
import torch.nn as nn
from torch import Tensor, tensor
from torch.nn import functional as F
import math
import copy


class MAE(nn.Module):
    """mean absolute error"""

    def __init__(self):
        super().__init__()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        return torch.mean((prediton - target).abs())


class MAE_2(nn.Module):
    """mean absolute error"""

    def __init__(self):
        super().__init__()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        return (prediton - target).abs()


class MAPE(nn.Module):
    """mean absolute percentage error"""

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        tabs = target.abs()
        tabs[tabs < self.epsilon] = self.epsilon
        return torch.mean((prediton - target).abs() / tabs)


class MAPE_2(nn.Module):
    """mean absolute percentage error"""

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        tabs = target.abs()
        tabs[tabs < self.epsilon] = self.epsilon
        return (prediton - target).abs() / tabs


class MSE(nn.Module):
    """mean square error"""

    def __init__(self):
        super().__init__()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        return torch.mean(torch.square(prediton - target))


class MSE_2(nn.Module):
    """mean square error"""

    def __init__(self):
        super().__init__()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        return torch.square(prediton - target)
