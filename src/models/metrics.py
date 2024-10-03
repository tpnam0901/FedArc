import torch
import torch.nn as nn
from configs.base import Config


class Accuracy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.name = "accuracy"

    def forward(self, outputs, labels):
        _, predicted = torch.max(outputs[0].detach().cpu().data, 1)
        correct = (predicted == labels.detach().cpu()).sum().item()
        return torch.tensor(correct / labels.size(0))

