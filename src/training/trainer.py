import torch
import torch.nn as nn
from src.training.loss import DPRLoss

class Trainer:
    def __init__(self):
        self.criterion = DPRLoss()

    def train(query_encoder: nn.Module, document_encoder: nn.Module):
        #raise NotImplementedError()
        epochs = 100
        batch_size = 8
        optimizer = torch.optim.AdamW(query_encoder.parameters())

