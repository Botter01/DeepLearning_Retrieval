import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from src.retrievers.models.attention import *
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, vocab_size: int, pooling_type: str = "cls"):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.vocab_size = vocab_size
        self.pooling_type = pooling_type
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.positional_enc = PositionalEncoding(self.emb_dim, dropout=0.1)
        self.model = nn.ModuleList(
            *[EncoderLayer(self.emb_dim, self.n_heads, self.feed_forward_dim, dropout_rate=0.1) for _ in range(self.n_layers)]

        )
        #raise NotImplementedError()

    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            return x[:,0,:]
        
        elif self.cfg.model.pooling_type == 'max':
            return torch.max(x, dim=1).values
        
        elif self.cfg.model.pooling_type == 'mean':
            return torch.mean(x, dim=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        mask = (x!= 0).unsqueeze(0)

        x = self.embedding(x)
        x = self.positional_enc(x)
        for layer in self.n_layers:
            x = layer(x, mask)
        return self.pool(x)
