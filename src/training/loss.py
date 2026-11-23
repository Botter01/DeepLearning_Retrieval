import torch
import torch.nn as nn
import torch.nn.functional as F

class DPRLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        In-batch softmax over cosine similarities scaled by 1/temperature.
        Embeddings are L2-normalised: with raw dot products the post-LayerNorm
        vectors have norm ~sqrt(emb_dim), logits are huge at init, and the
        cheapest way to lower the loss is to make every embedding identical
        (loss -> ln(batch_size), i.e. chance level).

        Returns: A 1d torch tensor
        """
        queries = F.normalize(queries, dim=-1)
        documents = F.normalize(documents, dim=-1)
        scores = queries @ documents.T / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, labels, reduction='none')
