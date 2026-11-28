import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for i in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.model(x)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadedAttention(num_heads, embed_dim, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x