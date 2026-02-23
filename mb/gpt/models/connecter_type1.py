import torch
from torch import nn
from typing import List, Optional

__all__ = ["BasicLLM"]

class BasicLLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.multihead_Attn = nn.MultiheadAttention(embedding_dim, num_heads=8)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)