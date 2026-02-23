import torch
from torch import nn
from typing import List, Optional

__all__ = ["BasicLLM"]

class BasicLLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
                            ##Creating Dense vector of input - lookup table of size (no. of tokens, embedding dimension)
        # self.multihead_Attn = nn.MultiheadAttention(embedding_dim, num_heads=8)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
    
    def __repr__(self):
        return f"BasicLLM(vocab_size={self.embedding.num_embeddings}, embedding_dim={self.embedding.embedding_dim})"