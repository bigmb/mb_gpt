import torch
from torch import nn
from typing import List, Optional

__all__ = ["BasicLLM"]

class BasicLLM(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int,
                 vlm_emb_dim: int,
                 text_emb_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(vlm_emb_dim + text_emb_dim, embedding_dim)

    def forward(self, 
                vlm_emb: torch.Tensor, 
                text_emb: torch.Tensor, 
                input_ids: torch.Tensor) -> torch.Tensor:
        
        token_emb = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        combined_emb = torch.cat([vlm_emb, text_emb], dim=-1)

        projected_emb = self.linear1(combined_emb)

        if projected_emb.dim() == 2:
            projected_emb = projected_emb.unsqueeze(1)

        output = token_emb + projected_emb

        return output
    
    def __repr__(self):
        return f"BasicLLM(vocab_size={self.embedding.num_embeddings}, embedding_dim={self.embedding.embedding_dim})"