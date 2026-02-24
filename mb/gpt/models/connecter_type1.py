import torch
from torch import nn
from typing import List, Optional

__all__ = ["BasicLLM"]

class BasicLLM(nn.Module):
    def __init__(self, 
                 vlm_emb_dim: int,
                 text_emb_dim: int,
                 embedding_dim: int):
        super().__init__()

        self.linear1 = nn.Linear(vlm_emb_dim + text_emb_dim, embedding_dim)

    def forward(self, 
                vlm_emb: torch.Tensor, 
                text_emb: torch.Tensor) -> torch.Tensor:
        

        combined_emb = torch.cat([vlm_emb, text_emb], dim=-1)

        projected_emb = self.linear1(combined_emb)

        if projected_emb.dim() == 2:
            projected_emb = projected_emb.unsqueeze(1)

        return projected_emb
    
    def __repr__(self):
        return f"BasicLLM(embedding_dim={self.linear1.out_features})"