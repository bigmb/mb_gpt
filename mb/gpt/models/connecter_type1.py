import torch
from torch import nn
from typing import Optional

__all__ = ["BasicLLM"]

class BasicLLM(nn.Module):
    def __init__(self, 
                 vlm_encoder: Optional[nn.Module] = None,
                 text_encoder: Optional[nn.Module] = None,
                 vlm_emb_dim: Optional[int] = None,
                 text_emb_dim: Optional[int] = None,
                 output_classes: int = 10):
        super().__init__()

        self.vlm_encoder = vlm_encoder if vlm_encoder is not None else nn.Identity()
        self.text_encoder = text_encoder if text_encoder is not None else nn.Identity()

        if vlm_emb_dim is not None and text_emb_dim is not None:
            self.linear1 = nn.Linear(vlm_emb_dim + text_emb_dim, output_classes)
        else:
            self.linear1 = nn.LazyLinear(output_classes)

    def forward(self, 
                vlm_emb: torch.Tensor, 
                text_emb: torch.Tensor) -> torch.Tensor:
        
        vlm_emb = self.vlm_encoder(vlm_emb)
        text_emb = self.text_encoder(text_emb)

        combined_emb = torch.cat([vlm_emb, text_emb], dim=-1)

        projected_emb = self.linear1(combined_emb)

        if projected_emb.dim() == 2:
            projected_emb = projected_emb.unsqueeze(1)

        return projected_emb


def model(
    *,
    vlm_encoder: Optional[nn.Module] = None,
    text_encoder: Optional[nn.Module] = None,
    output_classes: int = 10,
    vlm_emb_dim: Optional[int] = None,
    text_emb_dim: Optional[int] = None,
) -> nn.Module:
    """Factory for YAML selection.

    Expected to be called by `GetModel` with `vlm_encoder` and `text_encoder`.
    """

    return BasicLLM(
        vlm_encoder=vlm_encoder,
        text_encoder=text_encoder,
        vlm_emb_dim=vlm_emb_dim,
        text_emb_dim=text_emb_dim,
        output_classes=output_classes,
    )