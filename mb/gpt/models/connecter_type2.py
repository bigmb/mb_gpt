import torch
from torch import nn
from typing import Optional

__all__ = ["BasicLLM"]


class BasicLLM(nn.Module):
	"""A simple connector that combines VLM + text embeddings.

	This is intentionally minimal (same interface as connecter_type1) so you
	can select it from YAML without additional architecture parameters.
	"""

	def __init__(
		self,
		vlm_encoder: Optional[nn.Module] = None,
		text_encoder: Optional[nn.Module] = None,
		vlm_emb_dim: Optional[int] = None,
		text_emb_dim: Optional[int] = None,
		output_classes: int = 10,
	):
		super().__init__()
		self.vlm_encoder = vlm_encoder if vlm_encoder is not None else nn.Identity()
		self.text_encoder = text_encoder if text_encoder is not None else nn.Identity()

		# Type2: concatenate then run a 2-layer MLP (lazy when dims unknown)
		if (vlm_emb_dim is not None) and (text_emb_dim is not None):
			in_dim = int(vlm_emb_dim) + int(text_emb_dim)
			self.proj = nn.Sequential(
				nn.Linear(in_dim, in_dim),
				nn.GELU(),
				nn.Linear(in_dim, int(output_classes)),
			)
		else:
			self.proj = nn.Sequential(
				nn.LazyLinear(256),
				nn.GELU(),
				nn.Linear(256, int(output_classes)),
			)

	def forward(self, vlm_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
		vlm_emb = self.vlm_encoder(vlm_emb)
		text_emb = self.text_encoder(text_emb)
		x = torch.cat([vlm_emb, text_emb], dim=-1)
		out = self.proj(x)
		if out.dim() == 2:
			out = out.unsqueeze(1)
		return out


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

