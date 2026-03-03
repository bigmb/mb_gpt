import torch
from torch import nn

__all__ = ['GetModel']

class GetModel(nn.Module):
    def __init__(self, ModelParams):
        super().__init__()
        self.ModelParams = ModelParams
        self.model = self._get_model()
        
    def _get_model(self):
        if self.ModelParams.model_type == 'connecter_type1':
            from .connecter_type1 import BasicLLM
            model = BasicLLM(
                vlm_encoder=None,
                text_encoder=None,
                vlm_emb_dim=self.ModelParams.vlm_emb_dim,
                text_emb_dim=self.ModelParams.text_emb_dim,
                output_classes=self.ModelParams.output_classes
            )
        elif self.ModelParams.model_type == 'connecter_type2':
            from .connecter_type2 import BasicLLM
            model = BasicLLM(
                vlm_encoder=self.ModelParams.vlm_encoder,
                text_encoder=self.ModelParams.text_encoder,
                vlm_emb_dim=self.ModelParams.vlm_emb_dim,
                text_emb_dim=self.ModelParams.text_emb_dim,
                output_classes=self.ModelParams.output_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {self.ModelParams.model_type}")

        return model