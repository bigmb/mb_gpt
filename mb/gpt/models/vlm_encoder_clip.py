import torch
import torch.nn as nn

__all__ = ['VlmEncoderTest']

class VlmEncoderTest(nn.Module):
    def __init__(self,in_dim=512,out_dim=128):
        super(VlmEncoderTest, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
            return self.linear(x)
        
