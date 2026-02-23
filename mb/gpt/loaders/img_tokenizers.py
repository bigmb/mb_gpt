import torch
import torch.nn as nn


class VITokenizer(nn.Module):
    """ 
    Tokenizer for Vision and Language tasks.
    """
    def __init__(self,batch_images,patch_size=16,emb_dim=768,cls_token=False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.batch_images = batch_images
        self.num_patches = (batch_images.shape[2]//patch_size)**2
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1,1,emb_dim))
            self.positional_embeddings = nn.Parameter(torch.randn(1,self.num_patches+1, emb_dim))
        else:
            self.positional_embeddings = nn.Parameter(torch.randn(1,self.num_patches, emb_dim))
        self.patch_embeddings = nn.Linear(3*patch_size*patch_size, emb_dim)
        
    def forward(self,x):
        
        if x.shape[2] % self.patch_size != 0:
            raise ValueError(f"Patch size {self.patch_size} does not divide image width {x.shape[2]}")
        if x.shape[3] % self.patch_size != 0:
            raise ValueError(f"Patch size {self.patch_size} does not divide image height {x.shape[3]}")
        patches = torch.nn.functional.unfold(x, (self.patch_size,self.patch_size), self.patch_size).permute(0,2,1)
        patch_embeddings = self.patch_embeddings(patches)

        if hasattr(self,'cls_token'):
            cls_token = self.cls_token.expand(patches.shape[0],-1,-1)
            patch_embeddings = torch.cat((cls_token,patch_embeddings),dim=1)
        embeddings = patch_embeddings + self.positional_embeddings
        return embeddings
