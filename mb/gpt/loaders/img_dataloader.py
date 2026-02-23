import torch
from torch.utils.data import Dataset, DataLoader
import importlib
from typing import List, Optional
from dataclasses import dataclass

__all__ = ["VITokenizerDataset"]

@dataclass
class VITokenizerInput:
    image_paths: List[str]
    labels: List[int]
    text: Optional[List[str]] = None
    transform: Optional[callable] = None
    

class VITokenizerDataset(Dataset):
    def __init__(self, vit_input: VITokenizerInput):
        self.image_paths = vit_input.image_paths
        self.labels = vit_input.labels
        self.text = vit_input.text
        self.transform = vit_input.transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            from PIL import Image
        except Exception as e:  
            raise ImportError("PIL is required for VITokenizerDataset. Install `Pillow`.") from e

        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        text = self.text[idx] if self.text is not None else None

        if self.transform:
            image = self.transform(image)
        else:
            try:
                transforms = importlib.import_module("torchvision.transforms")
            except Exception as e:  
                raise ImportError(
                    "torchvision is required for the default VITokenizerDataset transform. "
                    "Either install `torchvision` or pass a custom `transform` callable."
                ) from e

            default_transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            image = default_transform(image)
        
        return image,label,text

    def __repr__(self):
        return f"VITokenizerDataset(num_samples={len(self)}), transform={self.transform})"