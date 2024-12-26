"""
DataLoader for tokenizer for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

__all__ = ["TokenizerDataset", "VITokenizerDataset"]

class TokenizerDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens)


class VITokenizerDataset(Dataset):
    def __init__(self, image_paths,labels,transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),
                                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            image = transform(image) 
        
        return image,label


def create_dataloader(texts, tokenizer, batch_size=32, shuffle=True):
    dataset = TokenizerDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader