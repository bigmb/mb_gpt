"""
DataLoader for tokenizer for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

__all__ = ["TokenizerDataset", "VITokenizerDataset", "TokenizerDataLoader"]

class TokenizerDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=None, pad_token="<|pad|>"):
        """
        Initialize the dataset.
        Args:
            texts: List of text strings.
            tokenizer: The tokenizer instance (must have encode method).
            max_length: Maximum length of tokens (for padding/truncation). Defaults to None (no truncation).
            pad_token: The token to use for padding. Defaults to "<|pad|>".
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        if self.max_length is not None:
            pad_id = self.tokenizer.vocab.get(self.pad_token, 0) 
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length] 
            else:
                tokens += [pad_id] * (self.max_length - len(tokens)) 
        
        return torch.tensor(tokens, dtype=torch.long) ## torch long for using Torch.Embedding


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

class TokenizerDataLoader(DataLoader):
    def __init__(self, texts, tokenizer, batch_size=4,shuffle=False,max_length=None, pad_token="<|pad|>", **kwargs):
        """
        Initialize the DataLoader.
        Args:
            texts: List of text strings.
            tokenizer: The tokenizer instance (must have encode method).
            max_length: Maximum length of tokens (for padding/truncation). Defaults to None (no truncation).
            pad_token: The token to use for padding. Defaults to "<|pad|>".
            **kwargs: Additional arguments for DataLoader.
        """
        self.dataset = TokenizerDataset(texts, tokenizer, max_length, pad_token)
        self.batch_size = batch_size
        self.shuffle = shuffle
        super().__init__(self.dataset, **kwargs)
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        Args:
            batch: A list of samples from the dataset.
        Returns:
            dict: A dictionary of the batch.
        """
        return torch.stack(batch)
    
    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)


# def create_dataloader(texts, tokenizer, batch_size=32, shuffle=True):
#     dataset = TokenizerDataset(texts, tokenizer)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader