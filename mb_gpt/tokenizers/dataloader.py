"""
DataLoader for tokenizer for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader

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

def create_dataloader(texts, tokenizer, batch_size=32, shuffle=True):
    dataset = TokenizerDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader