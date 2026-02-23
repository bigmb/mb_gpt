"""
DataLoader for tokenizer for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Optional
from .text_tokenizers import TextTokenizer

__all__ = ["TextTokenizerDataset", "TextTokenizerDataLoader"]

@dataclass
class TextTokenizerInput:
    texts: List[str]
    tokenizer: TextTokenizer
    max_length: Optional[int] = None
    pad_token: str = "<|pad|>"

class TextTokenizerDataset(Dataset):
    def __init__(self, tokenizer_input: TextTokenizerInput):
        """
        Initialize the dataset.
        Args:
            tokenizer_input: An instance of TextTokenizerInput containing texts, tokenizer, max_length, and pad_token.
        """
        self.texts = tokenizer_input.texts
        self.tokenizer = tokenizer_input.tokenizer
        self.max_length = tokenizer_input.max_length
        self.pad_token = tokenizer_input.pad_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        if self.max_length is not None:
            vocab = getattr(self.tokenizer, "vocab", None) or {}
            pad_id = vocab.get(self.pad_token, 0)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length] 
            else:
                tokens += [pad_id] * (self.max_length - len(tokens)) 
        
        return torch.tensor(tokens, dtype=torch.long) ## torch long for using Torch.Embedding

class TextTokenizerDataLoader(DataLoader):
    def __init__(self, tokenizer_input: TextTokenizerInput, batch_size=4, shuffle=False, **kwargs):
        """
        Initialize the DataLoader.
        Args:
            tokenizer_input: An instance of TextTokenizerInput containing texts, tokenizer, max_length, and pad_token.
            batch_size: The batch size for the DataLoader. Defaults to 4.
            shuffle: Whether to shuffle the data. Defaults to False.
            **kwargs: Additional arguments for DataLoader.
        """
        self.dataset = TextTokenizerDataset(tokenizer_input)
        self.batch_size = batch_size
        self.shuffle = shuffle
        super().__init__(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        Args:
            batch: A list of samples from the dataset.
        Returns:
            dict: A dictionary of the batch.
        """
        if not batch:
            return torch.empty((0, 0), dtype=torch.long)

        lengths = [int(x.numel()) for x in batch]
        if len(set(lengths)) == 1:
            return torch.stack(batch)

        max_len = max(lengths)
        vocab = getattr(self.dataset.tokenizer, "vocab", None) or {}
        pad_id = vocab.get(self.dataset.pad_token, 0)
        padded = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        for i, seq in enumerate(batch):
            padded[i, : seq.numel()] = seq
        return padded
    
    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )


# def create_dataloader(texts, tokenizer, batch_size=32, shuffle=True):
#     dataset = TokenizerDataset(texts, tokenizer)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader