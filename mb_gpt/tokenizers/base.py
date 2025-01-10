"""
Main tokenizer class for the GPT model.
"""

import regex as re
from typing import List, Optional, Tuple, Union
import tiktoken
import torch
from torch import nn

__all__ = ["Tokenizer", "VITokenizer"]

SPLIT_REGEX1 = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+""")
SPLIT_REGEX2 = re.compile(r"""[\w'-]+|[^\w\s]+|\s+""")

class Tokenizer(nn.Module):
    """
    Base tokenizer class for the GPT model.
    """
    def __init__(self) -> None:
        self.enc = None
        self.vocab = None
        self.texts = None

    def load_tiktoken(self, token_type='gpt2') -> None:
        """
        Load the TikToken tokenizer.
        """
        self.enc = tiktoken.get_encoding(token_type)
        return self.enc

    def _convert_split_text(self, text: str, split_regex: str = None) -> List[str]:
        """
        Split text into tokens.
        Args:
            text: The text to split.
            split_regex: The regex to use to split the text. Defaults to GPT-2 regex.
        Returns:
            A list of split text based on the split_regex.
        """
        if split_regex is None:
            split_regex = SPLIT_REGEX1
        return split_regex.findall(text)

    def _build_vocab(self, texts: List[str]) -> dict:
        """
        Build a vocabulary from a list of texts.
        Args:
            texts: A list of texts to build the vocabulary from.
        Returns:
            dict: A dictionary of the vocabulary.
        """
        words = sorted(list(set(texts)))
        words.extend(["<|endoftext|>", "<|unk|>", "<|pad|>"])
        vocab_size = len(words)
        self.vocab = {word: idx for idx, word in enumerate(words)}
        self.decode_vocab = {idx: word for idx, word in enumerate(words)}

        print(f"Vocabulary size: {vocab_size}")
        for idx,word in self.decode_vocab.items():
            print(f"{word}: {idx}")
            if idx > 10:
                break
        return self.vocab

    def _encode_text(self, text: str) -> List[int]:
        """
        Encode text into tokens.
        Appends <|endoftext|> at the end of the document.
        Args:
            text: The text to encode.
        Returns:
            List[int]: A list of encoded tokens.
        """
        tokens = self._convert_split_text(text)
        tokens.append("<|endoftext|>")
        return [self.vocab[token] if token in self.vocab else self.vocab["<|unk|>"] for token in tokens]

    def _decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode tokens into text.
        Args:
            tokens: The tokens to decode.
        Returns:
            str: The decoded text.
        """
        decoded = " ".join([list(self.vocab.keys())[list(self.vocab.values()).index(token)] for token in tokens])
        return decoded.replace("<|endoftext|>", "").strip() 

    def encode(self, text: str) -> List[int]:
        """
        Encode text into tokens.
        If the TikToken tokenizer is loaded, it will use that.
        Else needs an existing vocabulary to encode the text.
        Args:
            text: The text to encode.
        Returns:
            List[int]: A list of encoded tokens.
        """
        if self.enc is None:
            if self.vocab is None:
                raise ValueError("No tokenizer or vocabulary loaded.")
            return self._encode_text(text)
        return self.enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens into text.
        If the TikToken tokenizer is loaded, it will use that.
        Else needs an existing vocabulary to decode the tokens.
        Args:
            tokens: The tokens to decode.
        Returns:
            str: The decoded text.
        """
        if self.enc is None:
            if self.vocab is None:
                raise ValueError("No tokenizer or vocabulary loaded.")
            return self._decode_tokens(tokens)
        return self.enc.decode(tokens)


class VITokenizer(nn.Module):
    """ 
    Tokenizer for Vision and Language tasks.
    """
    def __init__(self,batch_images,patch_size=16,emb_dim=768,cls_token=False) -> None:
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


class AudioTokenizers():
    """
    Tokenizers for audio data.
    """
    def __init__(self) -> None:
        import torchaudio
        pass
    
    def _convert_audio_to_tokens(self, audio: str) -> List[str]:
        """
        Convert audio to a list of tokens.
        """
        pass

    def _convert_tokens_to_audio(self, tokens: List[str]) -> str:
        """
        Convert tokens to audio.
        """
        pass