"""
Main tokenizer class for the GPT model.
"""

from __future__ import annotations

import logging
import importlib
from typing import List, Pattern

try:
    import regex as _re 
    _HAS_REGEX = True
except Exception: 
    import re as _re 
    _HAS_REGEX = False

import tiktoken
import torch
from torch import nn
from mb.utils.logging import logg

__all__ = ["TextTokenizer", "VITokenizer"]

SPLIT_REGEX1: Pattern[str] | None = (
    _re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+""")
    if _HAS_REGEX
    else None
)
SPLIT_REGEX2: Pattern[str] = _re.compile(r"""[\w'-]+|[^\w\s]+|\s+""")


def _log_info(message: str, logger: logging.Logger | None = None) -> None:
    (logger or logging.getLogger(__name__)).info(message)


class TextTokenizer(nn.Module):
    """
    Base tokenizer class for the GPT model.
    """
    def __init__(self, logger=None) -> None:
        super().__init__()
        self.enc = None
        self.vocab = None
        self.decode_vocab = None
        self.texts = None
        self.logger = logger

    def load_tiktoken(self, token_type: str = "gpt2"):
        """
        Load the TikToken tokenizer.
        """
        self.enc = tiktoken.get_encoding(token_type)
        return self.enc

    def _convert_split_text(self, text: str, split_regex: Pattern[str] | None = None) -> List[str]:
        """
        Split text into tokens.
        Args:
            text: The text to split.
            split_regex: The regex to use to split the text. Defaults to GPT-2 regex.
        Returns:
            A list of split text based on the split_regex.
        """
        if split_regex is None:
            split_regex = SPLIT_REGEX1 or SPLIT_REGEX2
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

        logg.info(f"Vocabulary size: {vocab_size}", logger=self.logger)
        for idx,word in self.decode_vocab.items():
            logg.info(f"{word}: {idx}", logger=self.logger)
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
        vocab = self.vocab or {}
        unk_id = vocab.get("<|unk|>", 0)
        return [vocab.get(token, unk_id) for token in tokens]

    def _decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode tokens into text.
        Args:
            tokens: The tokens to decode.
        Returns:
            str: The decoded text.
        """
        if not self.decode_vocab:
            raise ValueError("No vocabulary loaded.")
        decoded = " ".join(self.decode_vocab.get(token, "<|unk|>") for token in tokens)
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
