"""
Main tokenizer class for the GPT model.
"""

import regex as re
from typing import List, Optional, Tuple, Union
import tiktoken
from PIL import Image
# import sentencepiece as spm

__all__ = ["Tokenizer", "VITokenizer"]

SPLIT_REGEX1 = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+""")
SPLIT_REGEX2 = re.compile(r"""[\w'-]+|[^\w\s]+|\s+""")

class Tokenizer:
    """
    Base tokenizer class for the GPT model.
    """
    def __init__(self) -> None:
        self.enc = None
        self.vocab = None

    def load_tiktoken(self,token_type='gpt2') -> None:
        """
        Load the TikToken tokenizer.
        """
        self.enc = tiktoken.get_encoding(token_type)
        return self.enc

    def _convert_split_text(self, text: str,split_regex: str = None) -> List[str]:
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
        vocab_size = len(words)
        self.vocab = {word: idx for idx, word in enumerate(words)}
        print(f"Vocabulary size: {vocab_size}")
        for idx,word in self.vocab.items():
            print(f"{word}: {idx}")
            if idx > 10:
                break
        return self.vocab

    def _encode_text(self, text: str) -> List[int]:
        """
        Encode text into tokens.
        Args:
            text: The text to encode.
            vocab: The vocabulary to use for encoding.
        Returns:
            List[int]: A list of encoded tokens.
        """
        return [self.vocab[token] for token in text]

    def _decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode tokens into text.
        Args:
            tokens: The tokens to decode.
            vocab: The vocabulary to use for decoding.
        Returns:
            str: The decoded text.
        """
        return " ".join([self.vocab[token] for token in tokens])

    def encode(self, text: str) -> List[int]:
        """
        Encode text into tokens.
        If the TikToken tokenizer is loaded, it will use that.
        Else needs an existing vocabulary to encode the text.
        Args:
            text: The text to encode.
            byte_encode: Whether to encode the text as bytes. Defaults to False.
        Returns:
            List[int]: A list of encoded tokens.
        """
        if self.enc is None:
            if self.vocab is None:
                raise ValueError("No tokenizer or vocabulary loaded.")
            return self._encode_text(text, self.vocab)
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
            return self._decode_tokens(tokens, self.vocab)
        return self.enc.decode(tokens)


class VITokenizer():
    """ 
    Tokenizer for Vision and Language tasks.
    """
    def __init__(self) -> None:
        pass

    def _read_image(self, image: str) -> Image:
        """
        Read an image.
        """
        if isinstance(image, Image.Image):
            return image
        return Image.open(image)

    def _to_patches(self, image: str, patch_size: int) -> List[str]:
        """
        Convert an image to patches.
        """
        pass


    def _convert_image_to_tokens(self, image: str) -> List[str]:
        """
        Convert an image to a list of tokens.
        """
        pass

    def _convert_tokens_to_image(self, tokens: List[str]) -> str:
        """
        Convert tokens to an image.
        """
        pass


class AudioTokenizers():
    """
    Tokenizers for audio data.
    """
    def __init__(self) -> None:
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