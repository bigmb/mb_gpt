import importlib
from typing import List


class AudioTokenizers():
    """
    Tokenizers for audio data.
    """
    def __init__(self,logger=None) -> None:
        try:
            importlib.import_module("torchaudio")
        except Exception:
            pass
        self.logger = logger
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