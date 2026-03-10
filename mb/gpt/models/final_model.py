from __future__ import annotations

import importlib
from typing import Any, Optional

from torch import nn

__all__ = ["GetModel"]


_ALIASES: dict[str, str] = {
    # text
    "smalvlm": "mb.gpt.models.text_encoder_smalvlm.model",
    "text_encoder_smalvlm": "mb.gpt.models.text_encoder_smalvlm.model",
    "textencodertest": "mb.gpt.models.text_encoder_smalvlm.TextEncoderTest",
    # vlm/clip
    "clip": "mb.gpt.models.vlm_encoder_clip.model",
    "vlm_encoder_clip": "mb.gpt.models.vlm_encoder_clip.model",
    "vlmencodertest": "mb.gpt.models.vlm_encoder_clip.VlmEncoderTest",
    # connectors
    "connecter_type1": "mb.gpt.models.connecter_type1.model",
    "connecter_type2": "mb.gpt.models.connecter_type2.model",
    "connector_type1": "mb.gpt.models.connecter_type1.model",
    "connector_type2": "mb.gpt.models.connecter_type2.model",
    "type1": "mb.gpt.models.connecter_type1.model",
    "type2": "mb.gpt.models.connecter_type2.model",
}


def _import_symbol(path: str) -> Any:
    module_path, _, symbol = path.rpartition(".")
    if not module_path or not symbol:
        raise ValueError(f"Invalid reference '{path}'. Expected 'module.symbol'.")
    module = importlib.import_module(module_path)
    return getattr(module, symbol)


def _resolve_ref(ref: str, *, base_pkg: str = "mb.gpt.models") -> str:
    name = ref.strip()
    if not name:
        raise ValueError("Empty model reference")

    alias = _ALIASES.get(name.lower())
    if alias is not None:
        return alias

    # already fully-qualified
    if name.startswith("mb."):
        return name

    # allow short refs relative to mb.gpt.models, like: vlm_encoder_clip.model
    if "." in name:
        return f"{base_pkg}.{name}"

    # allow module-only refs; assume `model` symbol
    return f"{base_pkg}.{name}.model"


def _build_nn_module(ref: Optional[str]) -> Optional[nn.Module]:
    if ref is None:
        return None
    s = str(ref).strip()
    if not s or s.lower() in {"none", "null", "identity"}:
        return None

    resolved = _resolve_ref(s)
    obj = _import_symbol(resolved)

    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, type) and issubclass(obj, nn.Module):
        return obj()
    if callable(obj):
        out = obj()
        if not isinstance(out, nn.Module):
            raise TypeError(f"Factory '{resolved}' did not return an nn.Module")
        return out
    raise TypeError(f"Reference '{resolved}' is not an nn.Module/class/factory")


def _build_connector(ref: str, *, vlm_encoder: Optional[nn.Module], text_encoder: Optional[nn.Module]) -> nn.Module:
    s = str(ref).strip()
    if not s:
        s = "gpt"

    lower = s.lower()
    if lower in {"gpt", "basic_gpt", "clm", "basictestmodel", "basic_test_model"}:
        from .basic_gpt import BasicGPTModel, GPTConfig

        return BasicGPTModel(GPTConfig())

    # connector reference (alias or import path)
    resolved = _resolve_ref(s)
    obj = _import_symbol(resolved)

    if isinstance(obj, nn.Module):
        return obj

    if isinstance(obj, type) and issubclass(obj, nn.Module):
        # try to pass encoders if supported
        try:
            return obj(vlm_encoder=vlm_encoder, text_encoder=text_encoder)
        except TypeError:
            return obj()

    if callable(obj):
        # expected signature: model(vlm_encoder=..., text_encoder=...)
        try:
            out = obj(vlm_encoder=vlm_encoder, text_encoder=text_encoder)
        except TypeError:
            out = obj()
        if not isinstance(out, nn.Module):
            raise TypeError(f"Connector factory '{resolved}' did not return an nn.Module")
        return out

    raise TypeError(f"Connector reference '{resolved}' is not an nn.Module/class/factory")

class GetModel(nn.Module):
    def __init__(self, ModelParams):
        super().__init__()
        self.ModelParams = ModelParams
        self.model = self._get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def _get_model(self):
        text_encoder = _build_nn_module(getattr(self.ModelParams, "text_encoder", None))
        vlm_encoder = _build_nn_module(
            getattr(self.ModelParams, "vlm_encoder", None)
            or getattr(self.ModelParams, "clm_encoder", None)
        )

        connecter = (
            getattr(self.ModelParams, "connecter", None)
            or getattr(self.ModelParams, "connector", None)
            or getattr(self.ModelParams, "model_type", None)
            or "gpt"
        )

        return _build_connector(str(connecter), vlm_encoder=vlm_encoder, text_encoder=text_encoder)