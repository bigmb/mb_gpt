"""YAML config loader + typed parameter objects.

model:
  text_encoder: text_encoder_smalvlm.model
  vlm_encoder: vlm_encoder_clip.model
  connecter: connecter_type1.model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

__all__ = [
    "YamlConfig",
    "ModelParams",
    "DataParams",
    "TrainParams",
    "OutputParams",
]

PathLike = Union[str, Path]


try:  
    from mb.utils.yaml_reader import read_yaml as _read_yaml  # type: ignore
except Exception:
    _read_yaml = None


@dataclass
class YamlConfig:
    """YAML config.

    Convenience: after loading, sections are available as attributes:

    - `cfg.TrainParams.<field>`
    - `cfg.DataParams.<field>`
    - `cfg.ModelParams.<field>`
    - `cfg.OutputParams.<field>`
    """

    path: Path
    data: Dict[str, Any]

    # Parsed sections (built from `data` in __post_init__)
    TrainParams: "TrainParams" = field(init=False)
    DataParams: "DataParams" = field(init=False)
    ModelParams: "ModelParams" = field(init=False)
    OutputParams: "OutputParams" = field(init=False)

    @classmethod
    def from_file(cls, path: PathLike, *, encoding: str = "utf-8") -> "YamlConfig":
        yaml_path = Path(path).expanduser().resolve()
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        loaded: Any
        if _read_yaml is not None:
            # mb_utils helper 
            try:
                loaded = _read_yaml(yaml_path)
            except Exception:
                loaded = _read_yaml(str(yaml_path))
        else:
            # fallback to PyYAML if installed
            try:
                import yaml  
            except Exception as e: 
                raise ImportError(
                    "Missing YAML reader. Install `mb_utils` (preferred) or `PyYAML`."
                ) from e

            text = yaml_path.read_text(encoding=encoding)
            loaded = yaml.safe_load(text)

        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise TypeError(
                f"YAML root must be a mapping/dict, got {type(loaded).__name__} in {yaml_path}"
            )

        return cls(path=yaml_path, data=dict(loaded))

    def __post_init__(self) -> None:
        # Build strongly-typed param objects for easy access.
        self.TrainParams = TrainParams.from_dict(self.data)
        self.DataParams = DataParams.from_dict(self.data)
        self.ModelParams = ModelParams.from_dict(self.data)
        self.OutputParams = OutputParams.from_dict(self.data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.data.get(key, default)


@dataclass
class ModelParams:
    """
    Selection-only model config.
    """

    text_encoder: Optional[str] = None
    vlm_encoder: Optional[str] = None
    connecter: str = "gpt"

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelParams":
        section = data.get("model") or data.get("ModelParams")
        if not isinstance(section, Mapping):
            section = {}

        # encoders
        text_encoder = section.get("text_encoder") or section.get("text")
        vlm_encoder = (
            section.get("vlm_encoder")
            or section.get("vlm")
            or section.get("clip_encoder")
            or section.get("clip")
            or section.get("clm_encoder")
            or section.get("clm")
        )

        # connector / model
        connecter = (
            section.get("connecter")
            or section.get("connector")
            or section.get("model_type")
            or section.get("type")
            or section.get("name")
            or section.get("model_name")
            or "gpt"
        )

        return cls(
            text_encoder=str(text_encoder) if text_encoder is not None else None,
            vlm_encoder=str(vlm_encoder) if vlm_encoder is not None else None,
            connecter=str(connecter),
            extra={
                k: v
                for k, v in section.items()
                if k
                not in {
                    "text_encoder",
                    "text",
                    "vlm_encoder",
                    "vlm",
                    "clip_encoder",
                    "clip",
                    "clm_encoder",
                    "clm",
                    "connecter",
                    "connector",
                    "model_type",
                    "type",
                    "name",
                    "model_name",
                }
            },
        )


@dataclass
class DataParams:
    name: str = "text_file"
    split: str = "train"
    data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    max_length: int = 256
    num_workers: int = 0
    shuffle: bool = True
    train_ratio: float = 0.9

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataParams":
        section = data.get("dataset") or data.get("DataParams")
        if not isinstance(section, Mapping):
            section = {}
        return cls(
            name=str(section.get("name", "text_file")),
            split=str(section.get("split", "train")),
            data_path=section.get("data_path") or section.get("path") or section.get("train_path"),
            val_data_path=section.get("val_data_path") or section.get("val_path"),
            max_length=int(section.get("max_length", section.get("block_size", 256))),
            num_workers=int(section.get("num_workers", 0)),
            shuffle=bool(section.get("shuffle", True)),
            train_ratio=float(section.get("train_ratio", 0.9)),
            extra={
                k: v
                for k, v in section.items()
                if k
                not in {
                    "name",
                    "split",
                    "data_path",
                    "path",
                    "train_path",
                    "val_data_path",
                    "val_path",
                    "max_length",
                    "block_size",
                    "num_workers",
                    "shuffle",
                    "train_ratio",
                }
            },
        )


@dataclass
class TrainParams:
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    logging_steps: int = 100
    save_steps: int = 500
    gpu: list[int] = field(default_factory=list)
    debug: bool = False
    seed: int = 1337
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = 1.0

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainParams":
        section = data.get("training") or data.get("TrainParams")
        if not isinstance(section, Mapping):
            section = {}
        epochs = section.get("epochs", section.get("num_epochs", section.get("num_epoch", 1)))

        raw_gpus = section.get("gpu", section.get("gpus", []))
        if raw_gpus is None:
            raw_gpus = []
        if isinstance(raw_gpus, (list, tuple)):
            gpus = list(raw_gpus)
        else:
            gpus = [raw_gpus]

        gpu_ids = []
        for g in gpus:
            try:
                gpu_ids.append(int(g))
            except Exception:
                continue

        return cls(
            batch_size=int(section.get("batch_size", 8)),
            epochs=int(epochs),
            learning_rate=float(section.get("learning_rate", 5e-5)),
            weight_decay=float(section.get("weight_decay", 0.0)),
            warmup_steps=int(section.get("warmup_steps", 0)),
            logging_steps=int(section.get("logging_steps", 100)),
            save_steps=int(section.get("save_steps", 500)),
            gpu=gpu_ids,
            debug=bool(section.get("debug", False)),
            seed=int(section.get("seed", 1337)),
            grad_accum_steps=int(section.get("grad_accum_steps", 1)),
            max_grad_norm=section.get("max_grad_norm", 1.0),
            extra={
                k: v
                for k, v in section.items()
                if k
                not in {
                    "batch_size",
                    "epochs",
                    "num_epochs",
                    "num_epoch",
                    "learning_rate",
                    "weight_decay",
                    "warmup_steps",
                    "logging_steps",
                    "save_steps",
                    "gpu",
                    "gpus",
                    "debug",
                    "seed",
                    "grad_accum_steps",
                    "max_grad_norm",
                }
            },
        )


@dataclass
class OutputParams:
    save_dir: str = "./outputs"
    print_output: bool = True
    output_path: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputParams":
        section = data.get("output") or data.get("OutputParams")
        if not isinstance(section, Mapping):
            section = {}
        return cls(
            save_dir=str(section.get("save_dir", "./outputs")),
            print_output=bool(section.get("print_output", True)),
            output_path=section.get("output_path"),
            extra={
                k: v
                for k, v in section.items()
                if k not in {"save_dir", "print_output", "output_path"}
            },
        )
