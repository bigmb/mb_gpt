"""YAML config loader.

Provides a small dataclass that reads a YAML file and returns a Python dict.

Example:
    from mb.gpt.utils.yaml_config import YamlConfig

    cfg = YamlConfig.from_file("config.yaml")
    config_dict = cfg.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


PathLike = Union[str, Path]


@dataclass(frozen=True, slots=True)
class YamlConfig:
    """Immutable YAML config.

    Attributes:
        path: Path to the YAML file.
        data: Parsed YAML content (must be a mapping/dict).
    """

    path: Path
    data: Dict[str, Any]

    @classmethod
    def from_file(cls, path: PathLike, *, encoding: str = "utf-8") -> "YamlConfig":
        """Load YAML from disk.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If PyYAML is not installed.
            TypeError: If the YAML root is not a mapping.
            ValueError: If the YAML cannot be parsed.
        """

        yaml_path = Path(path)
        text = yaml_path.read_text(encoding=encoding)

        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("Missing dependency: install `PyYAML` to read YAML configs.") from e

        try:
            loaded = yaml.safe_load(text)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML: {yaml_path}") from e

        if loaded is None:
            loaded = {}

        if not isinstance(loaded, Mapping):
            raise TypeError(
                f"YAML root must be a mapping/dict, got {type(loaded).__name__} in {yaml_path}"
            )

        return cls(path=yaml_path, data=dict(loaded))

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the config data."""

        return dict(self.data)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Dictionary-like access."""

        return self.data.get(key, default)
