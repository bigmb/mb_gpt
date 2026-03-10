"""Lightweight training summary logger.

Persists a JSON file with per-epoch metrics.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:  # optional dependency (provided by mb_utils)
    from mb.utils.logging import logg  # type: ignore
except Exception:  # pragma: no cover
    import logging

    class _FallbackLogg:
        @staticmethod
        def info(message: str, logger: Optional[logging.Logger] = None) -> None:
            (logger or logging.getLogger(__name__)).info(message)

        @staticmethod
        def warning(message: str, logger: Optional[logging.Logger] = None) -> None:
            (logger or logging.getLogger(__name__)).warning(message)

        @staticmethod
        def error(message: str, logger: Optional[logging.Logger] = None) -> None:
            (logger or logging.getLogger(__name__)).error(message)

    logg = _FallbackLogg()

__all__ = ['TrainSummary']

class TrainSummary:
    def __init__(self, data: dict,logger=None):
        self.summary = data
        save_dir = self.summary.get("save_dir")
        default_path = "./summary_output.json"
        if save_dir:
            default_path = str(Path(save_dir) / "train_summary.json")
        self.output_path = self.summary.get('output_path', default_path)
        self.print_output = self.summary.get('print_output', False)
        self.logger = logger
        self.history: list[dict[str, Any]] = []

        try:
            Path(self.output_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def __repr__(self):
        return f"TrainSummary(TrainSummary={self.summary})"
    
    def _epoch_data(self, epoch: int, loss: float, **kwargs):
        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "loss": float(loss),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        payload.update(kwargs)
        return payload

    def _loss_data(self, loss: float, **kwargs):
        payload: Dict[str, Any] = {"loss": float(loss)}
        payload.update(kwargs)
        return payload

    def _extra_data(self, **kwargs):
        return dict(kwargs)

    def log_epoch(self, epoch: int, loss: float, **kwargs):
        row = self._epoch_data(epoch, loss, **kwargs)
        self.history.append(row)

        out_path = Path(self.output_path).expanduser()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(self.history, indent=2, sort_keys=False), encoding="utf-8")
        except Exception as e:
            logg.warning(f"Failed to write training summary to {out_path}: {e}", logger=self.logger)

        if self.print_output:
            logg.info(f"Epoch {epoch}: loss={loss:.6f}", logger=self.logger)

        return row