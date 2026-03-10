from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .loaders.text_tokenizers import TextTokenizer
from .models.final_model import GetModel
from .utils.gpu_tools import get_gpus_by_least_usage
from .utils.train_summary import TrainSummary
from .utils.yaml_config import DataParams, ModelParams, OutputParams, TrainParams, YamlConfig

try:  # optional dependency (provided by mb_utils)
    from mb.utils.logging import logg, logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

    class _FallbackLogg:
        @staticmethod
        def info(message: str, logger: Optional[logging.Logger] = None) -> None:
            (logger or logging.getLogger(__name__)).info(message)

        @staticmethod
        def warning(message: str, logger: Optional[logging.Logger] = None) -> None:
            (logger or logging.getLogger(__name__)).warning(message)

    logg = _FallbackLogg()


__all__ = ["Trainer", "DDPTrainer", "load_params_from_yaml"]


def load_params_from_yaml(config_path: str) -> Tuple[TrainParams, DataParams, ModelParams, OutputParams]:
    cfg = YamlConfig.from_file(config_path)
    return cfg.TrainParams, cfg.DataParams, cfg.ModelParams, cfg.OutputParams


def _repo_root() -> Path:
    # mb/gpt/trainer.py -> repo root is two levels up from mb/
    return Path(__file__).resolve().parents[2]


def _default_text_path() -> Path:
    return _repo_root() / "data" / "the-verdict.txt"


class TokenBlockDataset(Dataset):
    """Contiguous blocks of tokens for next-token prediction."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size}")
        self.token_ids = token_ids
        self.block_size = int(block_size)
        self._num_blocks = max(0, (len(self.token_ids) - 1) // self.block_size)

    def __len__(self) -> int:
        return self._num_blocks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(idx) * self.block_size
        x = self.token_ids[start : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def _load_tokens(data_params: DataParams, tokenizer: TextTokenizer) -> list[int]:
    # Optional HF datasets support (only if installed)
    if (data_params.data_path is None) and str(data_params.name).lower().startswith("wikitext"):
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=data_params.split)
            text = "\n".join(ds["text"])  # type: ignore[index]
            return list(tokenizer.encode(text))
        except Exception as e:
            logg.warning(
                f"Could not load HuggingFace dataset '{data_params.name}'. Falling back to local text file. ({e})",
                logger=logger,
            )

    path = Path(data_params.data_path) if data_params.data_path else _default_text_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Text data not found at {path}. Set dataset.data_path in YAML to a valid text file."
        )
    text = path.read_text(encoding="utf-8")
    return list(tokenizer.encode(text))


def build_lm_dataloaders(
    train_params: TrainParams,
    data_params: DataParams,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    tokenizer = TextTokenizer(logger=logger)
    tokenizer.load_tiktoken("gpt2")

    token_ids = _load_tokens(data_params, tokenizer)
    if len(token_ids) < (data_params.max_length + 2):
        raise ValueError(
            f"Not enough tokens ({len(token_ids)}) for block_size={data_params.max_length}. "
            "Provide more data or reduce dataset.max_length."
        )

    split_idx = int(len(token_ids) * float(data_params.train_ratio))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_ds = TokenBlockDataset(train_tokens, block_size=int(data_params.max_length))
    val_ds = TokenBlockDataset(val_tokens, block_size=int(data_params.max_length))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"Train/val split too small for block_size={data_params.max_length}. "
            f"Got train_blocks={len(train_ds)} val_blocks={len(val_ds)}. "
            "Increase data size or adjust dataset.train_ratio/max_length."
        )

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    shuffle = bool(data_params.shuffle)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_params.batch_size),
        shuffle=(train_sampler is None and shuffle),
        sampler=train_sampler,
        num_workers=int(data_params.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_params.batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(data_params.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def _lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits of shape (B, T, V), got {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"Expected labels of shape (B, T), got {tuple(labels.shape)}")
    vocab = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1))


class Trainer:
    def __init__(
        self,
        TrainParams: TrainParams,
        DataParams: DataParams,
        ModelParams: ModelParams,
        OutputParams: Optional[OutputParams] = None,
    ):
        self.TrainParams = TrainParams
        self.DataParams = DataParams
        self.ModelParams = ModelParams
        self.OutputParams = OutputParams or OutputParams()

    def train_single_epoch(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, device: torch.device) -> float:
        model.train()
        total_loss = 0.0
        steps = 0

        grad_accum = max(1, int(getattr(self.TrainParams, "grad_accum_steps", 1)))
        optimizer.zero_grad(set_to_none=True)

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(input_ids)
            loss = _lm_loss(logits, labels)
            (loss / grad_accum).backward()

            total_loss += float(loss.detach().cpu().item())
            steps += 1

            if (step + 1) % grad_accum == 0:
                max_norm = getattr(self.TrainParams, "max_grad_norm", None)
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # flush remaining grads
        if steps % grad_accum != 0:
            max_norm = getattr(self.TrainParams, "max_grad_norm", None)
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return total_loss / max(1, steps)

    @torch.no_grad()
    def val_single_epoch(self, model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
        model.eval()
        total_loss = 0.0
        steps = 0
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(input_ids)
            loss = _lm_loss(logits, labels)
            total_loss += float(loss.detach().cpu().item())
            steps += 1
        return total_loss / max(1, steps)

    def train_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device,
        *,
        val_loader: Optional[DataLoader] = None,
        summary: Optional[TrainSummary] = None,
    ) -> None:
        for epoch in range(int(self.TrainParams.epochs)):
            train_loss = self.train_single_epoch(model, optimizer, train_loader, device)
            if summary is not None:
                summary.log_epoch(epoch + 1, train_loss, split="train")

            if val_loader is not None:
                val_loss = self.val_single_epoch(model, val_loader, device)
                if summary is not None:
                    summary.log_epoch(epoch + 1, val_loss, split="val")


def _ddp_mean(value: float, device: torch.device, world_size: int) -> float:
    if not dist.is_available() or not dist.is_initialized() or world_size <= 1:
        return float(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / world_size).item())


def _ddp_worker(
    rank: int,
    world_size: int,
    train_params: TrainParams,
    data_params: DataParams,
    model_params: ModelParams,
    output_params: OutputParams,
    gpu_ids: list[int],
    backend: str,
    master_addr: str,
    master_port: str,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    if backend == "nccl":
        gpu_id = int(gpu_ids[rank])
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.manual_seed(int(train_params.seed) + int(rank))

    train_loader, val_loader, train_sampler, _ = build_lm_dataloaders(
        train_params,
        data_params,
        rank=rank,
        world_size=world_size,
    )

    model = GetModel(model_params).to(device)
    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=float(train_params.learning_rate),
        weight_decay=float(train_params.weight_decay),
    )

    summary = None
    if rank == 0:
        summary = TrainSummary(
            {
                "save_dir": output_params.save_dir,
                "print_output": bool(output_params.print_output),
                "output_path": output_params.output_path,
            },
            logger=logger,
        )

    trainer = Trainer(train_params, data_params, model_params, output_params)
    for epoch in range(int(train_params.epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = trainer.train_single_epoch(ddp_model, optimizer, train_loader, device)
        val_loss = trainer.val_single_epoch(ddp_model, val_loader, device)

        train_loss = _ddp_mean(train_loss, device=device, world_size=world_size)
        val_loss = _ddp_mean(val_loss, device=device, world_size=world_size)

        if summary is not None:
            summary.log_epoch(epoch + 1, train_loss, split="train")
            summary.log_epoch(epoch + 1, val_loss, split="val")

    dist.destroy_process_group()


class DDPTrainer:
    def __init__(
        self,
        TrainParams: TrainParams,
        DataParams: DataParams,
        ModelParams: ModelParams,
        OutputParams: OutputParams,
    ):
        self.TrainParams = TrainParams
        self.DataParams = DataParams
        self.ModelParams = ModelParams
        self.OutputParams = OutputParams

        self.debug = bool(getattr(self.TrainParams, "debug", False))
        self.backend = "gloo" if (self.debug or (not torch.cuda.is_available())) else "nccl"

        gpu_ids = list(getattr(self.TrainParams, "gpu", []))
        if self.backend == "nccl":
            if not gpu_ids:
                # safe default: use a single least-used GPU
                ordered = get_gpus_by_least_usage(return_stats=False)
                gpu_ids = [int(ordered[0])] if ordered else [0]
        else:
            gpu_ids = []

        self.gpu_ids = gpu_ids
        self.world_size = max(1, len(self.gpu_ids)) if self.backend == "nccl" else 1

    def train_model(self, *, master_addr: str = "127.0.0.1", master_port: str = "12355") -> None:
        os.makedirs(self.OutputParams.save_dir, exist_ok=True)

        if self.world_size <= 1:
            device = torch.device(f"cuda:{self.gpu_ids[0]}" if (self.backend == "nccl") else "cpu")
            if device.type == "cuda":
                torch.cuda.set_device(device)

            train_loader, val_loader, _, _ = build_lm_dataloaders(self.TrainParams, self.DataParams)
            model = GetModel(self.ModelParams).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.TrainParams.learning_rate),
                weight_decay=float(self.TrainParams.weight_decay),
            )
            summary = TrainSummary(
                {
                    "save_dir": self.OutputParams.save_dir,
                    "print_output": bool(self.OutputParams.print_output),
                    "output_path": self.OutputParams.output_path,
                },
                logger=logger,
            )
            Trainer(self.TrainParams, self.DataParams, self.ModelParams, self.OutputParams).train_model(
                model,
                optimizer,
                train_loader,
                device,
                val_loader=val_loader,
                summary=summary,
            )
            return

        mp.spawn(
            _ddp_worker,
            nprocs=self.world_size,
            args=(
                self.world_size,
                self.TrainParams,
                self.DataParams,
                self.ModelParams,
                self.OutputParams,
                self.gpu_ids,
                self.backend,
                master_addr,
                master_port,
            ),
            join=True,
        )
