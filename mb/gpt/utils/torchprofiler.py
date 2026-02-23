
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import functools
from typing import Optional, Callable
from mb.utils.logging import logg

__all__ = ["TorchProfilerWrapper", "profile_torch", "quick_profile"]

class TorchProfilerWrapper:
    """
    A unified class to handle PyTorch profiling logic.
    Supports schedule-based profiling (wait -> warmup -> active).
    """
    def __init__(self, log_dir: str = "./log/profiler", wait: int = 1, warmup: int = 1, active: int = 3, repeat: int = 1):
        self.log_dir = log_dir
        self.schedule = torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat
        )
        self.activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        
    def get_profiler(self):
        return profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

def profile_torch(log_dir: str = "./log/decorator",logger=None):
    """
    A decorator to profile a specific function/method once.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Simple one-shot profile
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                on_trace_ready=tensorboard_trace_handler(log_dir)
            ) as prof:
                with record_function(f"## {func.__name__} ##"):
                    result = func(*args, **kwargs)
            logg.info(f"Profiling complete. Trace saved to {log_dir}", logger)
            return result
        return wrapper
    return decorator

def quick_profile(model: torch.nn.Module, inputs: torch.Tensor, log_dir: str = "./log/quick"):
    """
    Function for a quick one-off profile of a model forward pass.
    """
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=tensorboard_trace_handler(log_dir)) as prof:
        with record_function("model_inference"):
            return model(inputs)