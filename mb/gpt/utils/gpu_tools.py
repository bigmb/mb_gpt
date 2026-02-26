import torch

__all__ = ["gpu_count", "get_gpus_by_least_usage"]

def get_gpus_by_least_usage(return_stats: bool = False):
    """
    Get GPUs ordered by least memory usage.

    Usage for each GPU is computed as:
        usage_ratio = (total_bytes - free_bytes) / total_bytes

    Args:
        return_stats (bool):
            - False: return only GPU indices in ascending usage order.
            - True: return list of dict stats sorted by ascending usage.

    Returns:
        list:
            - If return_stats=False: [gpu_idx0, gpu_idx1, ...]
            - If return_stats=True: [
                  {
                    "gpu_id": int,
                    "name": str,
                    "free_gb": float,
                    "used_gb": float,
                    "total_gb": float,
                    "usage_ratio": float,
                  },
                  ...
              ]
    """
    if not torch.cuda.is_available():
        return []

    gpu_stats = []
    for gpu_id in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=gpu_id)
        used_bytes = total_bytes - free_bytes

        gpu_stats.append(
            {
                "gpu_id": gpu_id,
                "name": torch.cuda.get_device_name(gpu_id),
                "free_gb": free_bytes / (1024**3),
                "used_gb": used_bytes / (1024**3),
                "total_gb": total_bytes / (1024**3),
                "usage_ratio": (used_bytes / total_bytes) if total_bytes > 0 else 1.0,
            }
        )

    gpu_stats.sort(key=lambda item: (item["usage_ratio"], -item["free_gb"]))

    if return_stats:
        return gpu_stats

    return [item["gpu_id"] for item in gpu_stats]


def gpu_count(arrange: bool = True) -> list:
    """
    Return available GPU count or GPUs ordered by least usage.

    Args:
        arrange (bool):
            - True: returns GPU ids sorted by least usage.
            - False: returns [0, 1, ..., device_count-1].

    Returns:
        list: GPU ids.
    """
    if not torch.cuda.is_available():
        return []

    if arrange:
        return get_gpus_by_least_usage(return_stats=False)

    return list(range(torch.cuda.device_count()))
