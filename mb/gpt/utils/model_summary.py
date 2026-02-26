## Get details about the model, such as number of parameters, layers, etc.
from mb.utils.logging import logg
import torch

__all__ = ['ModelSummary', 'can_fit_training_step','']

def ModelSummary(model,logger=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    num_layers = len(list(model.modules()))
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'number_of_layers': num_layers
    }
    
    logg.info(f"Model Summary: {summary}", logger=logger)
    
    return summary

def can_fit_training_step(
    model,
    batch_size=10,
    vlm_input_dim=128,
    text_input_dim=256,
    dtype=torch.float32,
    device='cuda:0',
    lr=1e-3,
):
    if not torch.cuda.is_available():
        print('CUDA is not available on this machine.')
        return False

    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vlm_x = torch.randn(batch_size, vlm_input_dim, dtype=dtype, device=device)
    text_x = torch.randn(batch_size, text_input_dim, dtype=dtype, device=device)

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        optimizer.zero_grad(set_to_none=True)
        out = model(vlm_x, text_x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        peak_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        print(f'Training step fits on {device}. Peak allocated: {peak_mb:.2f} MB')
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'OOM on {device}: {e}')
            return False
        raise 

def can_fit_in_single_gpu(
    model,
    batch_size=10,
    vlm_input_dim=128,
    text_input_dim=256,
    dtype=torch.float32,
    device='cuda:0',
):
    if not torch.cuda.is_available():
        print('CUDA is not available on this machine.')
        return False

    model = model.to(device).eval()
    vlm_x = torch.randn(batch_size, vlm_input_dim, dtype=dtype, device=device)
    text_x = torch.randn(batch_size, text_input_dim, dtype=dtype, device=device)

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        with torch.no_grad():
            _ = model(vlm_x, text_x)
        peak_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        print(f'Fits on {device}. Peak allocated: {peak_mb:.2f} MB')
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'OOM on {device}: {e}')
            return False
        raise    