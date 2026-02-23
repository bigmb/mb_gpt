## Get details about the model, such as number of parameters, layers, etc.
from mb.utils.logging import logg

__all__ = ['ModelSummary']

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
