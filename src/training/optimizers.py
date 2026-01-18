import torch.optim as optim

try:
    from timm.optim.lion import Lion
except ImportError:
    Lion = None


def create_optimizers(model, optimizer_config):
    """Create main and auxiliary optimizers based on configuration."""
    main_params, aux_params = [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bit_estimator" in name:
            aux_params.append(param)
        else:
            main_params.append(param)
    
    optimizer_type = optimizer_config.optimizer_type.lower()
    
    if optimizer_type == "lion":
        if Lion is None:
            raise ImportError("Lion optimizer requested but could not be imported from timm.optim.lion.")
        optimizer = Lion(
            main_params, 
            lr=optimizer_config.base_lr, 
            weight_decay=optimizer_config.weight_decay
        )
        optimizer_aux = Lion(
            aux_params, 
            lr=optimizer_config.aux_lr, 
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(
            main_params, 
            lr=optimizer_config.base_lr, 
            weight_decay=optimizer_config.weight_decay
        )
        optimizer_aux = optim.AdamW(
            aux_params, 
            lr=optimizer_config.aux_lr, 
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(main_params, lr=optimizer_config.base_lr)
        optimizer_aux = optim.Adam(aux_params, lr=optimizer_config.aux_lr)
    else:
        print(f"Warning: Unknown optimizer '{optimizer_type}', defaulting to Adam")
        optimizer = optim.Adam(main_params, lr=optimizer_config.base_lr)
        optimizer_aux = optim.Adam(aux_params, lr=optimizer_config.aux_lr)
    
    print(f"Using {optimizer_type} optimizer for main parameters")
    print(f"Using {optimizer_type} optimizer for auxiliary parameters")
    
    return optimizer, optimizer_aux