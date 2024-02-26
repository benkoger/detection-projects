import torch

def get_lr_scheduler(optimizer, name, **cfg_lr):
    """ Choose specifed learning rate scheduler based on name."""
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg_lr)
    if name == "ReduceOnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, **cfg_lr)