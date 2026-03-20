import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_name():
    device = get_device()
    return device.type

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

import contextlib

@contextlib.contextmanager
def custom_autocast(device_type, dtype, enabled=True):
    if device_type == "mps":
        # MPS doesn't support generic autocast as smoothly,
        # but PyTorch supports torch.autocast("cpu") or torch.autocast("cuda").
        # For MPS, we generally don't use autocast or use standard inference precision manually.
        yield
    else:
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
            yield

def get_autocast_decorator(device_type, enabled=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if device_type == "mps":
                return func(*args, **kwargs)
            else:
                with torch.autocast(device_type=device_type, enabled=enabled):
                    return func(*args, **kwargs)
        return wrapper
    return decorator
