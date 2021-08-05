import torch

@torch.jit.script
def swish(x):
    return x * x.sigmoid()

