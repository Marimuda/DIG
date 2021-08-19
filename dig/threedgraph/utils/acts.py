import torch

@torch.jit.script
def swish(x):
    return (x * x.sigmoid())

@torch.jit.script
def swish_scaled(x):
    return (x * x.sigmoid()) * 1.666666666  # scaling swish with 1/0.6
