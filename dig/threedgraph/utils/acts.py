import torch

@torch.jit.script
def swish(x):
    return x * x.sigmoid()

@torch.jit.script
def silu(input):
    return (input * torch.sigmoid(input)) * 1.6666666  # scaling silu output with 1/0.6
