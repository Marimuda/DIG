from torch import Tensor as Tensor
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, MessagePassing

small_gain = 0.01


def pooling(x: Tensor, batch_info, method):
    if method == 'add':
        return global_add_pool(x, batch_info['batch'], batch_info['num_graphs'])
    elif method == 'mean':
        return global_mean_pool(x, batch_info['batch'], batch_info['num_graphs'])
    elif method == 'max':
        return global_max_pool(x, batch_info['batch'], batch_info['num_graphs'])
    else:
        raise ValueError("Pooling method not implemented")


class BatchNorm(nn.Module):
    def __init__(self, channels: int, use_x: bool):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        self.use_x = use_x

    def reset_parameters(self):
        self.bn.reset_parameters()

    def forward(self, u):
        if self.use_x:
            return self.bn(u)
        else:
            return self.bn(u.transpose(1, 2)).transpose(1, 2)


class EdgeCounter(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, batch, batch_size):
        n_edges = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return global_mean_pool(n_edges, batch, batch_size)[batch]
