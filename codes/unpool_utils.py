# Some util functions for unpooling layers.
# Useful function is
#   assemble_skip_z

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool import knn
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch.distributions.categorical import Categorical

def assemble_skip_z(usex, batch, device='cpu'):
    '''
    usex: a tensor of shape (N, k, d): N is number of graphs, k is number of nodes in each graph, d is dim of feature.
    batch: size of M, value taking as 0 - N-1; each graph will have at most k replicates.
        Note: if one graph have more than 1 nodes, we will take those 1 by 1 (in k's dimension)

    Return

    A tensor of shape M by d.

    '''
    total_batch = batch.max() + 1
    N, k, d = usex.shape
    collected = torch.gather(usex, 0, batch.unsqueeze(-1).unsqueeze(-1).repeat(1, k, d).to(device))
    node_orders = ((batch.unsqueeze(-1) == torch.arange(total_batch).unsqueeze(0)).cumsum(axis=0) - 1)\
            [(batch.unsqueeze(-1) == torch.arange(total_batch).unsqueeze(0))].unsqueeze(-1).unsqueeze(-1).repeat(1, k, d).to(device)
    result = torch.gather(collected, 1, node_orders)
    return result[:, 0, :]

def convert_Batch_to_datalist(x, edge_index, edge_attr=None, batch=None, edge_batch=None, device='cpu'):
    """Given a data batch from torch_geometric, return a list of Data.

    Args:
        x (tensor): All nodes features (n*d), n nodes, d dimension feature.
        edge_index (tensor of long): 2*m, m edges.
        edge_attr (tensor): m*p, m edges, p edge features.
        batch (tensor of long): n size.
        edge_batch (tensor of long): m size.

    Returns:
        a list of Data, each Data is torch_geometric.utils.Data.
    """
    result = []
    for j in range(max(batch) + 1):
        use_ids = (batch == j).to(device)
        use_edge_ids = (edge_batch == j).to(device)
        use_x = x[use_ids].to(device)
        use_edge_index = edge_index[:, use_edge_ids].to(device)
        if edge_attr is not None:
            use_edge_attr = edge_attr[use_edge_ids].to(device)
        remap_ids = torch.arange(len(x)).to(device)[use_ids]
        new_ids = torch.arange(len(use_x)).to(device)
        for k in range(len(new_ids)):
            use_edge_index[use_edge_index == remap_ids[k]] = new_ids[k].item()
        if edge_attr is not None:
            result.append(Data(x=use_x, edge_index=use_edge_index, edge_attr=use_edge_attr))
        else:
            result.append(Data(x=use_x, edge_index=use_edge_index))
    return result
