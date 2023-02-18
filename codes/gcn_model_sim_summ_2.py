# Some customized GCN layers.
# including
#   RLGCN: edge-conditional MPNN
#   EdgeAgg_4: For a nodes, aggregated with all neighbor edges features.
#   EdgeAgg_s: for testing.

from typing import final
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch_geometric
from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing, GINConv, unpool
from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool import knn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch.distributions import Categorical
from unpool_utils import assemble_skip_z, convert_Batch_to_datalist 

def custom_act_func(x, act_func):
    """Activation function..

    Args:
        x (tensor)
        act_func (str): choose from tanh, leaky_relu, relu, sigmoid. Otherwise, identity.

    Returns:
        result acting on x
    """
    if act_func == 'tanh':
        return torch.tanh(x)
    elif act_func == 'leaky_relu':
        return F.leaky_relu(x, 0.1)
    elif act_func == 'relu':
        return torch.relu(x)
    elif act_func == 'sigmoid':
        return torch.sigmoid(x)
    else:
        return x

class RLGCN(MessagePassing):
    """Edge conditional MPNN

    Args:
        in_dim (int): Input node feature dimension
        out_dim (int): Desired output node feature dimension
        edge_dim (int): Input edge dimension
        aggr (str, optional): How to aggregate all neighbor features connected to a node. Defaults to 'mean'.

    """
    def __init__(self, in_dim, out_dim, edge_dim, aggr='mean'):
        super(RLGCN, self).__init__(aggr=aggr)
        self.ln1 = nn.Linear(in_dim, out_dim)
        self.edge_ln = nn.Linear(edge_dim, in_dim*out_dim, bias=False)
        self.in_dim = in_dim
        self.out_dim = out_dim


    def forward(self, x, edge_index, edge_attr):
        """
        """
        return self.ln1(x) + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg = (self.edge_ln(edge_attr).view(-1, self.out_dim, self.in_dim) * x_j.view(-1, 1, self.in_dim)).sum(dim=2)
        return msg

class EdgeAgg_4(MessagePassing):
    """Simply calculate the edge aggregated at each node,     
    i.e. output_j = Wx_j + b + W_e AGG(e_{ij})

    Args:
        in_dim (int): Input node feature dimension
        out_dim (int): Desired output node feature dimension
        edge_dim (int): Input edge dimension
        aggr (str, optional): How to aggregate all edges connected to a node. Defaults to 'mean'.
    """
    def __init__(self, in_dim, out_dim, edge_dim, aggr='add'):
        super(EdgeAgg_4, self).__init__(aggr=aggr)
        self.ln1 = nn.Linear(in_dim, out_dim)
        self.ln2 = nn.Linear(edge_dim, out_dim, bias=False)
        self.ln0 = nn.Linear(out_dim*2, out_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        """
        x = self.ln1(x)
        x1 = torch.cat([x, self.propagate(edge_index, x=x, edge_attr=edge_attr)], axis=1)
        return F.leaky_relu(self.ln0(F.leaky_relu(x1, 0.1)), 0.1)

    def message(self, x_i, edge_attr):
        weight = self.ln2(edge_attr)
        return weight# + x_i


class EdgeAgg_s(MessagePassing):
    # This is a simple edge aggregation to control # of edges that is not working.
    """For testing purpose. ONLY used for QM9, having 4 types of nodes (C/N/O/F) and 3 types of edges (SINGLE/DOUBLE/TRIPLE)
    Calculating the number of electron bonded to an edge minus the maximum available bonds for the node
    
    A negative value means invalidity.
    """
    def __init__(self, aggr='add', device='cpu'):
        super(EdgeAgg_s, self).__init__(aggr=aggr)
        self.device = device

    def forward(self, x, edge_index, edge_attr):
        """
        """
        edge_input = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        in_x = (edge_input * torch.arange(1, 4).to(self.device).view(1, -1)).sum(axis=1).view(-1, 1)
        x = (x * torch.arange(4, 0, -1).to(self.device).view(1, -1)).sum(axis=1).view(-1, 1)

        return in_x - x

    def message(self, x_i, edge_attr):
        return edge_attr
