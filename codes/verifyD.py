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

class EdgeAgg_s(MessagePassing):
    # This is a simple edge aggregation to control # of edges that is not working.
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



class EdgeAgg_z(MessagePassing):
    # This is a simple edge aggregation to control # of edges that is not working.
    # For testing purpose.
    def __init__(self, aggr='add', device='cpu'):
        super(EdgeAgg_z, self).__init__(aggr=aggr)
        self.device = device

    def forward(self, x, edge_index, edge_attr):
        """
        """
        edge_input = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if edge_attr.size(1) == 4:
            in_x = (edge_input * torch.FloatTensor([1.0, 2.0, 3.0, 1.5]).to(self.device).view(1, -1)).sum(axis=1).view(-1, 1)
        else:
            in_x = (edge_input * torch.FloatTensor([1.0, 2.0, 3.0]).to(self.device).view(1, -1)).sum(axis=1).view(-1, 1)
        x = ((x[:, :9] * torch.FloatTensor([4, 3, 2, 1, 6.0, 2, 1, 1, 1]).to(self.device).view(1, -1)).sum(axis=1) + \
               (x[:, 9:12] * torch.FloatTensor([0, 1, -1]).to(self.device).view(1, -1)).sum(axis=1) ).view(-1, 1)

        if edge_attr.size(1) == 4:
            in_x2 = (edge_input * torch.FloatTensor([0, 0, 0, 1.0]).to(self.device).view(1, -1) > 0.5).type(torch.FloatTensor).sum(axis=1).view(-1, 1)
            x2 = (x[:, -1:] * torch.FloatTensor([1]).to(self.device).view(1, -1)).sum(axis=1).view(-1, 1)

            out = torch.cat([in_x - x, in_x2 - x2, x2 - in_x2], axis=1)
        else:
            out = torch.cat([in_x - x], axis=1)
        return out

    def message(self, x_i, edge_attr):
        return edge_attr



class SimPass(MessagePassing):
    def __init__(self, in_dim, inner=True, aggr='add', device='cpu'):
        super(SimPass, self).__init__(aggr=aggr)
        self.device = device
        self.in_dim = in_dim
        self.inner = inner

    def forward(self, x, edge_index):
        """
        """
        x_in = self.propagate(edge_index, x=x)
        if self.inner:
            return (x_in > 0) + (x > 0)
        else:
            return (x_in)

    def message(self, x_j):
        msg = (x_j > 0).type(torch.FloatTensor) # (self.edge_ln(edge_attr).view(-1, self.out_dim, self.in_dim) * x_j.view(-1, 1, self.in_dim)).sum(dim=2)
        return msg

class Cycle_ind(nn.Module):
    def __init__(self, in_dim, aggr='add', device='cpu'):
        super(Cycle_ind, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.sim_pass = SimPass(in_dim, device=device)
        self.sim_pass_f = SimPass(in_dim, device=device, inner=False)

    def forward(self, x, edge_index):
        """
        """
        use_len = x.size(0)
        ini_x = torch.diag(torch.ones(use_len))
        use_x = torch.diag(torch.ones(use_len)*(x[:, -1] > 0.5))
        use_xp = self.sim_pass(use_x, edge_index = edge_index)
        use_xp = self.sim_pass(use_xp, edge_index = edge_index)
        use_xf = self.sim_pass_f(torch.cat([ini_x, use_xp], axis=1), edge_index = edge_index)
        out_x = ((ini_x == 0) * (use_xf[:, :use_len] == 0) * (use_xf[:, use_len:] > 0)).sum(axis=1)
        return out_x



class verifyD(nn.Module):
    """Class to verify validity of molecule, using GCN.
    
    For testing purpose.
    """
    def __init__(self, mean_x=[], mean_em=[], es_target=21, x_coef=3, em_coef=3, es_coef=1, \
                add_inflow=False, add_inflow_coef=4, device='cpu'):
        super(verifyD, self).__init__()
        
        self.mean_x = torch.FloatTensor(mean_x).view(1, len(mean_x)).to(device)
        self.mean_em = torch.FloatTensor(mean_em).view(1, len(mean_em)).to(device)
        self.es_target = es_target
        self.x_coef = x_coef
        self.em_coef = em_coef
        self.es_coef = es_coef
        self.add_inflow = add_inflow
        self.add_inflow_coef = add_inflow_coef
        if add_inflow:
            self.inflow = EdgeAgg_s(device=device)
        self.device = device

    def forward(self, x, batch, edge_attr, edge_index, edge_batch):
        obtained_x = global_mean_pool(x, batch)
        self.loss1 = torch.pow(obtained_x - self.mean_x, 2).sum(axis=1)*self.x_coef
        obtained_e = global_mean_pool(edge_attr, edge_batch)
        self.loss2 = torch.pow(obtained_e - self.mean_em, 2).sum(axis=1)*self.em_coef
        coefs = (edge_index[0].to(self.device) > -1).type(torch.FloatTensor)
        added_coefs = global_add_pool(coefs.to(self.device), edge_batch.to(self.device))
        relu_res = F.leaky_relu(added_coefs - self.es_target, 0.3)
        self.loss3 = torch.pow(relu_res, 2)*self.es_coef
        out_loss = self.loss1 + self.loss2 + self.loss3
        if self.add_inflow:
            max_flow = global_max_pool(self.inflow(x=x, edge_index=edge_index, edge_attr=edge_attr), batch)
            self.loss4 = F.relu(max_flow)*self.add_inflow_coef

            out_loss += self.loss4.mean(axis=1)
        return -out_loss


class verifyZ(nn.Module):
    def __init__(self, mean_x=[], mean_em=[], es_target=21, x_coef=3, em_coef=3, es_coef=1, \
                add_inflow=False, add_inflow_coef=4, device='cpu', add_cycle=False, cycle_coef=3):
        super(verifyZ, self).__init__()
        self.mean_x = torch.FloatTensor(mean_x).view(1, len(mean_x)).to(device)
        self.mean_em = torch.FloatTensor(mean_em).view(1, len(mean_em)).to(device)
        self.es_target = es_target
        self.x_coef = x_coef
        self.em_coef = em_coef
        self.es_coef = es_coef
        self.add_inflow = add_inflow
        self.add_inflow_coef = add_inflow_coef
        self.cycle_coef = cycle_coef
        self.add_cycle = add_cycle
        if add_inflow:
            self.inflow = EdgeAgg_z(device=device)
        if add_cycle:
            self.cycleflow = Cycle_ind(in_dim=1, device=device)
        self.device = device

    def forward(self, x, batch, edge_attr, edge_index, edge_batch):
        obtained_x = global_mean_pool(x, batch)
        self.loss1 = torch.pow(obtained_x - self.mean_x, 2).sum(axis=1)*self.x_coef
        obtained_e = global_mean_pool(edge_attr, edge_batch)
        self.loss2 = torch.pow(obtained_e - self.mean_em, 2).sum(axis=1)*self.em_coef
        coefs = (edge_index[0].to(self.device) > -1).type(torch.FloatTensor)
        added_coefs = global_add_pool(coefs.to(self.device), edge_batch.to(self.device))
        relu_res = F.leaky_relu(added_coefs - self.es_target, 0.3)
        self.loss3 = torch.pow(relu_res, 2)*self.es_coef
        out_loss = self.loss1 + self.loss2 + self.loss3
        if self.add_inflow:
            max_flow = global_max_pool(self.inflow(x=x, edge_index=edge_index, edge_attr=edge_attr), batch)
            self.loss4 = F.relu(max_flow)*self.add_inflow_coef

            out_loss += self.loss4.sum(axis=1)
        if self.add_cycle:
            output = self.cycleflow(x, edge_index)
            num_aroma = global_add_pool(x[:, -1] > 0.5, batch).type(torch.FloatTensor)
            num_cycle = global_add_pool(output > 0, batch).type(torch.FloatTensor)
            out_loss += torch.pow(F.leaky_relu(num_cycle - num_aroma, 0.02), 2) * self.cycle_coef
        return -out_loss
