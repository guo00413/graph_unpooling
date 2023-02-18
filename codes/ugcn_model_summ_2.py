# One discriminator implementation, used for QM9.

from typing import final
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing

def custom_act_func(x, act_func):
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
    """A MPNN, but only using neighbor edges features to update node feature.

    Args:
        in_dim (int): input node feature
        out_dim (int): output node feature
        edge_dim (int): input edge feature
        aggr (string): type of aggregation for neighbor edge features; Defaults to 'add'.
    """
    def __init__(self, in_dim, out_dim, edge_dim, aggr='add'):
        super(EdgeAgg_4, self).__init__(aggr=aggr)
        self.ln1 = nn.Linear(in_dim, out_dim)
        self.ln2 = nn.Linear(edge_dim, out_dim, bias=False)
        self.ln0 = nn.Linear(out_dim*2, out_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Input: x - node features
               edge_index - 2*m long tensor for edge connection.
               edge_attr - m*d_w tensor for edge features.
        """
        x = self.ln1(x)
        x1 = torch.cat([x, self.propagate(edge_index, x=x, edge_attr=edge_attr)], axis=1)
        return F.leaky_relu(self.ln0(F.leaky_relu(x1, 0.1)), 0.1)

    def message(self, x_i, edge_attr):
        weight = self.ln2(edge_attr) # only passing the edge feature.
        return weight # Here we only pass neighbor edges 


class EdgeAgg_s(MessagePassing):
    # This is a simple edge aggregation to control # of edges that is not working.
    # Only applicable for QM9.
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


class pre_GCNModel_edge_3eos(nn.Module):
    """
    A classifier GCN for GAN, predicting if one is real or fake data.
    
    uses edge-conditional MPNN and global readout to predict a numeric value from a graph.
    """
    def __init__(self, in_dim, hidden_dim, edge_dim, edge_hidden_dim, lin_hidden_dim=None, out_hidden_dim=256, device='cpu', \
                check_batch=None, useBN=False, droprate=None, pool_method='trivial', \
                add_edge_link=False, add_conv_link=False, outBN=False, out_drop=0.3, \
                out_divide=4.0, add_edge_agg=True, real_trivial=False, final_layers=2, \
                add_trivial_feature=True, ln_comp=False):
        super(pre_GCNModel_edge_3eos, self).__init__()
        self.add_trivial_feature = add_trivial_feature
        self.final_layers = final_layers
        if lin_hidden_dim is None:
            lin_hidden_dim = hidden_dim
        self.lin_hidden_dim = lin_hidden_dim

        if add_conv_link:
            # By having this, we will add MPNN layers to help the prediction.
            self.conv1 = NNConv(in_dim, hidden_dim, nn=nn.Linear(edge_dim, in_dim*hidden_dim))
            self.conv2 = NNConv(hidden_dim, hidden_dim, nn=nn.Linear(edge_dim, hidden_dim*hidden_dim)) 
            in_dim_hidden = hidden_dim + in_dim
            if ln_comp:
                # This add to a multilayer preceptrons for final readout using sigmoid + tanh signals.
                if useBN:
                    self.sig_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.BatchNorm1d(lin_hidden_dim), \
                        nn.LeakyReLU(0.05), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
                    )
                    self.tanh_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.BatchNorm1d(lin_hidden_dim), \
                        nn.LeakyReLU(0.05), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
                    )
                else:
                    self.sig_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.LeakyReLU(0.05), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
                    )
                    self.tanh_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.LeakyReLU(0.05), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
                    )
            else:
                # Otherwise, we will use a simple linear layer for final readout using sigmoid + tanh signals.
                self.sig_ln = nn.Linear(in_dim_hidden, lin_hidden_dim)
                self.tanh_ln = nn.Linear(in_dim_hidden, lin_hidden_dim)

        self.add_edge_agg = add_edge_agg
        if add_edge_agg:
            # This is to add a simple aggregation for edge features.
            self.edge_res = EdgeAgg_s(device=device)
        if add_edge_link:
            # This is to add some layers to aggregate the edge features for each node.
            self.edge_agg1 = EdgeAgg_4(in_dim, edge_hidden_dim, edge_dim)
            self.edge_agg2 = EdgeAgg_4(edge_hidden_dim, edge_hidden_dim, edge_dim)

        self.droprate = droprate
        self.real_trivial = real_trivial
        if not real_trivial:
            self.ln_x0 = nn.Linear(in_dim, in_dim)
            self.ln_e0 = nn.Linear(edge_dim, edge_dim)
            self.ln_e1 = nn.Linear(edge_dim, edge_dim)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.device = device
        self.check_batch = check_batch

        self.useBN = useBN
        if self.add_trivial_feature:
            self.out_dim = self.in_dim + self.edge_dim*2
        else:
            self.out_dim = 0
        self.add_edge_link = add_edge_link
        self.add_conv_link = add_conv_link

        if self.add_conv_link:
            self.out_dim += self.lin_hidden_dim
        if add_edge_link:
            self.out_dim += edge_hidden_dim
        if self.add_edge_agg:
            self.out_dim += 1

        if self.final_layers == 2:
            self.final_lin1 = nn.Linear(self.out_dim, out_hidden_dim)
            self.final_lin2 = nn.Linear(out_hidden_dim, 1)
            self.output_dim = 1
        elif self.final_layers == 1:
            self.final_lin1 = nn.Linear(self.out_dim, 1)
            self.output_dim = 1
        else:
            self.output_dim = self.out_dim

        if useBN:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            self.bn4 = nn.BatchNorm1d(hidden_dim)
            self.edgebn1 = nn.BatchNorm1d(edge_hidden_dim)
            self.edgebn2 = nn.BatchNorm1d(edge_hidden_dim)
        if outBN:
            self.out_bn = nn.BatchNorm1d(self.out_dim)

        self.outBN = outBN
        if droprate:
            self.drop1 = Dropout(droprate)
            self.drop2 = Dropout(droprate)
            self.drop3 = Dropout(droprate)
        if out_drop:
            self.drop_out = Dropout(out_drop)
        self.out_drop = out_drop
        self.final_ln0 = nn.Linear(self.out_dim, out_hidden_dim)
        self.final_ln = nn.Linear(out_hidden_dim, 1)
        self.out_divide = out_divide

        self.ex2_ind = 1.0
        self.x0_ind = 1.0
        self.e0_ind = 1.0
        self.e1_ind = 1.0
        self.conv_ind = 1.0
        self.eagg_ind = 1.0



    def forward(self, data=None, x=None, edge_index=None, edge_attr=None, batch=None, edge_batch=None):
        # use GCNs to predict if sample is real or fake.
        x0 = x
        if data is not None:
            x0 = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch
            if hasattr(data, 'edge_index_batch'):
                edge_batch = data.edge_index_batch
            else:
                edge_batch = data.edge_batch


        if self.add_edge_link:
            ex = self.edge_agg1(x0, edge_index=edge_index, edge_attr=edge_attr)
            if self.useBN:
                ex = self.edgebn1(ex)
            ex = F.leaky_relu(ex, 0.05)
            ex = self.edge_agg2(ex, edge_index=edge_index, edge_attr=edge_attr)
            if self.useBN:
                ex = self.edgebn2(ex)
            ex = F.leaky_relu(ex, 0.05)
            # sum_ex2_pool = global_max_pool(ex, batch)



        if self.real_trivial:
            sum_x0_pool = (global_add_pool(x0, batch))
            sum_e0_pool = (global_add_pool(edge_attr, edge_batch))*0.5
            sum_e1_pool = (global_mean_pool(edge_attr, edge_batch))
        else:
            sum_x0_pool = self.ln_x0(global_add_pool(x0, batch))
            sum_e0_pool = self.ln_e0(global_add_pool(edge_attr, edge_batch)*0.5)
            sum_e1_pool = self.ln_e1(global_mean_pool(edge_attr, edge_batch))

        if self.add_conv_link:
            x = self.conv1(x0, edge_index=edge_index, edge_attr=edge_attr)
            if self.useBN:
                x = self.bn1(x)
            x = F.leaky_relu(x, 0.05)
            x = self.conv2(x, edge_index=edge_index, edge_attr=edge_attr)
            if self.useBN:
                x = self.bn2(x)
            x = F.leaky_relu(x, 0.05)
            if self.droprate:
                x = self.drop2(x)

        # Readout:
            x = torch.cat([x, x0], axis=1)
            sx = torch.clip(self.sig_ln(x), -30, 30)
            x = 1/(1+torch.exp(sx))*torch.tanh(self.tanh_ln(x))
            if self.droprate:
                x = self.drop3(x)
            out_x = global_add_pool(x, batch=batch)

        if self.add_trivial_feature:
            sum_pool = torch.cat([sum_x0_pool*self.x0_ind, \
                                sum_e0_pool*self.e0_ind, \
                                sum_e1_pool*self.e1_ind], dim=1)
        else:
            sum_pool = torch.zeros(sum_x0_pool.size(0), 0).to(self.device)

        if self.add_edge_agg:
            ae = self.edge_res(x0, edge_index, edge_attr)
            sum_pool = torch.cat([sum_pool, self.eagg_ind*torch.relu(global_max_pool(ae, batch=batch))], dim=1)

        if self.add_conv_link:
            sum_pool = torch.cat([sum_pool, out_x*self.conv_ind], dim=1)
        if self.add_edge_link:
            sum_pool = torch.cat([sum_pool, global_add_pool(ex, batch)*self.conv_ind], dim=1)
        out_x = sum_pool
        
        if self.check_batch:
            if len(out_x) < self.check_batch:
                out_x = torch.cat([out_x, torch.zeros(self.check_batch - len(out_x), out_x.shape[1]).to(self.device)])
        
        if self.outBN:
            out_x = self.out_bn(out_x)
        if self.final_layers == 2:
            out_x = F.leaky_relu(self.final_lin1(out_x), 0.05)
            out_x = F.tanh(self.final_lin2(out_x)/self.out_divide)
        elif self.final_layers == 1:
            out_x = F.tanh(self.final_lin1(out_x)/self.out_divide)
        else:
            pass
        return out_x

