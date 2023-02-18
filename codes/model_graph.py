# Including models for random graph and for protein.
# UL GAN and VAE GAN.

import numpy as np
import os
import scipy.optimize
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import long, optim, relu_
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv, GraphConv, global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing, GINConv, unpool
from unpool_utils import assemble_skip_z, convert_Batch_to_datalist 
from unpool_layers_simple_v2 import LinkLayer, EdgeAttrConstruct, UnpoolLayerEZ
from util_gnn import weight_initiate, generate_noise
from torch.distributions import Categorical
from torch_geometric.utils import add_self_loops, degree, to_undirected
from datetime import datetime, timedelta
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from torch_geometric.utils import dense_to_sparse



def convertA_to_data(X, A, gumbel=True, tau=0.5, hard=True, device='cpu', eps=1e-7, obtain_connected=False):
    """
    Given X, A and batch, return a list of torch_geometric.Data

    Return useX, edge_index, edge_attr
    """
    if gumbel:
        use_X = X
        use_A = F.gumbel_softmax(torch.log(A + eps), tau=tau, hard=hard)
    else:
        use_X = X
        use_A = A
    adj = (use_A[:, :, 0])
    # adj = ((adj + torch.transpose(adj, 0, 1)) > 0.5).type(torch.LongTensor)
    # Get use_ids
    final_X = torch.zeros(X.size(0), X.size(1) + 1)
    use_ids = (adj[torch.arange(X.size(0)), torch.arange(X.size(0))] > 0.5)
    final_X[:, X.size(1)] = adj[torch.arange(X.size(0)), torch.arange(X.size(0))]
    final_X[use_ids, :X.size(1)] = use_X[use_ids, :]#*final_X[:, X.size(1):]
    adj = adj * adj[torch.arange(X.size(0)), torch.arange(X.size(0))].view(1, -1) * adj[torch.arange(X.size(0)), torch.arange(X.size(0))].view(-1, 1)
    adj[torch.arange(X.size(0)), torch.arange(X.size(0))] = 0
    edge_index, edge_attr = dense_to_sparse(adj)

    return final_X, edge_index, edge_attr



class RGULGenerator(nn.Module):
    def __init__(self, in_dim, initial_dim, \
            hidden_dims=[[64], [128], [512]], \
            final_layer_dims=[], hidden_act='relu', \
            edge_hidden_dim=8, \
            leaky_relu_coef=0.05, device='cpu', \
            skip_z=True, skip_z_dims=None, \
            unpool_bn=True, link_bn=True, \
            link_act='leaky', unpool_para={}, \
            attr_bn=True, edge_out_dim=None, \
            fix_points=[0, 0], roll_ones=[2, 5], node_nums = [3, 6, 12], \
            node_feature=2, use_bn=True,
            final_bn=True,
            skipz_bn=True
            ):
        # By defaults:
        # 3->4->5
        # 3->6->12
        super(RGULGenerator, self).__init__()
        self.node_feature = node_feature
        self.skip_z = skip_z
        if skip_z and skip_z_dims is None:
            skip_z_dims = [d[-1]//4 for d in hidden_dims] + [hidden_dims[-1][-1]//4]

        self.skip_z_dims = skip_z_dims

        self.leaky_relu_coef = leaky_relu_coef
        self.hidden_dims = hidden_dims
        self.device = device
        self.in_dim = in_dim

        self.convs = nn.ModuleList([])
        self.unpools = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.use_bn = use_bn
        self.acts = nn.ModuleList([])
        self.edge_attr_layers = nn.ModuleList([])
        self.skipzs = nn.ModuleList([])
        self.ln0 = nn.Linear(in_dim, 3*initial_dim)
        self.first_edge_link = LinkLayer(initial_dim, useBN=link_bn, final_act=link_act)
        self.edge_attr_layers.append(EdgeAttrConstruct(initial_dim, edge_hidden_dim, edge_hidden_dim, useBN=attr_bn))
        self.initial_dim = initial_dim
        self.bn0 = nn.BatchNorm1d(initial_dim) # here is not 3*initial_dim, so we are taking normalization in all nodes dim.
        pre_dim = initial_dim
        self.skip_z = skip_z
        
        self.node_nums = node_nums
        for j, hds in enumerate(hidden_dims):
            use_convs = nn.ModuleList([])
            use_bns = nn.ModuleList([])
            use_acts = nn.ModuleList([])
            for hd in hds:
                edge_link = nn.Sequential(nn.Linear(edge_hidden_dim, hd*pre_dim))
                use_convs.append(NNConv(pre_dim, hd, nn=edge_link))
                if self.use_bn:
                    use_bns.append(nn.BatchNorm1d(hd))
                if hidden_act == 'relu':
                    use_acts.append(nn.ReLU())
                elif hidden_act == 'sigmoid':
                    use_acts.append(nn.Sigmoid())
                else:
                    use_acts.append(nn.LeakyReLU(self.leaky_relu_coef))
                pre_dim = hd
            self.convs.append(use_convs)
            self.bns.append(use_bns)
            self.acts.append(use_acts)
            if j < len(hidden_dims) - 1:
                self.unpools.append(UnpoolLayerEZ(pre_dim, pre_dim//2, edge_dim=edge_hidden_dim, \
                            fix_point=fix_points[j], useBN=unpool_bn, \
                            inner_link=None, outer_link=None, \
                            link_bn=link_bn, link_act=link_act, \
                            roll_ones=roll_ones[j], **unpool_para))
                pre_dim -= pre_dim//4
                if skip_z:
                    skip_j = j + 1
                    if skipz_bn:
                        self.skipzs.append(nn.Sequential(nn.Linear(self.in_dim, skip_z_dims[skip_j]*10), 
                                                        nn.BatchNorm1d(skip_z_dims[skip_j]*10), 
                                                        nn.LeakyReLU(self.leaky_relu_coef), \
                                                        nn.Linear(skip_z_dims[skip_j]*10, skip_z_dims[skip_j]*node_nums[skip_j]), \
                                                        nn.BatchNorm1d(skip_z_dims[skip_j]*node_nums[skip_j]), 
                                                        nn.LeakyReLU(self.leaky_relu_coef), \
                                                        ))
                    else:
                        self.skipzs.append(nn.Sequential(nn.Linear(self.in_dim, skip_z_dims[skip_j]*10), 
                                                        nn.LeakyReLU(self.leaky_relu_coef), \
                                                        nn.Linear(skip_z_dims[skip_j]*10, skip_z_dims[skip_j]*node_nums[skip_j]), \
                                                        nn.LeakyReLU(self.leaky_relu_coef), \
                                                        ))
                    pre_dim += skip_z_dims[skip_j]
                self.edge_attr_layers.append(EdgeAttrConstruct(pre_dim, edge_hidden_dim, edge_hidden_dim, useBN=attr_bn))

        self.final_node_layers = []
        for fdim in final_layer_dims:
            self.final_node_layers.append(nn.Linear(pre_dim, fdim))
            if final_bn:
                self.final_node_layers.append(nn.BatchNorm1d(fdim))
            self.final_node_layers.append(nn.LeakyReLU(leaky_relu_coef))
            pre_dim = fdim
        self.final_node_layers.append(nn.Linear(pre_dim, self.node_feature))
        self.final_node_layers = nn.Sequential(*self.final_node_layers)

    def forward(self, z):
        # Given z produce the fake data.
        n = z.size(0)
        xs = self.ln0(z)
        xs = F.leaky_relu(xs, self.leaky_relu_coef)
        xs = xs.view(3*n, self.initial_dim)
        xs = self.bn0(xs)
        xs = F.leaky_relu(xs, self.leaky_relu_coef)
        xs = xs.view(n, 3, self.initial_dim)
        a1 = self.first_edge_link(xs[:, 0, :], xs[:, 1, :])
        a2 = self.first_edge_link(xs[:, 1, :], xs[:, 2, :])
        a3 = self.first_edge_link(xs[:, 0, :], xs[:, 2, :])
        n1 = torch.arange(n).to(self.device)*3
        n2 = torch.arange(n).to(self.device)*3 + 1
        n3 = torch.arange(n).to(self.device)*3 + 2
        edge_prob = torch.zeros(n, 4).to(self.device)
        edge_prob[:, 0] = (a1[:, 0] + a2[:, 1] + a3[:, 1])/3
        edge_prob[:, 1] = (a2[:, 0] + a1[:, 1] + a3[:, 1])/3
        edge_prob[:, 2] = (a3[:, 0] + a2[:, 1] + a1[:, 1])/3
        edge_prob[:, 3] = (a1[:, 1] + a2[:, 1] + a3[:, 1])/3
        edge_prob = F.softmax(edge_prob, dim=1) + 1e-4
        m = Categorical(edge_prob)
        edge0_links = m.sample().to(self.device)
        prob_init_edge = m.log_prob(edge0_links).to(self.device)
        en0 = (edge0_links == 0)
        en1 = (edge0_links == 1)
        en2 = (edge0_links == 2)
        en3 = (edge0_links == 3)
        edge_index = torch.LongTensor(2, 2*(en0.sum() + en1.sum() + en2.sum()) + 3*en3.sum()).to(self.device)
        edge_index[0, :en0.sum()] = n1[en0]
        edge_index[1, :en0.sum()] = n3[en0]
        edge_index[0, en0.sum():en0.sum()*2] = n2[en0]
        edge_index[1, en0.sum():en0.sum()*2] = n3[en0]

        edge_index[0, 2*en0.sum():2*en0.sum()+en1.sum()] = n1[en1]
        edge_index[1, 2*en0.sum():2*en0.sum()+en1.sum()] = n3[en1]
        edge_index[0, 2*en0.sum()+en1.sum():2*en0.sum()+2*en1.sum()] = n1[en1]
        edge_index[1, 2*en0.sum()+en1.sum():2*en0.sum()+2*en1.sum()] = n2[en1]

        edge_index[0, 2*en0.sum()+2*en1.sum():2*en0.sum()+2*en1.sum()+en2.sum()] = n1[en2]
        edge_index[1, 2*en0.sum()+2*en1.sum():2*en0.sum()+2*en1.sum()+en2.sum()] = n2[en2]
        edge_index[0, 2*en0.sum()+2*en1.sum()+en2.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()] = n2[en2]
        edge_index[1, 2*en0.sum()+2*en1.sum()+en2.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()] = n3[en2]

        edge_index[0, 2*en0.sum()+2*en1.sum()+2*en2.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()+en3.sum()] = n1[en3]
        edge_index[1, 2*en0.sum()+2*en1.sum()+2*en2.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()+en3.sum()] = n2[en3]
        edge_index[0, 2*en0.sum()+2*en1.sum()+2*en2.sum()+en3.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()+2*en3.sum()] = n1[en3]
        edge_index[1, 2*en0.sum()+2*en1.sum()+2*en2.sum()+en3.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()+2*en3.sum()] = n3[en3]
        edge_index[0, 2*en0.sum()+2*en1.sum()+2*en2.sum()+2*en3.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()+3*en3.sum()] = n2[en3]
        edge_index[1, 2*en0.sum()+2*en1.sum()+2*en2.sum()+2*en3.sum():2*en0.sum()+2*en1.sum()+2*en2.sum()+3*en3.sum()] = n3[en3]


        xs = xs.view(3*n, self.initial_dim)
        edge_attr = self.edge_attr_layers[0](xs, edge_index)
        edge_attr = F.leaky_relu(edge_attr, self.leaky_relu_coef)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)

        batch = torch.arange(n).view(-1, 1).repeat(1, 3).view(-1)
        edge_batch = torch.zeros(edge_index.size(1)).type(torch.LongTensor)
        edge_batch.scatter_(0, torch.arange(edge_index.size(1)), batch[edge_index[0]])
        
        probs = []
        
        for j in range(len(self.convs)):
            for k in range(len(self.convs[j])):
                xs = self.convs[j][k](xs, edge_index=edge_index, edge_attr=edge_attr)
                if self.use_bn:
                    xs = self.bns[j][k](xs)
                xs = self.acts[j][k](xs)
            if j < len(self.convs) - 1:
                # do unpool.
                # add z skip
                # edge attr
                xs, batch, edge_index, edge_batch, prob_edge = self.unpools[j](xs, edge_index, edge_attr, batch)
                probs.append(prob_edge)
                if self.skip_z:
                    add_xs = self.skipzs[j](z)
                    skip_j = j + 1
                    add_xs = add_xs.view(add_xs.size(0), self.node_nums[skip_j], self.skip_z_dims[skip_j])
                    add_xs = assemble_skip_z(add_xs, batch, self.device)
                    xs = torch.cat([xs, add_xs], axis=1)
                edge_attr = self.edge_attr_layers[j + 1](xs, edge_index)
        xs = self.final_node_layers(xs)

        fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=None, batch=batch, edge_batch=edge_batch)
        fake_data1 = Batch.from_data_list(\
            [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                    for j in fake_data], \
                    follow_batch=['edge_index']).to(self.device)
        return fake_data1, prob_init_edge + sum(probs).to(self.device)

class RGADJ(nn.Module):
    def __init__(self, in_dim, hidden_dims=[64, 128, 32, 32], \
        useBN=False, nodeNum=12, node_outdim=2, edge_outdim=2, bias=True, device='cpu'):
        super(RGADJ, self).__init__()
        self.useBN = useBN
        self.nodeNum = nodeNum
        self.edge_outdim = edge_outdim
        self.node_outdim = node_outdim
        self.hidden_dims = hidden_dims
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.in_dim = in_dim
        pre_dim = in_dim
        for j in hidden_dims:
            self.linears.append(nn.Linear(pre_dim, j, bias=bias))
            pre_dim = j
            if self.useBN:
                self.bns.append(nn.BatchNorm1d(j))

        self.final_nodeout = nn.Linear(pre_dim, nodeNum*node_outdim, bias=bias)
        self.final_edgeout = nn.Linear(pre_dim, nodeNum*nodeNum*edge_outdim, bias=bias)
        self.device = device

    def generate_XA(self, z):
        for j in range(len(self.linears)):
            z = self.linears[j](z)
            if self.useBN:
                z = self.bns[j](z)
            z = F.leaky_relu(z, 0.05)
        out_X = self.final_nodeout(z).view(-1, self.nodeNum, self.node_outdim)
        out_A = F.leaky_relu(self.final_edgeout(z), 0.05).view(-1, self.nodeNum, self.nodeNum, self.edge_outdim)
        out_A = (out_A + torch.transpose(out_A, 1, 2))/2
        return out_X, F.softmax(out_A, dim=-1)

    def forward(self, z, tau=1.0, **kargs):
        out_X, out_A = self.generate_XA(z)
        fake_data = [convertA_to_data(out_X[j], out_A[j], gumbel=True, \
                        device=self.final_edgeout.weight.device, \
                        tau=tau,
                        **kargs) for j in range(len(out_X))]
        fake_data = [Data(x=j[0], edge_index=j[1], edge_attr=j[2]) for j in fake_data]
        fake_data1 = Batch.from_data_list(\
            fake_data, \
                    follow_batch=['edge_index']).to(self.device)
        return fake_data1

    def forward_node_edge(self, z):
        out_X, out_A = self.generate_XA(z)
        return out_X, out_A

class RGDiscrimininator(nn.Module):
    """
    Output hidden_dim*2 (one sum pool and one max pool) + edge_dim (mean pool from mean)
    """
    def __init__(self, in_dim, hidden_dim, lin_hidden_dim=None, direct_link_dims=None, out_hidden_dim=[128, 258], device='cpu', \
                useBN=False, droprate=None, outBN=False, out_drop=0.3, \
                final_layers=2, \
                conv_layers=2,
                last_act='sigmoid', relu_coef=0.05, outdim=1):
        
        # direct_link_dims should be a tuple of (list dims, lin_out_dim, list dims), i.e. ([128, 128], 128, [128, 256]) for hidden dimensions of conv/ global graph output dim/ hidden dimensions of out layers.
        super(RGDiscrimininator, self).__init__()
        self.relu_coef = relu_coef
        self.final_layers = final_layers
        if lin_hidden_dim is None:
            lin_hidden_dim = hidden_dim
        self.lin_hidden_dim = lin_hidden_dim
        self.direct_link = direct_link_dims

        self.convs = nn.ModuleList([])
        pre_dim = in_dim
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]*conv_layers
        for j in range(conv_layers):
            self.convs.append(GraphConv(pre_dim, hidden_dim[j]))
            pre_dim = hidden_dim[j]
            if useBN:
                self.convs.append(nn.BatchNorm1d(hidden_dim[j]))
            self.convs.append(nn.LeakyReLU(relu_coef))
            if droprate:
                self.convs.append(nn.Dropout(droprate))
        # self.convs = nn.Sequential(*self.convs)


        in_dim_hidden = hidden_dim[-1] + in_dim
        self.sig_ln = nn.Linear(in_dim_hidden, lin_hidden_dim)
        self.tanh_ln = nn.Linear(in_dim_hidden, lin_hidden_dim)

        self.droprate = droprate
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.useBN = useBN
        self.out_dim = self.lin_hidden_dim
        
        self.final_lns = []
        pre_dim = self.out_dim
        if isinstance(out_hidden_dim, int):
            out_hidden_dim = [out_hidden_dim] * self.final_layers
        
        for j in range(self.final_layers):
            if out_drop:
                self.final_lns.append(nn.Dropout(out_drop))
            self.final_lns.append(nn.Linear(pre_dim, out_hidden_dim[j]))
            pre_dim = out_hidden_dim[j]
            if outBN:
                self.final_lns.append(nn.BatchNorm1d(out_hidden_dim[j]))
            self.final_lns.append(nn.LeakyReLU(relu_coef))
        # self.final_lns.append(nn.Linear(pre_dim, 1))
        self.final_lns = nn.Sequential(*self.final_lns)
        if direct_link_dims is not None:
            self.direct_convs = nn.ModuleList([])
            self.direct_final_lns = []
            conv_layers, lin_graph_out_dim, final_layers = direct_link_dims
            pre_dim_d = in_dim
            for j in conv_layers:
                self.direct_convs.append(GraphConv(pre_dim_d, j))
                self.direct_convs.append(nn.LeakyReLU(relu_coef))
                pre_dim_d = j
            self.direct_sig_ln = nn.Linear(pre_dim_d, lin_graph_out_dim)
            self.direct_tanh_ln = nn.Linear(pre_dim_d, lin_graph_out_dim)
            pre_dim_d = lin_graph_out_dim
            for j in final_layers:
                self.direct_final_lns.append(nn.Linear(pre_dim_d, j))
                self.direct_final_lns.append(nn.LeakyReLU(relu_coef))
                pre_dim_d = j
            self.direct_final_lns = nn.Sequential(*self.direct_final_lns)
            
        
        if direct_link_dims is None:
            self.final_layer = nn.Linear(pre_dim, outdim)
        else:
            self.final_layer = nn.Linear((pre_dim + pre_dim_d), outdim)
        self.last_act = last_act

    def forward(self, data=None, x=None, edge_index=None, batch=None, edge_batch=None, edge_attr=None):
        # GCNs:
        if not hasattr(self, 'relu_coef'):
            self.relu_coef = 0.05
        x0 = x
        if data is not None:
            x0 = data.x
            edge_index = data.edge_index
            # edge_attr = data.edge_attr
            batch = data.batch
            if hasattr(data, 'edge_index_batch'):
                edge_batch = data.edge_index_batch
            else:
                edge_batch = data.edge_batch
            if hasattr(data, 'edge_attr'):
                edge_attr = data.edge_attr

        x = x0
        for op in self.convs:
            if isinstance(op, GCNConv) or isinstance(op, GraphConv):
                x = op(x, edge_index=edge_index, edge_weight=edge_attr)
            else:
                x = op(x)
        x = torch.cat([x, x0], axis=1)
        sx = torch.clip(self.sig_ln(x), -30, 30)
        x = 1/(1+torch.exp(sx))*torch.tanh(self.tanh_ln(x))

        out_x = global_add_pool(x, batch=batch)
        
        final_x = self.final_lns(out_x)
        if hasattr(self, 'direct_link') and self.direct_link is not None:
            # Add the non-BN's features.
            xd = x0
            for op in self.direct_convs:
                if isinstance(op, GCNConv) or isinstance(op, GraphConv):
                    xd = op(xd, edge_index=edge_index)
                else:
                    xd = op(xd)
            sxd = torch.clip(self.direct_sig_ln(xd), -30, 30)
            xd = 1/(1+torch.exp(sxd))*torch.tanh(self.direct_tanh_ln(xd))
            out_xd = global_add_pool(xd, batch=batch)
            final_xd = self.direct_final_lns(out_xd)
            final_x = torch.cat([final_x, final_xd], axis=1)
            
        final_x = self.final_layer(final_x)
        if self.last_act == 'sigmoid':
            out = F.sigmoid(final_x)
        elif self.last_act == 'tanh':
            out = F.tanh(final_x)
        elif self.last_act == 'leaky_relu':
            out = F.leaky_relu(final_x, self.relu_coef)
        elif self.last_act == 'relu':
            out = F.relu(final_x)
        else:
            out = final_x
        return out


class GANTrainerComb(object):
    def __init__(self, d, g, rand_dim, train_folder, tot_epoch_num=300, eval_iter_num=100, batch_size=64, \
        device=None, d_add=None, learning_rate_g=1e-3, learning_rate_d=1e-4, \
        lambda_g=0, max_train_G=2, tresh_add_trainG=0.2, \
        use_loss='wgan', \
        g_out_prob=False, lambda_rl=1.0, \
        lambda_nonodes = 0.,
        lambda_noedges = 0.,
        trainD=True, \
        reward_point=None, 
        initial_weight=True
        ):
        '''
        use_loss can be "wgan", or "bce", or "reward"
        reward_point is the integer where we need to make for the reward.
        #   1 for QED
        #   2 for logP
        #   3 for SAscore
        # TODO: add this additional loss type
        '''
        self.batch_size = batch_size
        self.reward_point = reward_point
        if device is None:
            device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
        if trainD:
            gcn_model = d.to(device)
        else:
            gcn_model = d
        generator = g.to(device)
        lr_d = learning_rate_d
        lr_g = learning_rate_g
        self.lambda_node = lambda_nonodes
        self.lambda_edge = lambda_noedges
        beta_d = (0.5, 0.999)
        beta_g = (0.5, 0.999)
        if trainD:
            optimD = optim.Adam([{'params': gcn_model.parameters()},
                                ], \
                                lr=lr_d, betas=beta_d)
            self.optimD = optimD
            if initial_weight:
                gcn_model.apply(weight_initiate)

        optimG = optim.Adam([{'params': generator.parameters()}, \
        #                      {'params': q_output.parameters()}\
                            ], lr=lr_g, betas=beta_g)
        if initial_weight:
            generator.apply(weight_initiate)
        z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=rand_dim, device=device, batch=self.batch_size)
        self.fix_noise = z_rand
        self.rand_dim = rand_dim

        self.gcn_model = gcn_model
        self.generator = generator
        self.lambda_g = lambda_g
        self.lambda_rl = lambda_rl

        self.folder = train_folder
        self.tot_epoch_num = tot_epoch_num
        self.eval_iter_num = eval_iter_num
        self.batch_size = batch_size
        self.max_train_G = max_train_G
        self.tresh_add_trainG = tresh_add_trainG
        self.use_loss = use_loss
        self.evals = []
        self.error_rates = []
        self.optimG = optimG
        self.g_out_prob = g_out_prob
        self.criterionD = nn.BCELoss(reduction='none')
        self.device = device



    def call_g(self, data=None, usex=None, batch=None, edge_index=None, edge_batch=None):
        if data is not None:
            usex = data.x
            batch = data.batch
            edge_index = data.edge_index
            edge_batch = data.edge_index_batch

        p1 = self.gcn_model(x=usex, \
                            edge_index=edge_index, \
                            batch=batch, edge_batch=edge_batch)
        probs_real = p1
        return probs_real

    def train(self, data_loader, verbose=False, \
        evaluate_num=1000, mol_data=None, saveZ=False, trainD=True, \
            alter_trainer=False, NN=1000, \
                reinforce_acclerate=False, rl_only=False,
                    acclerate_noscale=True, only_train=None, pre_train_D_steps=None):
        # Save all the z's.
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        ii = 0
        import time
        real_label = 1.0
        fake_label = 0.0
        self.temp_loss_D = []
        self.temp_loss_G = []
        add_train_g = 1
        kk = 0
        self.all_zs = []
        label = torch.full((self.batch_size*2, ), real_label, device=self.device)
        label[self.batch_size:] = fake_label
        if hasattr(self, 'eval_counts') and self.eval_counts:
            count_eval = True
            real_counts = []
            real_rewards = []
            fake_counts = []
            fake_rewards = []
        else:
            count_eval = False
        if pre_train_D_steps is not None:
            only_train = 'D'
        for epoch in range(self.tot_epoch_num):
            for i, data in enumerate(data_loader):
                if pre_train_D_steps is not None and ii > pre_train_D_steps:
                    if only_train is not None:
                        print ("RESET ONLY TRAIN")
                        only_train = None
                if trainD and (only_train is None or only_train == 'D'):
                    if max(data.batch) < self.batch_size - 1:
                        print (f'less than {self.batch_size}')
                        continue
                    data = data.to(self.device)
                    self.optimD.zero_grad()
                    # Real data
                    if verbose:
                        print (1)
                    label[:self.batch_size] = real_label
                    label[self.batch_size:] = fake_label
                    edge_attr = None
                    if hasattr(data, 'edge_attr'):
                        edge_attr = data.edge_attr
                    real_data = convert_Batch_to_datalist(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr, batch=data.batch, \
                                edge_batch=data.edge_index_batch)
                    z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=self.rand_dim, device=self.device, batch=self.batch_size)
                    z = z_rand
                    if saveZ:
                        self.all_zs.append(('d', z, data))
                    if self.g_out_prob:
                        fake_data, prob = self.generator(z)
                    else:
                        fake_data = self.generator(z, obtain_connected=True)

                    edge_attr = None
                    if hasattr(fake_data, 'edge_attr'):
                        edge_attr = fake_data.edge_attr
                    fake_data_list = convert_Batch_to_datalist(x=fake_data.x, edge_index=fake_data.edge_index, edge_attr=edge_attr, batch=fake_data.batch, \
                                edge_batch=fake_data.edge_index_batch)
                    
                    all_data = Batch.from_data_list(\
                        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                                for j in real_data] + \
                        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                                for j in fake_data_list], \
                                follow_batch=['edge_index']).to(self.device)
                    
                    
                    probs = self.call_g(all_data)
                    probs_real = probs[:self.batch_size]
                    probs_fake = probs[self.batch_size:]
                    probs_fake_backup = probs_fake

                    if self.use_loss == 'wgan':
                        loss = (probs.view(-1)*(0.5-label)*2).mean()
                        loss_real = (probs.view(-1)*(0.5-label)*2)[:self.batch_size].mean()
                        loss_fake = (probs.view(-1)*(0.5-label)*2)[self.batch_size:].mean()
                    else:
                        loss = self.criterionD(probs.view(-1), label)
                        loss_real = loss[:self.batch_size].mean()
                        loss_fake = loss[self.batch_size:].mean()
                        loss = loss.mean()
                    prob_loss_real = probs.view(-1)[:self.batch_size].mean()
                    prob_loss_fake = probs.view(-1)[self.batch_size:].mean()
                    loss.backward()
                    if self.lambda_g > 0:
                        raise ValueError("WE DON'T SUPPORT GRADIENT PENALTY FOR NUMERIC FEATURED GRAPH GENERATION.")
                    else:
                        gp = torch.zeros(1).mean()
                        D_loss = loss
                    if prob_loss_real.item() > prob_loss_fake.item() + self.tresh_add_trainG:
                        add_train_g = min(1+add_train_g, self.max_train_G)
                    else:
                        add_train_g = 1
                    self.optimD.step()
                else:
                    D_loss = torch.FloatTensor([0])
                    gp = torch.FloatTensor([0])
                    probs_real = torch.FloatTensor([0])
                    if i % 1000 == 0:
                        print ('Not train D...')

                if alter_trainer:
                    use_add_train_g = add_train_g*2
                else:
                    use_add_train_g = add_train_g
                if only_train is None or only_train == 'G':
                    pass
                else:
                    use_add_train_g = 0
                    G_loss = torch.FloatTensor([0])
                    gen_loss = torch.FloatTensor([0])
                    if i % 1000 == 0:
                        print ("Not train G... as only_train = ", only_train, '; ii = ', ii)
                for l in range(use_add_train_g):
                    self.optimG.zero_grad()


                    real_data = convert_Batch_to_datalist(x=data.x, edge_index=data.edge_index, edge_attr=None, batch=data.batch, \
                                edge_batch=data.edge_index_batch)
                    z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=self.rand_dim, device=self.device, batch=self.batch_size)
                    z = z_rand
                    if saveZ:
                        self.all_zs.append(('d', z, data))
                    if self.g_out_prob:
                        fake_data, prob = self.generator(z)
                    else:
                        fake_data = self.generator(z, obtain_connected=True)

                    fake_data_list = convert_Batch_to_datalist(x=fake_data.x, edge_index=fake_data.edge_index, edge_attr=None, batch=fake_data.batch, \
                                edge_batch=fake_data.edge_index_batch)
                    
                    all_data = Batch.from_data_list(\
                        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                                for j in real_data] + \
                        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                                for j in fake_data_list], \
                                follow_batch=['edge_index']).to(self.device)

                        
                    probs = self.call_g(all_data)
                    probs_real = probs[:self.batch_size]
                    probs_fake_backup = probs[self.batch_size:]
                    probs_fake = probs[self.batch_size:]

                    label.fill_(real_label)
                    if self.use_loss == 'wgan':
                        gen_loss = -(probs_fake.mean())
                    else:
                        gen_loss = self.criterionD(probs_fake.view(-1), label[self.batch_size:]).mean()
                        probs_fake = -self.criterionD(probs_fake.view(-1), label[self.batch_size:])
                    if self.lambda_edge > 0 or self.lambda_node > 0:
                        useX, useA = self.generator.generate_XA(z)
                        gen_loss_nonodes = useX[:, :, -1].sum(axis=1).mean()*self.lambda_node
                        gen_loss_edges = F.relu((1 - useX[:, :, -1]).sum(axis=1) - ((1 - useA[:, :, :, -1]).sum(axis=1).sum(axis=1) - 9)/2).mean()*self.lambda_edge
                        gen_loss += gen_loss_nonodes + gen_loss_edges
                    if self.g_out_prob:
                        if alter_trainer and (l % 2 == 0) and (not rl_only):
                            G_loss = gen_loss
                        elif (alter_trainer and (l % 2 == 1)) or rl_only:
                            if reinforce_acclerate:
                                if acclerate_noscale:
                                    G_loss = - (prob * ((probs_fake - probs_fake.mean())).detach().view(-1)).mean() * self.lambda_rl
                                else:
                                    G_loss = - (prob * ((probs_fake - probs_fake.mean())/(probs_fake.std() + 1e-3)).detach().view(-1)).mean() * self.lambda_rl
                            else:
                                G_loss = - (prob * probs_fake.detach().view(-1)).mean() * self.lambda_rl
                        else:
                            if reinforce_acclerate:
                                if acclerate_noscale:
                                    G_loss = gen_loss - (prob * ((probs_fake - probs_fake.mean())).detach().view(-1)).mean() * self.lambda_rl
                                else:
                                    G_loss = gen_loss - (prob * ((probs_fake - probs_fake.mean())/(probs_fake.std() + 1e-3)).detach().view(-1)).mean() * self.lambda_rl
                            else:
                                G_loss = gen_loss \
                                        - (prob * probs_fake.detach().view(-1)).mean() * self.lambda_rl
                    else:
                        G_loss = gen_loss
                    G_loss.backward()
                    self.optimG.step()
                if trainD:
                    self.temp_loss_D.append(D_loss.item())
                else:
                    D_loss = torch.sum(torch.FloatTensor([0, 0]))
                    gp = torch.sum(torch.FloatTensor([0, 0]))
                    probs_real = torch.FloatTensor([0, 0])
                self.temp_loss_G.append(G_loss.item())
                if ii % 100 == 0:
                    print(('[%d/%d][%d/%d]\t' 
                        'G Loss: %.4f;'
                        'D Loss: %.4f; GP: %.4f')
                        % (epoch+1, 200, i, len(data_loader), 
                            gen_loss.item(), \
                            D_loss.item() - gp.item(), gp.item()))
                    print ('now, we train G %d times with (prob fake = %.3f, prob real = %.3f)' % (add_train_g, \
                                                                                                    probs_fake_backup.mean(), \
                                                                                                    probs_real.mean()))
                    print ('Mean x/edge attr: ', fake_data.x.mean(axis=0), fake_data.edge_index.size(1)/2/self.batch_size)
                    if self.g_out_prob:
                        print ("Sample prob:", prob.mean().item())
                if ii % NN == 0:
                    eval_fake_data, eval_res = self.eval(evaluate_num, mol_data)
                    self.evals.append(eval_res)
                    torch.save(self.generator, os.path.join(self.folder, f'generator_{kk}.pt'))
                    torch.save(self.gcn_model, os.path.join(self.folder, f'gcn_model_{kk}.pt'))
                    # if saveZ:
                    #     torch.save(self.all_zs, os.path.join(self.folder, f'train_z{kk}.pt'))
                    self.all_zs = []
                    kk += 1
                ii += 1

    def eval(self, eval_num, mol_data):
        self.generator.eval()
        if hasattr(self, 'eval_counts') and self.eval_counts:
            self.gcn_model.eval()
            count_eval = True
        else:
            count_eval = False
        with torch.no_grad():
            eval_fake_data = []
            problem_info = []
            rewards = []
            counts = []
            for j in range(eval_num//self.batch_size + 1):
                z_rand, z_lr_cont, z_cate, z_cont = generate_noise(device=self.device, rand_dim=self.rand_dim, batch=self.batch_size)
                z = z_rand
                if self.g_out_prob:
                    fake_data, probs = self.generator(z)
                else:
                    fake_data = self.generator(z, obtain_connected=True)
                # if count_eval:
                #     rewards.extend(list(np.array(self.gcn_model(fake_data).detach().cpu())))
                #     counts.extend([fake_data[j].x.size(0) for j in range(fake_data.num_graphs)])

                fake_data = convert_Batch_to_datalist(x=fake_data.x, 
                                                    edge_index=fake_data.edge_index, \
                                                    batch=fake_data.batch, \
                                                    edge_batch=fake_data.edge_index_batch)
                if np.sum([j.x.isnan().sum().item() for j in fake_data]) > 0:
                    print ("Problem here")
                eval_fake_data.extend(fake_data)
            node_cnt = [j.x.size(0) for j in eval_fake_data]
            edge_cnt = [j.edge_index.size(1)//2 for j in eval_fake_data]
            node_feature = torch.cat([j.x for j in eval_fake_data], axis=0)
            eval_res = dict(mean_node_cnt = np.mean(node_cnt), 
                            std_node_cnt = np.std(node_cnt), 
                            mean_edge_cnt = np.mean(edge_cnt), 
                            std_edge_cnt = np.std(edge_cnt), 
                            mean_node_feature = node_feature.mean(), 
                            std_node_feature = node_feature.std())
            print ("Validation, uniqueness, novelty: ", eval_res) 
            print (mol_data.x.mean(), mol_data.x.std())
        self.generator.train()
        if hasattr(self, 'eval_counts') and self.eval_counts:
            self.gcn_model.train()
        return eval_fake_data, eval_res


class RGVAE(nn.Module):
    def __init__(self, decoder, encoder, enc_outdim, z_dim, 
                lr=1e-4, beta=(0.5, 0.999), g_prob=True, 
                permutation=False, lambda_rl=1e-1, max_num_nodes=12, folder='vae_rg', device='cpu', \
                beta_node=1, beta_edge=1, beta_edge_total=1, beta_node_degree=1, beta_node_feature=1, batch_size=128):
        # g_prob: T if the decoder produce a probability, F if not.
        super(RGVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_mu = nn.Linear(enc_outdim, z_dim)
        self.z_sigma = nn.Linear(enc_outdim, z_dim)
        self.g_prob = g_prob
        self.optim = optim.Adam([{'params': self.parameters()},], \
                            lr=lr, betas=beta)
        self.permutation = permutation
        self.max_num_nodes = max_num_nodes
        self.folder = folder
        if not os.path.exists(folder):
            os.mkdir(self.folder)
        self.device = device
        self.lambda_rl = lambda_rl
        self.beta_node = beta_node
        self.beta_edge = beta_edge
        self.beta_edge_total = beta_edge_total
        self.beta_node_degree = beta_node_degree
        self.beta_node_feature = beta_node_feature
        self.batch_size = batch_size

    def forward(self, data, beta=1e-1, verbose=False, save_eval=False):
        
        usex = data.x
        batch = data.batch
        edge_index = data.edge_index
        edge_batch = data.edge_index_batch

        h = self.encoder(x=usex, \
                            edge_index=edge_index, \
                            batch=batch, edge_batch=edge_batch)
        
        # Obtain h from encoder
        z_mu = self.z_mu(h)
        z_lsgms = self.z_sigma(h)
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = torch.randn(z_sgm.size()).to(self.device)
        z = eps*z_sgm + z_mu
        if z.size(0) < self.batch_size:
            # Replicate z to 128 times.
            repeat_cnt = self.batch_size//z.size(0)
            z = z.repeat((repeat_cnt, 1))
        # use decoder on z
        
        if self.g_prob:
            produced_data, probs = self.decoder(z)
        else:
            produced_data = self.decoder(z)
            probs = None
        
        # Reconstruction error
        # Permutation or not.
        if self.permutation:
            pass
        fake_x, fake_mask = to_dense_batch(produced_data.x, produced_data.batch, max_num_nodes=self.max_num_nodes)
        real_x, real_mask = to_dense_batch(data.x, data.batch, max_num_nodes=self.max_num_nodes)
        if real_x.size(0) < self.batch_size:
            repeat_cnt = self.batch_size//real_x.size(0)
            real_mask = real_mask.repeat((repeat_cnt, 1))
            real_x = real_x.repeat((repeat_cnt, 1, 1))
            
        # Node existence error
        node_loss = torch.abs(fake_mask.type(torch.FloatTensor) - real_mask.type(torch.FloatTensor)).sum(axis=1)
        # Node feature error
        node_f_loss = (torch.pow(fake_x - real_x, 2).sum(axis=2)*fake_mask*real_mask).sum(axis=1)
        # Edge existence error
        fake_adj = to_dense_adj(produced_data.edge_index, produced_data.batch, max_num_nodes=self.max_num_nodes)
        real_adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.max_num_nodes)
        if real_adj.size(0) < self.batch_size:
            repeat_cnt = self.batch_size//real_adj.size(0)
            real_adj = real_adj.repeat((repeat_cnt, 1, 1))
        edge_loss = (torch.abs(fake_adj - real_adj).sum(axis=2)*real_mask).sum(axis=1)/real_mask.sum(axis=1)
        
        edge_total_loss = torch.abs(fake_adj.sum(axis=(1, 2)) - real_adj.sum(axis=(1, 2)))
        node_degree_loss = (torch.abs(fake_adj.sum(axis=(2)) - real_adj.sum(axis=(2)))*fake_mask).sum(axis=1)/(fake_mask.sum(axis=1)+1e-6)
        # This is the KL loss for VAE.
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss = beta*loss_kl.to(self.device) + edge_loss.to(self.device)*self.beta_edge + \
                    node_f_loss.to(self.device)*self.beta_node_feature + \
                    node_loss.to(self.device)*self.beta_node + \
                    node_degree_loss.to(self.device)*self.beta_node_degree + \
                    edge_total_loss.to(self.device)*self.beta_edge_total
        if verbose:
            print (node_loss.shape, edge_loss.shape, node_f_loss.shape)
            print (f"Node loss: {node_loss.mean()}; node feature loss: {node_f_loss.mean()}; Edge loss: {edge_loss.mean()}; Edge total loss: {edge_total_loss.mean()}; Node degree erro: {node_degree_loss.mean()}")
            print ("node loss: ", node_f_loss.mean(), '; edge_loss:', edge_loss.min(), ' to ', edge_loss.max(), ' in ', edge_loss.shape)
        if save_eval:
            self.evals.append(dict(edge_loss=edge_loss.mean().item(), node_loss=node_loss.mean().item(), node_f_loss=node_f_loss.mean().item(), edge_total_loss=edge_total_loss.mean().item(),))
        return loss, probs
    
    def train(self, dataloader, epoch=5, beta=1e-2, verbose_step=10000, save_step=1000):
        ii = 0
        k = 1
        self.evals = []
        for i in epoch:
            for data in dataloader:
                data = data.to(self.device)
                self.optim.zero_grad()
                loss, probs = self(data, beta=beta, verbose=(ii % verbose_step == 0), save_eval=(save_step and ii % save_step == 0))
                if probs is None:
                    loss.mean().backward()
                    self.optim.step()
                else:
                    (loss.mean() + self.lambda_rl*((loss - loss.mean()).detach() * probs).mean()).backward()
                    # Change inside of this to `loss - loss.mean()`
                    self.optim.step()
                if save_step and ii % save_step == 0:
                    torch.save(self.decoder, os.path.join(self.folder, f'decoder_{k}.pt'))
                    torch.save(self, os.path.join(self.folder, f'vae_{k}.pt'))
                    k += 1
                ii += 1
