
import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import long, optim
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing, GINConv, unpool
from .unpool_layers_simple_v2 import LinkLayer, UnpoolLayer, EdgeAttrConstruct, UnpoolLayerE, UnpoolLayerEZ
from torch.distributions import Categorical
from torch_geometric.utils import add_self_loops, degree, to_undirected
from .unpool_utils import assemble_skip_z, convert_Batch_to_datalist 


class encoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dims=[32, 64], out_dim=128, use_bn=True, hidden_act='relu', \
        leaky_relu_coef=0.02):
        super(encoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dims = hidden_dims
        self.bns = []
        self.convs = []
        self.acts = []
        self.leaky_relu_coef = leaky_relu_coef
        pre_dim = in_dim
        for hd in hidden_dims:
            self.convs.append(NNConv(pre_dim, hd, nn=nn.Linear(edge_dim, pre_dim*hd)))
            self.bns.append(nn.BatchNorm1d(hd))
            if hidden_act == 'relu':
                self.acts.append(nn.ReLU())
            elif hidden_act == 'sigmoid':
                self.acts.append(nn.Sigmoid())
            else:
                self.acts.append(nn.LeakyReLU(self.leaky_relu_coef))
            pre_dim = hd
        self.bns = nn.ModuleList(self.bns)
        self.convs = nn.ModuleList(self.convs)
        self.acts = nn.ModuleList(self.acts) 
        self.s_lin = nn.Linear(hd, out_dim)
        self.t_lin = nn.Linear(hd, out_dim)
        
        
        
        
    def forward(self, x, edge_index, edge_attr, batch, edge_batch):
        for i in range(len(self.acts)):
            conv = self.convs[i]
            bn = self.bns[i]
            act = self.acts[i]
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = act(x)
            
        s_x = self.s_lin(x)
        t_x = self.t_lin(x)
        s_x = torch.clip(s_x, -30, 30)
        f_x = 1/(1+torch.exp(s_x)) * torch.tanh(t_x)
        feature = global_add_pool(f_x, batch=batch)
        return feature


class decoder(nn.Module):
    def __init__(self, in_dim, z_dim, max_node_num, node_feature, edge_feature, hidden_dims=[128, 256, 512], hidden_act='relu', \
        leaky_relu_coef=0.05, device='cpu'):
        super(decoder, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.encode_11 = nn.Linear(in_dim, z_dim)
        self.encode_12 = nn.Linear(in_dim, z_dim)
        self.lins = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.acts = nn.ModuleList([])
        self.hidden_dims = hidden_dims
        self.leaky_relu_coef = leaky_relu_coef
        pre_dim = z_dim
        for hd in hidden_dims:
            self.lins.append(nn.Linear(pre_dim, hd))
            self.bns.append(nn.BatchNorm1d(hd))
            if hidden_act == 'relu':
                self.acts.append(nn.ReLU())
            elif hidden_act == 'sigmoid':
                self.acts.append(nn.Sigmoid())
            else:
                self.acts.append(nn.LeakyReLU(self.leaky_relu_coef))
            pre_dim = hd
        self.edge_out_dim = max_node_num*(max_node_num+1)//2
        self.edge_feature = edge_feature
        self.node_feature = node_feature
        self.max_node_num = max_node_num
        self.map_A = nn.Linear(hd, max_node_num*(max_node_num+1)//2)
        self.map_X = nn.Linear(hd, max_node_num*node_feature)
        self.map_E = nn.Linear(hd, max_node_num*(max_node_num+1)//2 * edge_feature)
    
    def forward(self, h):
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        # eps = Variable(torch.randn(z_sgm.size())).cuda()
        eps = torch.randn(z_sgm.size()).to(self.device)
        z = eps*z_sgm + z_mu
        
        for i in range(len(self.lins)):
            z = self.lins[i](z)
            z = self.bns[i](z)
            z = self.acts[i](z)
        x = self.map_X(z)
        # This is for QM9:
        x = x.view(-1, self.max_node_num, self.node_feature)
        # out1 = torch.softmax(x[:, :, :4], dim=-1)
        # out2 = torch.softmax(x[:, :, 4:7], dim=-1)
        # out3 = torch.softmax(x[:, :, 7:10], dim=-1)
        # x = torch.cat([out1, out2, out3], axis=2)
        A = self.map_A(z)
        E = self.map_E(z)
        E = E.view(-1, self.edge_out_dim, self.edge_feature)
        return x, A, E, z_mu, z_lsgms



class decoderUL(nn.Module):
    def __init__(self, in_dim, z_dim, initial_dim, max_node_num, node_feature, edge_feature, \
            hidden_dims=[[128], [256], [512]], hidden_act='relu', \
            edge_hidden_dim=16,
            leaky_relu_coef=0.05, device='cpu', \
            skip_z=True, skip_z_dims=None, \
            unpool_bn=True, link_bn=True, \
            link_act='leaky', unpool_para={}, \
            attr_bn=True, edge_out_dim=None, \
            fix_points=[1, 1], roll_ones=[0, 0], node_nums = [3, 6, 9]
            ):
        # 3->5->6~9
        super(decoderUL, self).__init__()
        if edge_out_dim is None:
            edge_out_dim = 2*edge_feature
        self.skip_z = skip_z
        if skip_z and skip_z_dims is None:
            skip_z_dims = [d[-1]//4 for d in hidden_dims] + [hidden_dims[-1][-1]//4]

        self.skip_z_dims = skip_z_dims

        self.leaky_relu_coef = leaky_relu_coef
        self.hidden_dims = hidden_dims
        self.leaky_relu_coef = leaky_relu_coef
        self.device = device
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.encode_11 = nn.Linear(in_dim, z_dim)
        self.encode_12 = nn.Linear(in_dim, z_dim)
        self.convs = nn.ModuleList([])
        self.unpools = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.acts = nn.ModuleList([])
        self.edge_attr_layers = nn.ModuleList([])
        self.skipzs = nn.ModuleList([])

        self.ln0 = nn.Linear(z_dim, 3*initial_dim)
        self.first_edge_link = LinkLayer(initial_dim, useBN=link_bn, final_act=link_act)
        self.edge_attr_layers.append(EdgeAttrConstruct(initial_dim, edge_hidden_dim, edge_hidden_dim, useBN=attr_bn))
        self.initial_dim = initial_dim
        self.bn0 = nn.BatchNorm1d(initial_dim)
        pre_dim = initial_dim
        self.skip_z = skip_z
        
        self.node_nums = node_nums
        for j, hds in enumerate(hidden_dims):
            use_convs = nn.ModuleList([])
            use_bns = nn.ModuleList([])
            use_acts = nn.ModuleList([])
            for hd in hds:
                edge_link = nn.Sequential(nn.Linear(edge_hidden_dim, edge_hidden_dim), \
                                          nn.BatchNorm1d(edge_hidden_dim), \
                                          nn.LeakyReLU(self.leaky_relu_coef), \
                                            nn.Linear(edge_hidden_dim, hd*pre_dim))
                use_convs.append(NNConv(pre_dim, hd, nn=edge_link))
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
                    self.skipzs.append(nn.Sequential(nn.Linear(self.z_dim, skip_z_dims[skip_j]*10), 
                                                    nn.BatchNorm1d(skip_z_dims[skip_j]*10), 
                                                    nn.LeakyReLU(self.leaky_relu_coef), \
                                                    nn.Linear(skip_z_dims[skip_j]*10, skip_z_dims[skip_j]*node_nums[skip_j]), \
                                                    nn.BatchNorm1d(skip_z_dims[skip_j]*node_nums[skip_j]), 
                                                    nn.LeakyReLU(self.leaky_relu_coef), \
                                                    ))
                    pre_dim += skip_z_dims[skip_j]
                self.edge_attr_layers.append(EdgeAttrConstruct(pre_dim, edge_hidden_dim, edge_hidden_dim, useBN=attr_bn))
        
        self.final_edge_lin = nn.Sequential(nn.Linear(pre_dim, edge_hidden_dim*2), \
                                    nn.BatchNorm1d(edge_hidden_dim*2), \
                                    nn.LeakyReLU(self.leaky_relu_coef), \
                                    nn.Linear(edge_hidden_dim*2, edge_hidden_dim), \
                                    nn.BatchNorm1d(edge_hidden_dim), \
                                    nn.LeakyReLU(self.leaky_relu_coef))
        self.final_edge_attr = EdgeAttrConstruct(pre_dim + edge_hidden_dim, edge_out_dim, edge_feature, useBN=attr_bn)
        self.final_node_layer = nn.Linear(pre_dim, node_feature)

    def forward(self, h):
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        # eps = Variable(torch.randn(z_sgm.size())).cuda()
        eps = torch.randn(z_sgm.size()).to(self.device)
        z = eps*z_sgm + z_mu
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
        # 0: no 0,1 link
        # 1: no 1,2 link
        # 2: no 0,2 link
        # 3: all links
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
        
        # TODO: finish the forward using UL.
        for j in range(len(self.convs)):
            for k in range(len(self.convs[j])):
                xs = self.convs[j][k](xs, edge_index=edge_index, edge_attr=edge_attr)
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
        edge_attr = self.final_edge_attr(torch.cat([self.final_edge_lin(xs), xs], axis=1), edge_index)
        xs = self.final_node_layer(xs)
        return xs, edge_attr, edge_index, batch, edge_batch, prob_init_edge + sum(probs).to(self.device), z_mu, z_lsgms






class GraphVAE(nn.Module):
    def __init__(self, encoder, decoder, max_num_nodes, pool='sum', device='cpu', decode_type='adj', \
            adj_loss='bce', nodedim1=4, nodedim2=7, nodedim3=10, max_iters=75):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(GraphVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.nodedim1 = nodedim1
        self.nodedim2 = nodedim2
        self.nodedim3 = nodedim3
        self.max_iter = max_iters

        self.adj_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.max_num_nodes = max_num_nodes

        self.decode_type = decode_type
        # Add this initiatization?
        # for m in self.modules():
        #     if isinstance(m, model.GraphConv):
        #         m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        self.pool = pool
        self.adj_loss = adj_loss
        

    def recover_adj_lower(self, l):
        # l comes as N * adj_dim
        # NOTE: Not assume 1 per minibatch
        adj = torch.zeros(l.size(0), self.max_num_nodes, self.max_num_nodes)
        for j in range(l.size(0)):
            adj[j][torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l[j]
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = lower * torch.eye(self.max_num_nodes).view(1, self.max_num_nodes, self.max_num_nodes)
        return lower + torch.transpose(lower, 1, 2) - diag

    def recover_adj_lower_feature(self, l):
        # l comes as N * adj_dim * feature_dim
        # NOTE: Not assume 1 per minibatch
        adj = torch.zeros(l.size(0), self.max_num_nodes, self.max_num_nodes, l.size(2))
        for j in range(l.size(0)):
            adj[j][torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1, :] = l[j, :, :]
        return adj

    def recover_full_adj_from_lower_feature(self, lower):
        diag = lower * torch.eye(self.max_num_nodes).view(1, self.max_num_nodes, self.max_num_nodes, 1)
        return lower + torch.transpose(lower, 1, 2) - diag


    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func, \
                matching_edge_features, matching_edge_features_recon, edge_sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i], matching_features_recon[a])
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b] * \
                                            edge_sim_func(matching_edge_features[i, j], matching_edge_features_recon[a, b])
        return S

    def mpm(self, x_init, S, max_iters=None):
        if max_iters is None:
            if hasattr(self, 'max_iters'):
                max_iters = self.max_iters
            else:
                max_iters = 75
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x 

    def feature_similarity(self, f1, f2):
        return ((f1*f2).sum())

    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def node_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1).mean()


    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros(*adj.shape)
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def permute_feature(self, feature, curr_ind, target_ind):
        ''' Permute features.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        feature_permuted = torch.zeros((self.max_num_nodes, feature.size(1)))
        feature_permuted[:, :] = feature[ind, :]
        return feature_permuted


    def generate_adj(self, x, edge_index, edge_attr, batch, edge_batch, num_index=None):
        num_batches = edge_batch.max().item() + 1
        adj_tensor = torch.zeros(num_batches, self.max_num_nodes, self.max_num_nodes)
        attr_tensor = torch.zeros(num_batches, self.max_num_nodes, self.max_num_nodes, edge_attr.size(1)).to(self.device)
        x_tensor = torch.zeros(num_batches, self.max_num_nodes, x.size(1)).to(self.device)
        if not num_index  is None:
            pre_index = 0
        for j in range(num_batches):
            use_index = edge_batch == j
            use_x_index = batch == j
            use_edge_index = edge_index[:, use_index]
            use_edge_attr = edge_attr[use_index]
            if num_index is not None:
                adj_tensor[j, use_edge_index[0, :] - pre_index, use_edge_index[1, :] - pre_index] = 1
                adj_tensor[j, torch.arange(sum(use_x_index)), torch.arange(sum(use_x_index))] = 1
                attr_tensor[j, use_edge_index[0, :] - pre_index, use_edge_index[1, :] - pre_index] = use_edge_attr
            else:
                x_index = torch.arange(len(batch))[use_x_index]
                index_map = dict(zip(x_index.tolist(), range(len(x_index))))
                copy_use_edge_index = torch.clone(use_edge_index)
                for k in index_map:
                    use_edge_index[copy_use_edge_index == k] = index_map[k]
                adj_tensor[j, use_edge_index[0, :], use_edge_index[1, :]] = 1
                adj_tensor[j, torch.arange(sum(use_x_index)), torch.arange(sum(use_x_index))] = 1
                attr_tensor[j, use_edge_index[0, :], use_edge_index[1, :]] = use_edge_attr
            x_tensor[j, :sum(use_x_index), :] = x[use_x_index, :]
            if not num_index is None:
                pre_index = num_index[j]
        return x_tensor.to(self.device), adj_tensor.to(self.device), attr_tensor.to(self.device)

    def forward(self, x, edge_attr, edge_index, batch, edge_batch, debug=False, verbose=False):
        
        # To generate adj based edge index and edge batch 
        # To batch_size * node_dim * node_dim
        if not hasattr(self, 'decode_type'):
            self.decode_type = 'adj'
        num_batches = max(batch) + 1
        num_index = torch.zeros(num_batches, dtype=torch.long)
        for j in range(num_batches):
            num_index[j] = (batch <= j).sum()
        input_features, adj, attr_tensor = self.generate_adj(x, edge_index, edge_attr, batch, edge_batch, num_index)
        graph_h = self.encoder(x, edge_index, edge_attr, batch, edge_batch)
        if self.decode_type == 'adj':
            x, A, E, z_mu, z_lsgms = self.decoder(graph_h)
            A = torch.sigmoid(A)
            E = torch.softmax(E, dim=-1)
            out_tensor = A.cpu().data
            recon_adj_lower = self.recover_adj_lower(out_tensor)
            recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)
            out_E = E.cpu().data
            recon_adj_lower_E = self.recover_adj_lower_feature(out_E)
            recon_adj_tensor_E = self.recover_full_adj_from_lower_feature(recon_adj_lower_E)
        else:
            xs, g_edge_attr, g_edge_index, g_batch, g_edge_batch, probs, z_mu, z_lsgms = self.decoder(graph_h)
            g_num_batches = max(g_batch) + 1
            g_edge_attr = torch.softmax(g_edge_attr, dim=-1)
            g_input_features, g_adj, g_attr_tensor = self.generate_adj(xs, \
                                g_edge_index, g_edge_attr, g_batch, g_edge_batch)
            recon_adj_tensor = g_adj
            recon_adj_tensor_E = g_attr_tensor
            x = g_input_features
            
            A = g_adj[:, torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].view(g_num_batches, -1)
            E = g_attr_tensor[:, torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1, :].view(g_num_batches, -1, g_edge_attr.size(1))

        
        # HERE is process of out_featuers.
        out1 = torch.softmax(x[:, :, :self.nodedim1], dim=-1)
        out2 = torch.softmax(x[:, :, self.nodedim1:self.nodedim2], dim=-1)
        out3 = torch.softmax(x[:, :, self.nodedim2:self.nodedim3], dim=-1)
        out_features = torch.cat([out1, out2, out3], axis=2)



        # Add some process for out_features.

        # set matching features be degree
        # out_features = torch.sum(recon_adj_tensor, 1)
        adj_data = adj.cpu().data#[0]
        # adj_features = torch.sum(adj_data, 1)
        attr_data = attr_tensor.cpu().data
        input_data = input_features.cpu().data
        loss = torch.zeros(num_batches).to(self.device)
        if debug:
            loss_adj = torch.zeros(num_batches).to(self.device)
            loss_E = torch.zeros(num_batches).to(self.device)
            loss_X = torch.zeros(num_batches).to(self.device)
            loss_KL = torch.zeros(num_batches).to(self.device)
        for j in range(num_batches):
            if verbose:
                print (j)
            S = self.edge_similarity_matrix(adj_data[j], recon_adj_tensor[j].cpu().data, \
                    input_data[j], \
                    out_features[j].cpu().data,
                    self.feature_similarity, attr_data[j], \
                    recon_adj_tensor_E[j].cpu().data, self.feature_similarity)
            init_corr = 1 / self.max_num_nodes
            init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
            assignment = self.mpm(init_assignment, S)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.detach().numpy())
            feature_permuted = self.permute_feature(input_features[j], row_ind, col_ind)
            adj_permuted = self.permute_adj(adj_data[j], row_ind, col_ind)
            attr_permuted = self.permute_adj(attr_tensor[j], row_ind, col_ind)

            feature_permuted_var = (feature_permuted).to(self.device)
            adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()
            adj_vectorized_var = (adj_vectorized).to(self.device)
            attr_vectorized = attr_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()
            attr_vectorized_var = (attr_vectorized).to(self.device)
            if not hasattr(self, 'adj_loss') or self.adj_loss == 'bce':
                adj_recon_loss = self.adj_recon_loss(A[j], adj_vectorized_var)
                attr_recon_loss = self.feature_recon_loss(E[j], attr_vectorized_var)
            elif self.adj_loss == 'ul':
                adj_recon_loss = self.adj_recon_loss_ul(A[j], adj_vectorized_var)
                attr_recon_loss = self.adj_recon_loss_ul(E[j], attr_vectorized_var)
            else:
                adj_recon_loss = self.adj_recon_loss_ul(A[j], adj_vectorized_var)
                use_edges = (A[j] * adj_vectorized_var * attr_vectorized_var.sum(axis=1) == 1)
                attr_recon_loss = self.feature_recon_loss(E[j][use_edges], attr_vectorized_var[use_edges])
                
            fea_recon_loss = self.feature_recon_loss(out_features[j], feature_permuted_var)
            loss_kl = -0.5 * torch.sum(1 + z_lsgms[j] - z_mu[j].pow(2) - z_lsgms[j].exp())
            loss_kl /= self.max_num_nodes * self.max_num_nodes # normalize
            loss[j] = adj_recon_loss + loss_kl + fea_recon_loss + attr_recon_loss
            if debug:
                loss_adj[j] = adj_recon_loss
                loss_E[j] = attr_recon_loss
                loss_X[j] = fea_recon_loss
                loss_KL[j] = loss_kl

        if debug:
            if self.decode_type == 'adj':
                return loss_adj, loss_E, loss_X, loss_KL
            else:
                return (loss_adj, loss_E, loss_X, loss_KL), probs
        if self.decode_type == 'adj':
            return loss
        else:
            return loss, probs

    def forward_test(self, input_features, adj):
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor([[1,1,0,0], [1,1,1,0], [0,1,1,1], [0,0,1,1]])
        adj_features = torch.Tensor([2,3,3,2])

        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor([[1,1,1,0], [1,1,0,1], [1,0,1,0], [0,1,0,1]])
        adj_features1 = torch.Tensor([3,3,2,2])
        S = self.edge_similarity_matrix(adj_data, adj_data1, adj_features, adj_features1,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)

        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())

        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)

        adj_recon_loss = self.adj_recon_loss(permuted_adj, adj_data1)

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_truth, adj_pred)

    def adj_recon_loss_ul(self, adj_truth, adj_pred):
        return torch.pow(adj_truth - adj_pred, 2).mean()*4

    def feature_recon_loss(self, feature_truth, feature_pred):
        return F.binary_cross_entropy(feature_truth, feature_pred)
