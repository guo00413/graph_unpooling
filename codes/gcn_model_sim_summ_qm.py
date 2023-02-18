# This file contains generators (UL- and Adj-based) for QM9
# Also some discriminators that will be used for optimizing chemical property for QM9
# The discriminator for learning distribution of QM9 is in ugcn_model_summ_2.py.


from typing import final
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout
from adj_generator import convertA_to_data_enriched
from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool, global_mean_pool, NNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
from torch.distributions import Categorical
from unpool_layers_simple_v2 import LinkLayer, EdgeAttrConstruct, UnpoolLayerEZ
from unpool_utils import assemble_skip_z, convert_Batch_to_datalist 
# Load some GCN from gcn_model_sim_summ_2
from gcn_model_sim_summ_2 import EdgeAgg_4, RLGCN


class UnpoolGeneratorQ(nn.Module):
    # The UL-based generator for QM9.
    def __init__(self, in_dim, edge_dim, node_dim, node_hidden_dim, edge_hidden_dim, use_x_bn=False, 
                 use_e_bn=False, unpool_bn=False, link_bn=False, attr_bn=False, 
                 skip_z=False, skip_zdim=None, conv_type='nn', device='cpu', 
                 last_act='leaky', link_act='leaky', unpool_para={}, without_aroma=False, 
                 relu_coef=0.05):
        """The generator for QM9 dataset, using UL.
        initialLayer (3-node graph)->MPNN->UL (4-5 node graph)->MPNN->UL (6-9 node graph)->MPNN->Linear

        Args:
            in_dim (int): input dim
            edge_dim (int): output edge dimension
            node_dim (int): output node dimension
            node_hidden_dim (list): the hidden dim for nodes
            edge_hidden_dim (int): the hidden dim for edges
            use_x_bn (bool, optional): if we use BN for node features. Recommend to True. Defaults to False
            use_e_bn (bool, optional): if we use BN for edge features. Defaults to False
            unpool_bn (bool, optional): If we use BN for unpooling layer. Defaults to False.
            link_bn (bool, optional): If we use BN for initial layer and links in unpooling layer. Defaults to False.
            attr_bn (bool, optional): If we use BN for edge attribute generation layer. Defaults to False.
            skip_z (bool, optional): If we use skip-z connection. Defaults to False.
            skip_zdim (list, optional): List of skip-z output dimensions for each unpooling layer. If None, we will use hidden_dim[j]//4. Defaults to None.
            conv_type (str, optional): MPNN type, nn will be edge-conditional MPNN/comp will be 2 layers/ others will be RLGCN. Defaults to 'nn'.
            device (str, optional): Use device. Defaults to 'cpu'.
            last_act (str, optional): The final activation functions. Defaults to 'leaky'.
            link_act (str, optional): The activation function for link layers in UL. Defaults to 'leaky'.
            unpool_para (dict, optional): additional parameters for unpooling layers (e.g. additional_link, etc). Defaults to {}.
            without_aroma (bool, optional): Recommend use as True.. Defaults to False.
            relu_coef (float, optional): The negative slope for LeakyRelus. Defaults to 0.05.
        """
        super(UnpoolGeneratorQ, self).__init__()
        self.relu_coef = relu_coef
        
        
        if not isinstance(node_hidden_dim, list):
            node_hidden_dim = [node_hidden_dim]*6

        if skip_zdim is None:
            # If skip zdim is None then use default number that will be reduced by unpooling.
            # Defaults to node_hiddem_dim's 1/4.
            skip_zdim = [j//4 for j in node_hidden_dim]

        self.skip_zdim = skip_zdim
        self.skip_z = skip_z
        self.device = device

        self.use_x_bn = use_x_bn
        self.use_e_bn = use_e_bn
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.node_hidden_dim = list(node_hidden_dim)
        self.edge_hidden_dim = edge_hidden_dim
        # The initial layer, linear layers to generate 3*d_y, which can be reshaped to N*3*d_y.
        self.in_layer1 = nn.Linear(in_dim, node_hidden_dim[0]*6) # get all nodes' info
        self.in_layer2 = nn.Linear(node_hidden_dim[0]*6, 3*node_hidden_dim[0])
        if self.use_x_bn:
            self.inbn1 = BatchNorm1d(node_hidden_dim[0]*6)
            self.inbn2 = BatchNorm1d(node_hidden_dim[0]) # Second BN will be applied after reshaping, so dim is d_y.

        # Link layer to build edges in initial layer.
        self.first_edge_link = LinkLayer(node_hidden_dim[0], useBN=link_bn, final_act=link_act)
        # To build edge features in initial layer.
        self.edge0_attr_layer = EdgeAttrConstruct(node_hidden_dim[0], edge_hidden_dim, edge_hidden_dim, useBN=attr_bn)
        if conv_type == 'nn':
            self.conv1 = NNConv(node_hidden_dim[0], node_hidden_dim[1], nn=nn.Linear(edge_hidden_dim, node_hidden_dim[0]*node_hidden_dim[1]))
        elif conv_type == 'nn_comp':
            edge_link1 = nn.Sequential(nn.Linear(edge_hidden_dim, edge_hidden_dim), nn.LeakyReLU(self.relu_coef), \
                                        nn.Linear(edge_hidden_dim, edge_hidden_dim), nn.LeakyReLU(self.relu_coef), \
                                        nn.Linear(edge_hidden_dim, node_hidden_dim[0]*node_hidden_dim[1]))
            self.conv1 = NNConv(node_hidden_dim[0], node_hidden_dim[1], nn=edge_link1)
        else:
            self.conv1 = RLGCN(node_hidden_dim[0], node_hidden_dim[1], edge_hidden_dim)
        self.unpool1 = UnpoolLayerEZ(node_hidden_dim[1], node_hidden_dim[1]//2, edge_dim=edge_hidden_dim, fix_point=1, useBN=unpool_bn, \
                                    inner_link=None, outer_link=None, link_bn=link_bn, link_act=link_act, roll_ones=1, **unpool_para)
        
        # If we don't have skip_z, the node dimension will be reduced by d_y//4.
        if self.skip_z:
            node_hidden_dim[1] = self.node_hidden_dim[1]
        else:
            node_hidden_dim[1] = node_hidden_dim[1] - node_hidden_dim[1]//4
        self.edge1_attr_layer = EdgeAttrConstruct(node_hidden_dim[1], edge_hidden_dim, edge_hidden_dim, useBN=attr_bn)

        if conv_type == 'nn':
            self.conv2 = NNConv(node_hidden_dim[1], node_hidden_dim[2], nn=nn.Linear(edge_hidden_dim, node_hidden_dim[1]*node_hidden_dim[2]))
        elif conv_type == 'nn_comp':
            edge_link2 = nn.Sequential(nn.Linear(edge_hidden_dim, edge_hidden_dim), nn.LeakyReLU(self.relu_coef), \
                                        nn.Linear(edge_hidden_dim, edge_hidden_dim), nn.LeakyReLU(self.relu_coef), \
                                        nn.Linear(edge_hidden_dim, node_hidden_dim[1]*node_hidden_dim[2]))
            self.conv2 = NNConv(node_hidden_dim[1], node_hidden_dim[2], nn=edge_link2)
        else:
            self.conv2 = RLGCN(node_hidden_dim[1], node_hidden_dim[2], edge_hidden_dim)
        self.unpool2 = UnpoolLayerEZ(node_hidden_dim[2], node_hidden_dim[2]//2, edge_dim=edge_hidden_dim, fix_point=1, useBN=unpool_bn, \
                                    inner_link=None, outer_link=None, link_bn=link_bn, link_act=link_act, roll_ones=1, **unpool_para)

        # If we don't have skip_z, the node dimension will be reduced by d_y//4.
        if self.skip_z:
            node_hidden_dim[2] = self.node_hidden_dim[2]
        else:
            node_hidden_dim[2] = node_hidden_dim[2] - node_hidden_dim[2]//4
        self.edge2_attr_layer = EdgeAttrConstruct(node_hidden_dim[2], edge_hidden_dim, edge_hidden_dim, useBN=attr_bn)
        if conv_type == 'nn':
            self.conv4 = NNConv(node_hidden_dim[2], node_hidden_dim[3], nn=nn.Linear(edge_hidden_dim, node_hidden_dim[2]*node_hidden_dim[3]))
        elif conv_type == 'nn_comp':
            edge_link4 = nn.Sequential(nn.Linear(edge_hidden_dim, edge_hidden_dim), nn.LeakyReLU(self.relu_coef), \
                                        nn.Linear(edge_hidden_dim, edge_hidden_dim), nn.LeakyReLU(self.relu_coef), \
                                        nn.Linear(edge_hidden_dim, node_hidden_dim[2]*node_hidden_dim[3]))
            self.conv4 = NNConv(node_hidden_dim[2], node_hidden_dim[3], nn=edge_link4)
        else:
            self.conv4 = RLGCN(node_hidden_dim[2], node_hidden_dim[3], edge_hidden_dim)

        self.final_node_layer0 = nn.Linear(node_hidden_dim[3], node_hidden_dim[4])
            
        # with final node feature, to produce some additional information for final edge feature:
        self.edge4_attr_layer = EdgeAttrConstruct(node_dim, edge_hidden_dim, edge_hidden_dim, useBN=attr_bn)
        # The final edge feature will depends on final edge, output of edge4_attr_layer, agg of final nodes.
        self.final_edge_layer0 = nn.Linear(edge_hidden_dim*2 + node_hidden_dim[4], edge_hidden_dim)
        self.final_node_layer1 = nn.Linear(node_hidden_dim[4], node_dim)
        self.final_edge_layer1 = nn.Linear(edge_hidden_dim, edge_dim)
        
        # Add BNs for node features.
        if self.use_x_bn:
            node_hidden_dim = list(self.node_hidden_dim)

            self.xbn_0 = BatchNorm1d(node_hidden_dim[0])
            self.xbn_1 = BatchNorm1d(node_hidden_dim[1])

            self.xbn_2 = BatchNorm1d(node_hidden_dim[1] - node_hidden_dim[1]//4)
            if self.skip_z:
                node_hidden_dim[0] = self.node_hidden_dim[0]
            else:
                node_hidden_dim[0] = node_hidden_dim[0] - node_hidden_dim[0]//4
            self.xbn_3 = BatchNorm1d(node_hidden_dim[2])


            self.xbn_4 = BatchNorm1d(node_hidden_dim[2] - node_hidden_dim[2]//4)
            if self.skip_z:
                node_hidden_dim[2] = self.node_hidden_dim[2]
            else:
                node_hidden_dim[2] = node_hidden_dim[2] - node_hidden_dim[2]//4
            self.xbn_5 = BatchNorm1d(node_hidden_dim[3])
            self.xbn_6 = BatchNorm1d(node_hidden_dim[4])
        # Add BNs for edge features.
        if self.use_e_bn:
            self.ebn_0 = BatchNorm1d(self.edge_hidden_dim)
            self.ebn_1 = BatchNorm1d(self.edge_hidden_dim)
            self.ebn_2 = BatchNorm1d(self.edge_hidden_dim)
            self.ebn_3 = BatchNorm1d(self.edge_hidden_dim)
        self.last_act = last_act
        self.without_ar = without_aroma
        # Add skip-z layers.
        if self.skip_z:
            if self.use_x_bn:
                skip_zdim = self.skip_zdim[1]
                self.skip_1 = nn.Sequential(nn.Linear(self.in_dim, skip_zdim*10), 
                                                nn.BatchNorm1d(skip_zdim*10), 
                                                nn.LeakyReLU(0.1), \
                                                nn.Linear(skip_zdim*10, skip_zdim*5), \
                                                nn.BatchNorm1d(skip_zdim*5), 
                                                nn.LeakyReLU(0.1), \
                                                )
                skip_zdim = self.skip_zdim[2]
                self.skip_2 = nn.Sequential(nn.Linear(self.in_dim, skip_zdim*15), 
                                                nn.BatchNorm1d(skip_zdim*15), 
                                                nn.LeakyReLU(0.1), \
                                                nn.Linear(skip_zdim*15, skip_zdim*9), \
                                                nn.BatchNorm1d(skip_zdim*9), 
                                                nn.LeakyReLU(0.1), \
                                                )
            else:
                self.skip_1 = nn.Sequential(nn.Linear(self.in_dim, self.skip_zdim*10), 
                                                nn.LeakyReLU(0.1), \
                                                nn.Linear(self.skip_zdim*10, self.skip_zdim*5), \
                                                nn.LeakyReLU(0.1), \
                                                )
                self.skip_2 = nn.Sequential(nn.Linear(self.in_dim, self.skip_zdim*15), 
                                                nn.LeakyReLU(0.1), \
                                                nn.Linear(self.skip_zdim*15, self.skip_zdim*9), \
                                                nn.LeakyReLU(0.1), \
                                                )

    def forward(self, z, gumbel_sample=True, tau=1.0, hard=True, **kwarg):
        # A rough procedure:
        #  in rand -> 3*node_hidden
        #  build edge_index for those 3.
        #  build edge_attr for those 3.
        #  MPNN
        #  Unpool to size of 4-5 (with 1 fix point and 1 rolling point in UL).
        #  build edge_attr for those 4-5
        #  MPNN
        #  Unpool to size of 6-9 (with 1 fix point and 1 rolling point in UL).
        #  build edge_attr for those 6-9
        #  MPNN
        # Final layers.
        # Output: 
        # 1. Batch of fake data.
        # 2. total log probability
        if not hasattr(self, 'relu_coef'):
            self.relu_coef = 0.05
        self.device = self.final_edge_layer1.weight.device
        n = z.size(0)
        xs = self.in_layer1(z)
        if self.use_x_bn:
            xs = self.inbn1(xs)
        xs = F.leaky_relu(xs, self.relu_coef)
        xs = self.in_layer2(xs)
        xs = xs.view(3*n, self.node_hidden_dim[0])
        if self.use_x_bn:
            xs = self.inbn2(xs)
        ################################################################# 

        xs = F.leaky_relu(xs, self.relu_coef)
        # Reshape to N*3*d_y
        xs = xs.view(n, 3, self.node_hidden_dim[0])
        # Next, build edges.
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

        # With edges, shape back to 3N*d_y, following torch_geometric's structure.
        xs = xs.view(3*n, self.node_hidden_dim[0])
        
        edge_attr = self.edge0_attr_layer(xs, edge_index)
        if self.use_e_bn:
            edge_attr = self.ebn_0(edge_attr)
        edge_attr = F.leaky_relu(edge_attr, self.relu_coef)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)

        # edge_index, edge_attr = edge_index.to(self.device), edge_attr.to(self.device)
        batch = torch.arange(n).view(-1, 1).repeat(1, 3).view(-1)

        edge_batch = torch.zeros(edge_index.size(1)).type(torch.LongTensor)
        edge_batch.scatter_(0, torch.arange(edge_index.size(1)), batch[edge_index[0]])

        # Up to here, we finished initial layer construction.
        
        # MPNN 1
        xs = self.conv1(xs, edge_index=edge_index, edge_attr=edge_attr)

        if self.use_x_bn:
            xs = self.xbn_1(xs)
        xs = F.leaky_relu(xs, self.relu_coef)
        # UL 1
        xs, batch, edge_index, edge_batch, prob_edge1 = self.unpool1(xs, edge_index, edge_attr, batch)

        if self.skip_z:
            add_xs = self.skip_1(z)
            add_xs = add_xs.view(add_xs.size(0), 5, self.skip_zdim[1])
            add_xs = assemble_skip_z(add_xs, batch, self.device)
            xs = torch.cat([xs, add_xs], axis=1)

        edge_attr = self.edge1_attr_layer(xs, edge_index)
        if self.use_e_bn:
            edge_attr = self.ebn_1(edge_attr)
        edge_attr = F.leaky_relu(edge_attr, self.relu_coef)
        # MPNN 2
        xs = self.conv2(xs, edge_index=edge_index, edge_attr=edge_attr)
        if self.use_x_bn:
            xs = self.xbn_3(xs)
        xs = F.leaky_relu(xs, self.relu_coef)
        # UL 2
        xs, batch, edge_index, edge_batch, prob_edge2 = self.unpool2(xs, edge_index, edge_attr, batch)

        if self.skip_z:
            add_xs = self.skip_2(z)
            add_xs = add_xs.view(add_xs.size(0), 9, self.skip_zdim[2])
            add_xs = assemble_skip_z(add_xs, batch, self.device)
            xs = torch.cat([xs, add_xs], axis=1)

        edge_attr = self.edge2_attr_layer(xs, edge_index)
        if self.use_e_bn:
            edge_attr = self.ebn_2(edge_attr)
        edge_attr = F.leaky_relu(edge_attr, self.relu_coef)

        # MPNN 3
        xs = self.conv4(xs, edge_index=edge_index, edge_attr=edge_attr)
        if self.use_x_bn:
            xs = self.xbn_5(xs)
        xs = F.leaky_relu(xs, self.relu_coef)

        # Final layers
        xs = self.final_node_layer0(xs)
        if self.use_x_bn:
            xs = self.xbn_6(xs)
        xf = F.leaky_relu(xs, self.relu_coef)
        xs = self.final_node_layer1(xf)

        if self.last_act == 'leaky':
            xs = F.leaky_relu(xs, self.relu_coef)
        elif self.last_act == 'tanh':
            xs = F.tanh(xs)
        elif self.last_act == 'sigmoid':
            xs = F.sigmoid(xs)
        else:
            xs = F.leaky_relu(xs, self.relu_coef)

        # xs = F.softmax(xs, dim=1) + 1e-4
        x = torch.zeros_like(xs)
        x[:, :4] = F.softmax(xs[:, :4], dim=1) + 1e-5
        x[:, 4:7] = F.softmax(xs[:, 4:7], dim=1) + 1e-5
        x[:, 7:10] = F.softmax(xs[:, 7:10], dim=1) + 1e-5
        if hasattr(self, 'without_ar') and self.without_ar:
            pass
        else:
            x[:, 10:12] = F.softmax(xs[:, 10:12], dim=1) + 1e-5

        # Final Edge
        edge_attr = self.final_edge_layer0(torch.cat([edge_attr, \
            self.edge4_attr_layer(xs, edge_index), F.leaky_relu(xf[edge_index[0]] + \
                xf[edge_index[1]], self.relu_coef)], dim=1))
        if self.use_e_bn:
            edge_attr = self.ebn_3(edge_attr)
        edge_attr = F.leaky_relu(edge_attr, self.relu_coef)
        edge_attr = self.final_edge_layer1(edge_attr)
        if self.last_act == 'leaky':
            edge_attr = F.leaky_relu(edge_attr, self.relu_coef)
        elif self.last_act == 'tanh':
            edge_attr = F.tanh(edge_attr)
        elif self.last_act == '2tanh':
            edge_attr = F.tanh(edge_attr)*2
        elif self.last_act == '3tanh':
            edge_attr = F.tanh(edge_attr)*3
        elif self.last_act == 'sigmoid':
            edge_attr = F.sigmoid(edge_attr)
        else:
            edge_attr = F.leaky_relu(edge_attr, self.relu_coef)
        edge_attr = F.softmax(edge_attr, dim=1) + 1e-4

        if xs.isinf().sum() > 0:
            print ("Possible error....")
            xs[xs.isinf()] = torch.sign(xs[xs.isinf()]) * 3.318e38
        if gumbel_sample:
            xs1 = F.gumbel_softmax(torch.log(x[:, :4]), dim=1, hard=hard, tau=tau)
            xs2 = F.gumbel_softmax(torch.log(x[:, 4:7]), dim=1, hard=hard, tau=tau)
            xs3 = F.gumbel_softmax(torch.log(x[:, 7:10]), dim=1, hard=hard, tau=tau)
            if hasattr(self, 'without_ar') and self.without_ar:
                pass
            else:
                xs4 = F.gumbel_softmax(torch.log(x[:, 10:12]), dim=1, hard=hard, tau=tau)
            edge_attr = F.gumbel_softmax(torch.log(edge_attr), dim=1, hard=hard, tau=tau)
        if hasattr(self, 'without_ar') and self.without_ar:
            xs = torch.cat([xs1, xs2, xs3], axis=1)
        else:
            xs = torch.cat([xs1, xs2, xs3, xs4[:, :1]], axis=1)

        fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
        fake_data1 = Batch.from_data_list(\
            [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                    for j in fake_data], \
                    follow_batch=['edge_index']).to(self.device)
        return fake_data1, prob_edge1.to(self.device) + prob_edge2.to(self.device) + prob_init_edge



class pre_GCNModel_edge_reward(nn.Module):
    """
    A reward prediction GCN.
    
    uses edge-conditional MPNN and global readout to predict a numeric value from a graph.
    """
    def __init__(self, in_dim, hidden_dim, edge_dim, edge_hidden_dim, lin_hidden_dim=None, out_hidden_dim=256, device='cpu', \
                check_batch=None, useBN=False, droprate=None, pool_method='trivial', \
                add_edge_link=False, add_conv_link=False, outBN=False, out_drop=0.3, \
                out_divide=4.0, add_edge_agg=True, real_trivial=False, final_layers=2, \
                add_trivial_feature=True, ln_comp=False, last_act='sigmoid', relu_coef=0.05):
        super(pre_GCNModel_edge_reward, self).__init__()
        self.relu_coef = relu_coef
        self.final_layers = final_layers
        if lin_hidden_dim is None:
            lin_hidden_dim = hidden_dim
        self.lin_hidden_dim = lin_hidden_dim

        if add_conv_link:
            self.conv1 = NNConv(in_dim, hidden_dim, nn=nn.Linear(edge_dim, in_dim*hidden_dim))
            self.conv2 = NNConv(hidden_dim, hidden_dim, nn=nn.Linear(edge_dim, hidden_dim*hidden_dim)) 
            in_dim_hidden = hidden_dim + in_dim
            if ln_comp:
                if useBN:
                    self.sig_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.BatchNorm1d(lin_hidden_dim), \
                        nn.LeakyReLU(self.relu_coef), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
                    )
                    self.tanh_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.BatchNorm1d(lin_hidden_dim), \
                        nn.LeakyReLU(self.relu_coef), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
0                    )
                else:
                    self.sig_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.LeakyReLU(self.relu_coef), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
                    )
                    self.tanh_ln = nn.Sequential(
                        nn.Linear(in_dim_hidden, lin_hidden_dim), \
                        nn.LeakyReLU(self.relu_coef), \
                        nn.Linear(lin_hidden_dim, lin_hidden_dim), \
                    )
            else:
                self.sig_ln = nn.Linear(in_dim_hidden, lin_hidden_dim)
                self.tanh_ln = nn.Linear(in_dim_hidden, lin_hidden_dim)

        if add_edge_link:
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
        self.out_dim = 0
        self.add_edge_link = add_edge_link
        self.add_conv_link = add_conv_link

        if self.add_conv_link:
            self.out_dim += self.lin_hidden_dim
        if add_edge_link:
            self.out_dim += edge_hidden_dim

        if self.final_layers == 2:
            self.final_lin1 = nn.Linear(self.out_dim, out_hidden_dim)
            self.final_lin2 = nn.Linear(out_hidden_dim, 1)
            self.output_dim = 1
        elif self.final_layers == 1:
            self.final_lin2 = nn.Linear(self.out_dim, 1)
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
            self.out_bn2 = nn.BatchNorm1d(out_hidden_dim)

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
        self.last_act = last_act
        # switchers for different components.
        self.ex2_ind = 1.0
        self.x0_ind = 1.0
        self.e0_ind = 1.0
        self.e1_ind = 1.0
        self.conv_ind = 1.0
        self.eagg_ind = 1.0



    def forward(self, data=None, x=None, edge_index=None, edge_attr=None, batch=None, edge_batch=None):
        # GCNs:
        if not hasattr(self, 'relu_coef'):
            self.relu_coef = 0.05
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
            ex = F.leaky_relu(ex, self.relu_coef)
            ex = self.edge_agg2(ex, edge_index=edge_index, edge_attr=edge_attr)
            if self.useBN:
                ex = self.edgebn2(ex)
            ex = F.leaky_relu(ex, self.relu_coef)



        if self.add_conv_link:
            x = self.conv1(x0, edge_index=edge_index, edge_attr=edge_attr)
            if self.useBN:
                x = self.bn1(x)
            x = F.leaky_relu(x, self.relu_coef)
            x = self.conv2(x, edge_index=edge_index, edge_attr=edge_attr)
            if self.useBN:
                x = self.bn2(x)
            x = F.leaky_relu(x, self.relu_coef)
            if self.droprate:
                x = self.drop2(x)
        # Readout:
            x = torch.cat([x, x0], axis=1)
            sx = torch.clip(self.sig_ln(x), -30, 30)
            x = 1/(1+torch.exp(sx))*torch.tanh(self.tanh_ln(x))
            if self.droprate:
                x = self.drop3(x)
            out_x = global_add_pool(x, batch=batch)

        sum_pool = torch.zeros(out_x.size(0), 0).to(self.device)

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
            out_x = self.final_lin1(out_x)
            if self.outBN:
                out_x = self.out_bn2(out_x)
            out_x = F.leaky_relu(out_x, self.relu_coef)
            if self.last_act == 'sigmoid':
                out_x = F.sigmoid(self.final_lin2(out_x)/self.out_divide)
            else:
                out_x = F.tanh(self.final_lin2(out_x)/self.out_divide)
        elif self.final_layers == 1:
            if self.last_act == 'sigmoid':
                out_x = F.sigmoid(self.final_lin1(out_x)/self.out_divide)
            else:
                out_x = F.tanh(self.final_lin1(out_x)/self.out_divide)
        else:
            pass
        return out_x


class AdjGenerator_qm(nn.Module):
    # The Adj-based generator for QM9
    def __init__(self, in_dim, hidden_dims=[64, 128, 32, 32], \
        useBN=False, nodeNum=9, edge_outdim=4, node_outdim=11, bias=True, device='cpu', \
        without_aroma=False
        ):
        """_summary_

        Args:
            in_dim (_type_): Input z's dimension.
            hidden_dims (list, optional): Dimensions of hidden layers. Defaults to [64, 128, 32, 32].
            useBN (bool, optional): If use BN. Defaults to True.
            nodeNum (int, optional): Number of node in output graph. Defaults to 9.
            edge_outdim (int, optional): Dimension of edge feature in output. Defaults to 4.
            node_outdim (int, optional): Dimension of node feature in output. Defaults to 11.
            bias (bool, optional): If we add bias in linear layers. Defaults to True.
            device (str, optional): which device to use. Defaults to 'cpu'.
            without_aroma (bool, optional): Shall we not use aroma - Recommend to True. Defaults to False.
        """
        super(AdjGenerator_qm, self).__init__()
        self.useBN = useBN
        self.nodeNum = nodeNum
        self.edge_outdim = edge_outdim
        self.node_outdim = node_outdim
        self.hidden_dims = hidden_dims
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.in_dim = in_dim
        pre_dim = in_dim
        self.without_ar = without_aroma
        for j in hidden_dims:
            self.linears.append(nn.Linear(pre_dim, j, bias=bias))
            pre_dim = j
            if self.useBN:
                self.bns.append(nn.BatchNorm1d(j))

        self.final_nodeout = nn.Linear(pre_dim, nodeNum*node_outdim, bias=bias)
        self.final_edgeout = nn.Linear(pre_dim, nodeNum*nodeNum*edge_outdim, bias=bias)
        self.device = device

    def generate_XA(self, z):
        """Given z, produce X/A (probabilities)

        Args:
            z (float tensor): input random tensor (N * d)

        Returns:
            Float tensor (N*9*10): output node features (with probabilities).
            Float tensor (N*9*9*4): output edge features (with probabilities).
        """
        for j in range(len(self.linears)):
            z = self.linears[j](z)
            if self.useBN:
                z = self.bns[j](z)
            z = F.leaky_relu(z, 0.1)
        out_X = F.leaky_relu(self.final_nodeout(z), 0.1).view(-1, self.nodeNum, self.node_outdim)
        out_A = F.leaky_relu(self.final_edgeout(z), 0.1).view(-1, self.nodeNum, self.nodeNum, self.edge_outdim)
        out_A = (out_A + torch.transpose(out_A, 1, 2))/2
        ox1 = F.softmax(out_X[:, :, :5], dim=-1)
        ox2 = F.softmax(out_X[:, :, 5:8], dim=-1)
        ox3 = F.softmax(out_X[:, :, 8:11], dim=-1)
        if self.without_ar:
            pass
            out_x = torch.cat([ox1, ox2, ox3], dim=2)
        else:
            out_X = torch.cat([out_X, torch.ones(len(out_X), out_X.size(1), 1).to(out_X.device)], dim=2)
            ox4 = F.softmax(out_X[:, :, 11:13], dim=-1)
            out_x = torch.cat([ox1, ox2, ox3, ox4], dim=2)
        return out_x, F.softmax(out_A, dim=-1)

    def forward(self, z, tau=1.0, **kargs):
        """Given z, produce a Batch of fake data.

        Args:
            z (float tensor): input random tensor (N * d)
            tau (float, optional): Tau for Gumbel Softmax. Defaults to 1.0.
            kargs (dict): additional kwargs for `convertA_to_data_enriched`.

        Returns:
            torch_geometric.Batch: A batch of generated fake data.
        """
        out_X, out_A = self.generate_XA(z)
        node_dim = 13 # with AR, we need 2 more dimensions for atom's aroma type.
        if self.without_ar:
            node_dim = 11
        fake_data = [convertA_to_data_enriched(out_X[j], out_A[j], node_dim=node_dim, atom_dim=5, \
                        edge_dim=4, without_ar=self.without_ar, 
                        gumbel=True, \
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

