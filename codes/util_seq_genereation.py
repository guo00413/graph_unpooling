# This file is for sequentially generation only.
# Visualization purpose..
#   sequential_generation_zinc for ZINC
#   sequential_generation_va for QM9


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
from adj_generator import convertA_to_data
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch.distributions import Categorical
from unpool_utils import assemble_skip_z, convert_Batch_to_datalist 

def sequential_generation_zinc(generator, z, gumbel_sample=True, tau=1.0, hard=True):
    """Given trained generator and input random vector z. This function is used for ZINC's generator. 
    generate generated data after initial layer, each unpooling layer and final output.
    
    Return
        fakedata1-6 for results after initial layer, 1st UL, 2nd UL, 3rd UL, 4th UL and final output.
    """
    generator.device = generator.final_edge_layer1.weight.device
    generator.eval()
    n = z.size(0)
    xs = generator.in_layer1(z)
    ########## Possibly, we don't need to run BN and Directly use Relu.
    xs = xs.view(3*n, generator.node_hidden_dim[0])
    if generator.use_x_bn:
        xs = generator.inbn2(xs)
    ################################################################# 

    xs = F.leaky_relu(xs, 0.05)
    xs = xs.view(n, 3, generator.node_hidden_dim[0])
    a1 = generator.first_edge_link(xs[:, 0, :], xs[:, 1, :])
    a2 = generator.first_edge_link(xs[:, 1, :], xs[:, 2, :])
    a3 = generator.first_edge_link(xs[:, 0, :], xs[:, 2, :])
    n1 = torch.arange(n).to(generator.device)*3
    n2 = torch.arange(n).to(generator.device)*3 + 1
    n3 = torch.arange(n).to(generator.device)*3 + 2
    edge_prob = torch.zeros(n, 4).to(generator.device)
    # TODO: for REINFORCE: change the inplace fashion.
    edge_prob[:, 0] = (a1[:, 0] + a2[:, 1] + a3[:, 1])/3
    edge_prob[:, 1] = (a2[:, 0] + a1[:, 1] + a3[:, 1])/3
    edge_prob[:, 2] = (a3[:, 0] + a2[:, 1] + a1[:, 1])/3
    edge_prob[:, 3] = (a1[:, 1] + a2[:, 1] + a3[:, 1])/3
    edge_prob = F.softmax(edge_prob, dim=1) + 1e-4
    m = Categorical(edge_prob)
    edge0_links = m.sample().to(generator.device)

    prob_init_edge = m.log_prob(edge0_links).to(generator.device)
    # 0: no 0,1 link
    # 1: no 1,2 link
    # 2: no 0,2 link
    # 3: all links
    en0 = (edge0_links == 0)
    en1 = (edge0_links == 1)
    en2 = (edge0_links == 2)
    en3 = (edge0_links == 3)
    edge_index = torch.LongTensor(2, 2*(en0.sum() + en1.sum() + en2.sum()) + 3*en3.sum()).to(generator.device)
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


    xs = xs.view(3*n, generator.node_hidden_dim[0])

    edge_attr = generator.edge0_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_0(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)

    batch = torch.arange(n).view(-1, 1).repeat(1, 3).view(-1)

    edge_batch = torch.zeros(edge_index.size(1)).type(torch.LongTensor)
    edge_batch.scatter_(0, torch.arange(edge_index.size(1)), batch[edge_index[0]])
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data1 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv1(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_1(xs)
    xs = F.leaky_relu(xs, 0.05)

    xs, batch, edge_index, edge_batch, prob_edge1 = generator.unpool1(xs, edge_index, edge_attr, batch)
    if generator.skip_z:
        add_xs = generator.skip_1(z)
        add_xs = add_xs.view(add_xs.size(0), 5, generator.skip_zdim[1])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge1_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_1(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data2 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv2(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_2(xs)
    xs = F.leaky_relu(xs, 0.05)


    # xs = generator.conv3(xs, edge_index=edge_index, edge_attr=edge_attr)
    # if generator.use_x_bn:
    #     xs = generator.xbn_3(xs)
    # xs = F.leaky_relu(xs, 0.05)

    xs, batch, edge_index, edge_batch, prob_edge2 = generator.unpool2(xs, edge_index, edge_attr, batch)
    if generator.skip_z:
        add_xs = generator.skip_2(z)
        add_xs = add_xs.view(add_xs.size(0), 9, generator.skip_zdim[2])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge2_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_2(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data3 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv5(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_5(xs)
    xs = F.leaky_relu(xs, 0.05)


    xs, batch, edge_index, edge_batch, prob_edge3 = generator.unpool3(xs, edge_index, edge_attr, batch)
    if generator.skip_z:
        add_xs = generator.skip_3(z)
        add_xs = add_xs.view(add_xs.size(0), 18, generator.skip_zdim[4])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge3_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_3(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data4 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv6(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_6(xs)
    xs = F.leaky_relu(xs, 0.05)

    # xs = generator.conv7(xs, edge_index=edge_index, edge_attr=edge_attr)
    # if generator.use_x_bn:
    #     xs = generator.xbn_7(xs)
    # xs = F.leaky_relu(xs, 0.05)

    xs, batch, edge_index, edge_batch, prob_edge4 = generator.unpool4(xs, edge_index, edge_attr, batch)
    if generator.skip_z:
        add_xs = generator.skip_4(z)
        add_xs = add_xs.view(add_xs.size(0), 36, generator.skip_zdim[6])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge4_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_4(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data5 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv8(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_8(xs)
    xs = F.leaky_relu(xs, 0.05)

    xs = generator.final_node_layer0(xs)
    if generator.use_x_bn:
        xs = generator.xbn_fn_0(xs)
    xf = F.leaky_relu(xs, 0.05)

    xs = generator.final_node_layer1(xf)
    if generator.last_act == 'leaky':
        xs = F.leaky_relu(xs, 0.05)
    elif generator.last_act == 'tanh':
        xs = F.tanh(xs)
    elif generator.last_act == 'sigmoid':
        xs = F.sigmoid(xs)
    else:
        xs = F.leaky_relu(xs, 0.05)
    if hasattr(generator, 'without_ar') and generator.without_ar:
        pass
    else:
        xs = torch.cat([xs, torch.ones(len(xs), 1).to(xs.device)], axis=1)
    x = torch.zeros_like(xs)
    x[:, :9] = F.softmax(xs[:, :9], dim=1) + 1e-5
    x[:, 9:12] = F.softmax(xs[:, 9:12], dim=1) + 1e-5
    x[:, 12:15] = F.softmax(xs[:, 12:15], dim=1) + 1e-5
    if hasattr(generator, 'without_ar') and generator.without_ar:
        pass
    else:
        x[:, 15:17] = F.softmax(xs[:, 15:17], dim=1) + 1e-5


    edge_attr = generator.final_edge_layer0(torch.cat([edge_attr, generator.edge5_attr_layer(xs[:, :generator.node_dim], edge_index), F.leaky_relu(xf[edge_index[0]] + xf[edge_index[1]], 0.05)], dim=1))
    if generator.use_e_bn:
        edge_attr = generator.ebn_fn_0(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)

    edge_attr = generator.final_edge_layer1(edge_attr)
    if generator.last_act == 'leaky':
        edge_attr = F.leaky_relu(edge_attr, 0.05)
    elif generator.last_act == 'tanh':
        edge_attr = F.tanh(edge_attr)
    elif generator.last_act == 'sigmoid':
        edge_attr = F.sigmoid(edge_attr)
    else:
        edge_attr = F.leaky_relu(edge_attr, 0.05)
    edge_attr = F.softmax(edge_attr, dim=1) + 1e-5

    if gumbel_sample:
        xs1 = F.gumbel_softmax(torch.log(x[:, :9]), dim=1, hard=hard, tau=tau)
        xs2 = F.gumbel_softmax(torch.log(x[:, 9:12]), dim=1, hard=hard, tau=tau)
        xs3 = F.gumbel_softmax(torch.log(x[:, 12:15]), dim=1, hard=hard, tau=tau)
        if hasattr(generator, 'without_ar') and generator.without_ar:
            pass
        else:
            xs4 = F.gumbel_softmax(torch.log(x[:, 15:17]), dim=1, hard=hard, tau=tau)
        edge_attr = F.gumbel_softmax(torch.log(edge_attr), dim=1, hard=hard, tau=tau)

    if hasattr(generator, 'without_ar') and generator.without_ar:
        xs = torch.cat([xs1, xs2, xs3], axis=1)
    else:
        xs = torch.cat([xs1, xs2, xs3, xs4[:, :1]], axis=1)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data6 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)
    return fake_data1, fake_data2, fake_data3, fake_data4, fake_data5, fake_data6



def sequential_generation(generator, z):
    """Given trained generator and input random vector z. This function is used for QM9's generator. 
    This function is backup, please use sequential_generation_va
    generate generated data after initial layer, each unpooling layer and final output.
    
    Return
        fakedata1-4 for results after initial layer, 1st UL, 2nd UL and final output.
    """
    generator.eval()
    generator.device = generator.final_edge_layer1.weight.device
    n = z.size(0)
    xs = generator.in_layer1(z)
    if generator.use_x_bn:
        xs = generator.inbn1(xs)
    xs = F.leaky_relu(xs, 0.05)
    xs = generator.in_layer2(xs)
    ########## Possibly, we don't need to run BN and Directly use Relu.
    xs = xs.view(3*n, generator.node_hidden_dim[0])
    if generator.use_x_bn:
        xs = generator.inbn2(xs)
    ################################################################# 

    xs = F.leaky_relu(xs, 0.05)
    xs = xs.view(n, 3, generator.node_hidden_dim[0])
    a1 = generator.first_edge_link(xs[:, 0, :], xs[:, 1, :])
    a2 = generator.first_edge_link(xs[:, 1, :], xs[:, 2, :])
    a3 = generator.first_edge_link(xs[:, 0, :], xs[:, 2, :])
    n1 = torch.arange(n).to(generator.device)*3
    n2 = torch.arange(n).to(generator.device)*3 + 1
    n3 = torch.arange(n).to(generator.device)*3 + 2
    edge_prob = torch.zeros(n, 4).to(generator.device)
    edge_prob[:, 0] = (a1[:, 0] + a2[:, 1] + a3[:, 1])/3
    edge_prob[:, 1] = (a2[:, 0] + a1[:, 1] + a3[:, 1])/3
    edge_prob[:, 2] = (a3[:, 0] + a2[:, 1] + a1[:, 1])/3
    edge_prob[:, 3] = (a1[:, 1] + a2[:, 1] + a3[:, 1])/3
    edge_prob = F.softmax(edge_prob, dim=1) + 1e-4
    m = Categorical(edge_prob)
    edge0_links = m.sample().to(generator.device)

    prob_init_edge = m.log_prob(edge0_links).to(generator.device)
    # 0: no 0,1 link
    # 1: no 1,2 link
    # 2: no 0,2 link
    # 3: all links
    en0 = (edge0_links == 0)
    en1 = (edge0_links == 1)
    en2 = (edge0_links == 2)
    en3 = (edge0_links == 3)
    edge_index = torch.LongTensor(2, 2*(en0.sum() + en1.sum() + en2.sum()) + 3*en3.sum()).to(generator.device)
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


    xs = xs.view(3*n, generator.node_hidden_dim[0])
    # if generator.use_x_bn:
    #     xs = generator.xbn_0(xs)
    # xs = F.leaky_relu(xs, 0.1)
    edge_attr = generator.edge0_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_0(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)

    # edge_index, edge_attr = edge_index.to(generator.device), edge_attr.to(generator.device)
    batch = torch.arange(n).view(-1, 1).repeat(1, 3).view(-1)

    # print (edge_index, batch)

    edge_batch = torch.zeros(edge_index.size(1)).type(torch.LongTensor)
    edge_batch.scatter_(0, torch.arange(edge_index.size(1)), batch[edge_index[0]])
    # print (xs, edge_index, edge_attr)
    xs = generator.conv1(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_1(xs)
    xs = F.leaky_relu(xs, 0.05)

    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_1 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)


    if generator.unpool_type == 'normal':
        xs, batch, edge_index, edge_batch, prob_edge1 = generator.unpool1(xs, edge_index, batch)
    else:
        xs, batch, edge_index, edge_batch, prob_edge1 = generator.unpool1(xs, edge_index, edge_attr, batch)

    if generator.skip_z:
        add_xs = generator.skip_1(z)
        add_xs = add_xs.view(add_xs.size(0), 5, generator.skip_zdim[1])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge1_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_1(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_2 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv2(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_3(xs)
    xs = F.leaky_relu(xs, 0.05)
    # xs = generator.conv3(xs, edge_index=edge_index, edge_attr=edge_attr)


    if generator.unpool_type == 'normal':
        xs, batch, edge_index, edge_batch, prob_edge2 = generator.unpool2(xs, edge_index, batch)
    else:
        xs, batch, edge_index, edge_batch, prob_edge2 = generator.unpool2(xs, edge_index, edge_attr, batch)

    if generator.skip_z:
        add_xs = generator.skip_2(z)
        add_xs = add_xs.view(add_xs.size(0), 9, generator.skip_zdim[2])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge2_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_2(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_3 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv4(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_5(xs)
    xs = F.leaky_relu(xs, 0.05)

    xs = generator.final_node_layer0(xs)
    if generator.use_x_bn:
        xs = generator.xbn_6(xs)
    xf = F.leaky_relu(xs, 0.05)
    xs = generator.final_node_layer1(xf)

    if generator.last_act == 'leaky':
        xs = F.leaky_relu(xs, 0.05)
    elif generator.last_act == 'tanh':
        xs = F.tanh(xs)
    elif generator.last_act == 'sigmoid':
        xs = F.sigmoid(xs)
    else:
        xs = F.leaky_relu(xs, 0.05)

    xs = F.softmax(xs, dim=1) + 1e-4

    edge_attr = generator.final_edge_layer0(torch.cat([edge_attr, generator.edge4_attr_layer(xs, edge_index), F.leaky_relu(xf[edge_index[0]] + xf[edge_index[1]], 0.05)], dim=1))
    if generator.use_e_bn:
        edge_attr = generator.ebn_3(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, 0.05)
    edge_attr = generator.final_edge_layer1(edge_attr)
    if generator.last_act == 'leaky':
        edge_attr = F.leaky_relu(edge_attr, 0.05)
    elif generator.last_act == 'tanh':
        edge_attr = F.tanh(edge_attr)
    elif generator.last_act == 'sigmoid':
        edge_attr = F.sigmoid(edge_attr)
    else:
        edge_attr = F.leaky_relu(edge_attr, 0.05)
    edge_attr = F.softmax(edge_attr, dim=1) + 1e-4

    if xs.isinf().sum() > 0:
        print ("Possible error....")
        xs[xs.isinf()] = torch.sign(xs[xs.isinf()]) * 3.318e38

    xs = F.gumbel_softmax(torch.log(xs), dim=1, hard=True, tau=1.0)
    edge_attr = F.gumbel_softmax(torch.log(edge_attr), dim=1, hard=True, tau=1.0)

    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_4 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)
    generator.train()
    return fake_data_1, fake_data_2, fake_data_3, fake_data_4




def sequential_generation_va(generator, z, gumbel_sample=True, tau=1.0, hard=True):
    """Given trained generator and input random vector z. This function is used for QM9's generator. 
    generate generated data after initial layer, each unpooling layer and final output.
    
    Return
        fakedata1-4 for results after initial layer, 1st UL, 2nd UL and final output.
    """
    generator.eval()
    generator.device = generator.final_edge_layer1.weight.device
    n = z.size(0)


    n = z.size(0)
    xs = generator.in_layer1(z)
    if generator.use_x_bn:
        xs = generator.inbn1(xs)
    xs = F.leaky_relu(xs, generator.relu_coef)
    xs = generator.in_layer2(xs)
    ########## Possibly, we don't need to run BN and Directly use Relu.
    xs = xs.view(3*n, generator.node_hidden_dim[0])
    if generator.use_x_bn:
        xs = generator.inbn2(xs)
    ################################################################# 

    xs = F.leaky_relu(xs, generator.relu_coef)
    xs = xs.view(n, 3, generator.node_hidden_dim[0])
    a1 = generator.first_edge_link(xs[:, 0, :], xs[:, 1, :])
    a2 = generator.first_edge_link(xs[:, 1, :], xs[:, 2, :])
    a3 = generator.first_edge_link(xs[:, 0, :], xs[:, 2, :])
    n1 = torch.arange(n).to(generator.device)*3
    n2 = torch.arange(n).to(generator.device)*3 + 1
    n3 = torch.arange(n).to(generator.device)*3 + 2
    edge_prob = torch.zeros(n, 4).to(generator.device)
    edge_prob[:, 0] = (a1[:, 0] + a2[:, 1] + a3[:, 1])/3
    edge_prob[:, 1] = (a2[:, 0] + a1[:, 1] + a3[:, 1])/3
    edge_prob[:, 2] = (a3[:, 0] + a2[:, 1] + a1[:, 1])/3
    edge_prob[:, 3] = (a1[:, 1] + a2[:, 1] + a3[:, 1])/3
    edge_prob = F.softmax(edge_prob, dim=1) + 1e-4
    m = Categorical(edge_prob)
    edge0_links = m.sample().to(generator.device)

    prob_init_edge = m.log_prob(edge0_links).to(generator.device)
    # 0: no 0,1 link
    # 1: no 1,2 link
    # 2: no 0,2 link
    # 3: all links
    en0 = (edge0_links == 0)
    en1 = (edge0_links == 1)
    en2 = (edge0_links == 2)
    en3 = (edge0_links == 3)
    edge_index = torch.LongTensor(2, 2*(en0.sum() + en1.sum() + en2.sum()) + 3*en3.sum()).to(generator.device)
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


    xs = xs.view(3*n, generator.node_hidden_dim[0])
    # if generator.use_x_bn:
    #     xs = generator.xbn_0(xs)
    # xs = F.leaky_relu(xs, 0.1)
    edge_attr = generator.edge0_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_0(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, generator.relu_coef)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)

    # edge_index, edge_attr = edge_index.to(generator.device), edge_attr.to(generator.device)
    batch = torch.arange(n).view(-1, 1).repeat(1, 3).view(-1)

    # print (edge_index, batch)

    edge_batch = torch.zeros(edge_index.size(1)).type(torch.LongTensor)
    edge_batch.scatter_(0, torch.arange(edge_index.size(1)), batch[edge_index[0]])
    # print (xs, edge_index, edge_attr)
    xs = generator.conv1(xs, edge_index=edge_index, edge_attr=edge_attr)

    if generator.use_x_bn:
        xs = generator.xbn_1(xs)
    xs = F.leaky_relu(xs, generator.relu_coef)

    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_1 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)


    if generator.unpool_type == 'normal':
        xs, batch, edge_index, edge_batch, prob_edge1 = generator.unpool1(xs, edge_index, batch)
    else:
        xs, batch, edge_index, edge_batch, prob_edge1 = generator.unpool1(xs, edge_index, edge_attr, batch)

    # if generator.use_x_bn:
    #     xs = generator.xbn_2(xs)
    # xs = F.leaky_relu(xs, generator.relu_coef)

    if generator.skip_z:
        add_xs = generator.skip_1(z)
        add_xs = add_xs.view(add_xs.size(0), 5, generator.skip_zdim[1])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge1_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_1(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, generator.relu_coef)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_2 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv2(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_3(xs)
    xs = F.leaky_relu(xs, generator.relu_coef)
    # xs = generator.conv3(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.unpool_type == 'normal':
        xs, batch, edge_index, edge_batch, prob_edge2 = generator.unpool2(xs, edge_index, batch)
    else:
        xs, batch, edge_index, edge_batch, prob_edge2 = generator.unpool2(xs, edge_index, edge_attr, batch)
    # if generator.use_x_bn:
    #     xs = generator.xbn_4(xs)
    # xs = F.leaky_relu(xs, generator.relu_coef)

    if generator.skip_z:
        add_xs = generator.skip_2(z)
        add_xs = add_xs.view(add_xs.size(0), 9, generator.skip_zdim[2])
        add_xs = assemble_skip_z(add_xs, batch, generator.device)
        xs = torch.cat([xs, add_xs], axis=1)

    edge_attr = generator.edge2_attr_layer(xs, edge_index)
    if generator.use_e_bn:
        edge_attr = generator.ebn_2(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, generator.relu_coef)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_3 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)

    xs = generator.conv4(xs, edge_index=edge_index, edge_attr=edge_attr)
    if generator.use_x_bn:
        xs = generator.xbn_5(xs)
    xs = F.leaky_relu(xs, generator.relu_coef)

    #########################################################
    # TEMP change: now reduce the number of layers.

    # edge_attr = generator.edge3_attr_layer(xs, edge_index)
    # if generator.use_e_bn:
    #     edge_attr = generator.ebn_3(edge_attr)
    # edge_attr = F.leaky_relu(edge_attr, generator.relu_coef)

    # xs = generator.conv5(xs, edge_index=edge_index, edge_attr=edge_attr)
    # if generator.use_x_bn:
    #     xs = generator.xbn_6(xs)
    # xs = F.leaky_relu(xs, generator.relu_coef)
    #########################################################

    xs = generator.final_node_layer0(xs)
    if generator.use_x_bn:
        xs = generator.xbn_6(xs)
    xf = F.leaky_relu(xs, generator.relu_coef)
    xs = generator.final_node_layer1(xf)

    if generator.last_act == 'leaky':
        xs = F.leaky_relu(xs, generator.relu_coef)
    elif generator.last_act == 'tanh':
        xs = F.tanh(xs)
    elif generator.last_act == 'sigmoid':
        xs = F.sigmoid(xs)
    else:
        xs = F.leaky_relu(xs, generator.relu_coef)

    # xs = F.softmax(xs, dim=1) + 1e-4
    x = torch.zeros_like(xs)
    x[:, :4] = F.softmax(xs[:, :4], dim=1) + 1e-5
    x[:, 4:7] = F.softmax(xs[:, 4:7], dim=1) + 1e-5
    x[:, 7:10] = F.softmax(xs[:, 7:10], dim=1) + 1e-5
    if hasattr(generator, 'without_ar') and generator.without_ar:
        pass
    else:
        x[:, 10:12] = F.softmax(xs[:, 10:12], dim=1) + 1e-5

    ## TODO: which xs should be used here?
    ## TODO: add global here as well?
    edge_attr = generator.final_edge_layer0(torch.cat([edge_attr, \
        generator.edge4_attr_layer(xs, edge_index), F.leaky_relu(xf[edge_index[0]] + \
            xf[edge_index[1]], generator.relu_coef)], dim=1))
    if generator.use_e_bn:
        edge_attr = generator.ebn_3(edge_attr)
    edge_attr = F.leaky_relu(edge_attr, generator.relu_coef)
    edge_attr = generator.final_edge_layer1(edge_attr)
    if generator.last_act == 'leaky':
        edge_attr = F.leaky_relu(edge_attr, generator.relu_coef)
    elif generator.last_act == 'tanh':
        edge_attr = F.tanh(edge_attr)
    elif generator.last_act == 'sigmoid':
        edge_attr = F.sigmoid(edge_attr)
    else:
        edge_attr = F.leaky_relu(edge_attr, generator.relu_coef)
    edge_attr = F.softmax(edge_attr, dim=1) + 1e-4

    if xs.isinf().sum() > 0:
        print ("Possible error....")
        xs[xs.isinf()] = torch.sign(xs[xs.isinf()]) * 3.318e38
    # if gumbel_sample:
    #     xs = F.gumbel_softmax(torch.log(xs), dim=1, hard=hard, tau=tau)
    #     edge_attr = F.gumbel_softmax(torch.log(edge_attr), dim=1, hard=hard, tau=tau)
    if gumbel_sample:
        xs1 = F.gumbel_softmax(torch.log(x[:, :4]), dim=1, hard=hard, tau=tau)
        xs2 = F.gumbel_softmax(torch.log(x[:, 4:7]), dim=1, hard=hard, tau=tau)
        xs3 = F.gumbel_softmax(torch.log(x[:, 7:10]), dim=1, hard=hard, tau=tau)
        if hasattr(generator, 'without_ar') and generator.without_ar:
            pass
        else:
            xs4 = F.gumbel_softmax(torch.log(x[:, 10:12]), dim=1, hard=hard, tau=tau)
        edge_attr = F.gumbel_softmax(torch.log(edge_attr), dim=1, hard=hard, tau=tau)
    if hasattr(generator, 'without_ar') and generator.without_ar:
        xs = torch.cat([xs1, xs2, xs3], axis=1)
    else:
        xs = torch.cat([xs1, xs2, xs3, xs4[:, :1]], axis=1)
    fake_data = convert_Batch_to_datalist(x=xs, edge_index=edge_index, edge_attr=edge_attr, batch=batch, edge_batch=edge_batch)
    fake_data_4 = Batch.from_data_list(\
        [Data(x=j.x, edge_index=j.edge_index, edge_attr=j.edge_attr) \
                for j in fake_data], \
                follow_batch=['edge_index']).to(generator.device)
    generator.train()
    return fake_data_1, fake_data_2, fake_data_3, fake_data_4



