import networkx as nx
from torch.distributions import Categorical
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, global_add_pool


class GraphNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, layer=3, global_out=False, final_hidden=64):
        super(GraphNN, self).__init__()
        self.convs = nn.ModuleList([GraphConv(in_dim, hidden_dim)])
        for j in range(layer - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim)) 
        self.convs.append(GraphConv(hidden_dim, out_dim))
        self.global_out = global_out
        if global_out:
            self.final_layers = nn.Sequential(nn.Linear(out_dim, final_hidden), nn.LeakyReLU(0.01), nn.Linear(final_hidden, 1), nn.Sigmoid())
        
    def forward(self, x, edge_index, batch):
        for j in self.convs:
            x = j(x=x, edge_index=edge_index)
            x = F.leaky_relu(x, 0.01)
        if self.global_out:
            x = global_add_pool(x, batch=batch)
            x = self.final_layers(x)
        return x

class GraphGCPN(nn.Module):
    # Include 3 policy networks
    # Include a discrimininator
    # Include generate molecule from beginning.
    # Include interaction and produce the Loss that we used to optimize.
    # i.e. min(clip(p/p.detach(), 1-epsilon, 1+epsilon) A
    def __init__(self, max_node_num=12, concate_dim=32, node_gnn_para={}, hidden_dim=64):
        super(GraphGCPN, self).__init__()
        self.max_node_num = max_node_num
        self.node_gnn = GraphNN(1, concate_dim, **node_gnn_para)
        self.first_prob = nn.Sequential(nn.Linear(concate_dim, hidden_dim), nn.LeakyReLU(0.01), 
                                        nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.second_prob = nn.Sequential(nn.Linear(concate_dim*2, hidden_dim), nn.LeakyReLU(0.01), 
                                        nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.terminate_prob = nn.Sequential(nn.Linear(concate_dim, hidden_dim), nn.LeakyReLU(0.01), 
                                        nn.Linear(hidden_dim, 2), nn.Sigmoid())
        
        self.discriminator = GraphNN(1, 64, 64, 3, True, 64)
        self.concate_dim = concate_dim


    def generate(self, num_graphs, old_model=None):
        results = []
        for j in range(num_graphs):
            graph_prob_pairs = []
            
            initial_graph_x = torch.ones(3, 1)
            initial_graph_edges = torch.zeros(2, 2,dtype=int)
            initial_graph_edges[0, 0] = 1
            initial_graph_edges[1, 1] = 1
            batch = torch.zeros(3, dtype=int)
            for k in range(self.max_node_num*(self.max_node_num-1)//2):
                # At maximum, we can take this number of steps.
                embeded = self.node_gnn(initial_graph_x, initial_graph_edges, batch)
                first_prob = self.first_prob(embeded[:-1, :])
                m = Categorical(first_prob.view(-1))
                first_node = m.sample()
                prob1 = m.probs[first_node]
                embeded2 = torch.cat([embeded, embeded[first_node, :].view(1, self.concate_dim).repeat(embeded.size(0), 1)], axis=1)
                second_prob = self.second_prob(embeded2)
                m = Categorical(second_prob.view(-1))
                second_node = m.sample()
                prob2 = m.probs[second_node]
                terminal_prob = self.terminate_prob(embeded[:, :].sum(axis=0).view(1, -1))
                m = Categorical(terminal_prob.view(-1))
                terminate_node = m.sample()
                prob3 = m.probs[terminate_node]
                if old_model is not None: # find pi_old
                    embeded = old_model.node_gnn(initial_graph_x, initial_graph_edges, batch)
                    first_prob = old_model.first_prob(embeded[:-1, :])
                    m = Categorical(first_prob.view(-1))
                    prob1_old = m.probs[first_node]
                    embeded2 = torch.cat([embeded, embeded[first_node, :].view(1, self.concate_dim).repeat(embeded.size(0), 1)], axis=1)
                    second_prob = old_model.second_prob(embeded2)
                    m = Categorical(second_prob.view(-1))
                    prob2_old = m.probs[second_node]
                    terminal_prob = old_model.terminate_prob(embeded[:, :].sum(axis=0).view(1, -1))
                    m = Categorical(terminal_prob.view(-1))
                    prob3_old = m.probs[terminate_node]
                    old_probs =prob1_old*prob2_old*prob3_old
                    if second_node < initial_graph_x.size(0) - 1:
                        initial_graph_x = initial_graph_x[:-1, :]
                    initial_graph_edges = torch.cat([initial_graph_edges, torch.LongTensor([[first_node, second_node], [second_node, first_node]])], axis=1)

                    graph_prob_pairs.append((Data(x=initial_graph_x, edge_index=initial_graph_edges), prob1*prob2*prob3, old_probs.detach()))
                else:
                    if second_node < initial_graph_x.size(0) - 1:
                        initial_graph_x = initial_graph_x[:-1, :]
                    initial_graph_edges = torch.cat([initial_graph_edges, torch.LongTensor([[first_node, second_node], [second_node, first_node]])], axis=1)

                    graph_prob_pairs.append((Data(x=initial_graph_x, edge_index=initial_graph_edges), prob1*prob2*prob3))
                if terminate_node == 0:
                    break

                initial_graph_x = torch.cat([initial_graph_x, torch.ones(1, 1)], axis=0)
            results.append(graph_prob_pairs)
        return results

import copy
def training_process(gcpn, real_data, optimD, optimRL, batch_size=32, epoch=10, iteration=20, d_train_step = 10, device='cpu', epsilon=0.2):
    ii = 0
    for i in range(iteration):
        for k, data in enumerate(real_data):
            labels = torch.zeros(batch_size*2, 1).to(device)
            labels[:batch_size, 0] = 1
            data = data.to(device)
            probs_r = gcpn.discriminator(torch.ones(data.x.size(0), 1).to(device), data.edge_index, data.batch)
            fake_data = gcpn.generate(batch_size)
            fake_data = [j[-1][0] for j in fake_data]
            fake_data_batch = Batch.from_data_list(fake_data)
            probs_f = gcpn.discriminator(fake_data_batch.x, fake_data_batch.edge_index, fake_data_batch.batch)
            probs = torch.cat([probs_r, probs_f], axis=0)
            if k == d_train_step:
                print ('In D:', probs[:batch_size].mean().item(), probs[batch_size:].mean().item())
            loss = F.binary_cross_entropy(probs, labels)
            optimD.zero_grad()
            loss.backward()
            optimD.step()
            if k > d_train_step:
                break
        old_model = copy.deepcopy(gcpn)
        for j in range(epoch):
            optimRL.zero_grad()
            fake_data = gcpn.generate(batch_size, old_model=old_model)
            fake_data_list = [j[-1][0] for j in fake_data]
            fake_data_batch = Batch.from_data_list(fake_data_list)
            probs = gcpn.discriminator(fake_data_batch.x, fake_data_batch.edge_index, fake_data_batch.batch)
            labels = torch.ones(batch_size, 1).to(device)
            rewards = -F.binary_cross_entropy(probs, labels, reduce=False).detach()
            if j == epoch - 1:
                print (rewards.mean().item())
            temp_loss = 0
            for k, f_data in enumerate(fake_data):
                reward = rewards[k]
                for data, probs, probs_old in f_data:
                    temp_loss += (torch.clip(probs/(probs_old + 1e-7), 1-epsilon, 1+epsilon)*reward)
            (-temp_loss).backward()
            optimRL.step()
            ii += 1
