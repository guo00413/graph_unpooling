import numpy as np
import os
import scipy.optimize
import torch
import torch.nn as nn
from torch import long, optim, relu_
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphConv, global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing, GINConv, unpool
from util_gnn import generate_noise
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from torch_geometric.utils import dense_to_sparse

class ADJGANSampler(torch.utils.data.Dataset):
    def __init__(self, x_list, a_list, g_list):
        self.adj_all = a_list
        self.x_all = x_list
        self.g_all = g_list

    def __len__(self):
        return len(self.x_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].clone()
        x_copy = self.x_all[idx].clone()
        return {'x':x_copy,'adj':adj_copy,  # This need further packed during the training process.
                }

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        # self.relu = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight)
        return y

class GraphConv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight_s = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        # self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.FloatTensor(1, output_dim))
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight) + torch.matmul(x, self.weight_s) + self.bias
        return y

class Discriminator(nn.Module):
    def __init__(self, convs=[256, 256], in_dim=2, final_layers=[256, 256], use_bias=False):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.convs = []
        pre_dim = self.in_dim
        for j in convs:
            if use_bias:
                self.convs.append(GraphConv2(pre_dim, j))
            else:
                self.convs.append(GraphConv(pre_dim, j))
            pre_dim = j
        self.convs = nn.ModuleList(self.convs)
        self.final_layers = []
        self.sig_ln = nn.Linear(pre_dim, pre_dim)
        self.tanh_ln = nn.Linear(pre_dim, pre_dim)
        pre_dim = pre_dim + self.in_dim * 2 + 2
        for j in final_layers:
            self.final_layers.append(nn.Linear(pre_dim, j))
            self.final_layers.append(nn.LeakyReLU(0.01))
            pre_dim = j
        self.final_layers.append(nn.Linear(pre_dim, 1))
        self.final_layers.append(nn.Sigmoid())
        self.final_layers = nn.Sequential(*self.final_layers)
        for conv in self.convs:
            if hasattr(conv, 'bias'):
                nn.init.constant(conv.bias, 0.0)
            if hasattr(conv, 'weight'):
                nn.init.xavier_uniform(conv.weight,gain=nn.init.calculate_gain('relu'))
            if hasattr(conv, 'weight_s'):
                nn.init.xavier_uniform(conv.weight_s,gain=nn.init.calculate_gain('relu'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        

    def forward(self, x0, adj):
        # adj is N * n * n
        # x is N * n * p
        x = x0
        for j in self.convs:
            x = j(x, adj)
            x = F.leaky_relu(x, 0.01)
        check_x = x.sum(axis=1)
        sx = torch.clip(self.sig_ln(check_x), -30, 30)
        x = 1/(1+torch.exp(sx))*torch.tanh(self.tanh_ln(check_x))
        
        x = torch.cat([x, x0.sum(axis=1), x0.mean(axis=1), adj.sum(axis=(1, 2)).view(x0.size(0), 1), adj.mean(axis=(1, 2)).view(x0.size(0), 1)], axis=1)
        return self.final_layers(x)



class Generator(nn.Module):
    def __init__(self, in_dim=128, out_dim=2, max_node_num=12, final_layers=[256, 256, 256], full_map=False):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.final_layers = []
        pre_dim = in_dim
        for j in final_layers:
            self.final_layers.append(nn.Linear(pre_dim, j))
            self.final_layers.append(nn.LeakyReLU(0.01))
            pre_dim = j
        self.final_layers = nn.Sequential(*self.final_layers)
        self.max_node_num = max_node_num
        self.node_output = nn.Linear(pre_dim, max_node_num*out_dim)
        self.edge_output = nn.Linear(pre_dim, max_node_num*(max_node_num+1))
        self.full_map = full_map

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.final_layers(x)
        nodes = self.node_output(x)
        adj = self.edge_output(x)
        nodes = nodes.view(x.size(0), self.max_node_num, self.out_dim)
        adj = F.softmax(adj.view(x.size(0), self.max_node_num*(self.max_node_num + 1)//2, 2), dim=2)
        use_A = F.gumbel_softmax(torch.log(adj + 1e-7), tau=1, hard=True)
        use_adj = torch.zeros(x.size(0), self.max_node_num, self.max_node_num)
        use_adj[:, torch.triu(torch.ones(self.max_node_num, self.max_node_num)) == 1] = use_A[:, :, 0]
        use_adj = use_adj + use_adj.transpose(1, 2)
        if hasattr(self, 'full_map') and self.full_map:
            pass
        else:
            nodes = nodes * use_adj[:, torch.arange(self.max_node_num), torch.arange(self.max_node_num)].view(nodes.size(0), nodes.size(1), 1)
        use_adj[:, torch.arange(self.max_node_num), torch.arange(self.max_node_num)] = 0
        return nodes, use_adj
    
    
    def generate(self, z):
        nodes, use_adj = self(z)
        data = []
        for j in range(nodes.size(0)):
            use_xids = (nodes[j].sum(axis=1) != 0)
            usex = nodes[j][use_xids]
            use_adj[j, torch.arange(self.max_node_num), torch.arange(self.max_node_num)] = 0
            edge_adj = use_adj[j][use_xids, :][:, use_xids]
            use_edge, _ = dense_to_sparse(edge_adj)
            data.append(Data(x=usex, edge_index=use_edge))
        return data

class GANTrainerComb(object):
    def __init__(self, d, g, rand_dim=128, train_folder='adjGAN_0925',  
        batch_size=64, \
        learning_rate_g=1e-3, learning_rate_d=1e-4, \
        max_train_G=2, tresh_add_trainG=0.2, \
        use_loss='bce', \
        ):
        self.batch_size = batch_size
        gcn_model = d
        generator = g
        lr_d = learning_rate_d
        lr_g = learning_rate_g
        beta_d = (0.5, 0.999)
        beta_g = (0.5, 0.999)
        optimD = optim.Adam([{'params': gcn_model.parameters()},
                            ], \
                            lr=lr_d, betas=beta_d)
        self.optimD = optimD

        optimG = optim.Adam([{'params': generator.parameters()}, \
        #                      {'params': q_output.parameters()}\
                            ], lr=lr_g, betas=beta_g)
        self.optimG = optimG
        self.rand_dim = rand_dim

        self.gcn_model = gcn_model
        self.generator = generator

        self.folder = train_folder

        self.batch_size = batch_size
        self.max_train_G = max_train_G
        self.tresh_add_trainG = tresh_add_trainG
        self.use_loss = use_loss

        self.criterionD = nn.BCELoss(reduction='none')


    def train(self, data_loader, iteration, verbose=False, \
        save_num=200):
        add_train_g = 1
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        ii = 0
        import time
        real_label = 1.0
        fake_label = 0.0
        for epoch in range(iteration):
            for i, data in enumerate(data_loader):
                x = data['x']
                adj = data['adj']
                if len(x) < self.batch_size:
                    print ("small batch size...")
                    continue
                self.optimD.zero_grad()
                label = torch.full((self.batch_size*2, ), real_label)
                label[:self.batch_size] = real_label
                label[self.batch_size:] = fake_label
                z, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=self.rand_dim, batch=self.batch_size)
                fake_x, fake_adj = self.generator(z)
                if ii % 100 == 0:
                    print (f"fake_x mean: {fake_x.mean().item()}, std: {fake_x.std().item()}; adj mean:{fake_adj.mean().item()}")
                all_x = torch.cat([x, fake_x], axis=0)
                all_adj = torch.cat([adj, fake_adj], axis=0)
                probs = self.gcn_model(all_x, all_adj)
                loss = self.criterionD(probs.view(-1), label)

                loss_real = loss[:self.batch_size].mean()
                loss_fake = loss[self.batch_size:].mean()
                loss = loss.mean()
                prob_loss_real = probs.view(-1)[:self.batch_size].mean()
                prob_loss_fake = probs.view(-1)[self.batch_size:].mean()
                loss.backward()
                if prob_loss_real.item() > prob_loss_fake.item() + self.tresh_add_trainG:
                    add_train_g = min(1+add_train_g, self.max_train_G)
                else:
                    add_train_g = 1
                self.optimD.step()
                use_add_train_g = add_train_g

                for l in range(use_add_train_g):
                    self.optimG.zero_grad()
                    z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=self.rand_dim, batch=self.batch_size)
                    z = z_rand
                    fake_x, fake_adj = self.generator(z)

                    probs = self.gcn_model(fake_x, fake_adj)

                    label.fill_(real_label)
                    gen_loss = self.criterionD(probs.view(-1), label[self.batch_size:]).mean()
                    gen_loss.backward()
                    self.optimG.step()
                if ii % 100 == 0:
                    print ('now, we train G %d times with (prob fake = %.3f, prob real = %.3f)' % (add_train_g, \
                                                                                                    prob_loss_fake.mean(), \
                                                                                                    prob_loss_real.mean()))
                    if ii % save_num == 0:
                        torch.save(self.generator, os.path.join(self.folder, f'generator_{ii}.pt'))
                        torch.save(self.gcn_model, os.path.join(self.folder, f'gcn_model_{ii}.pt'))
                ii += 1

