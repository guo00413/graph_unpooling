import networkx as nx
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphConv, global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing, GINConv, unpool
import os

# Need a GCN (think of which way to use, based on edge_index or based on A?)

class GraphAFSampler(torch.utils.data.Dataset):
    def __init__(self, x_list, a_list, g_list, node_dim=2, max_num_node=None, max_prev_node=None, iteration=20000, batch_size=64, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.adj_all = a_list
        self.len_all = [len(j) for j in x_list]
        self.x_all = x_list
        self.g_all = g_list
        self.n = max_num_node
        if max_prev_node is None:
            self.max_prev_node = self.n-1
        else:
            self.max_prev_node = max_prev_node
        self.node_dim = node_dim
    def __len__(self):
        return len(self.x_all)

    def __getitem__(self, idx):
        # Obtain two series of Data's, ids and targets (Z^X_i, Z^A_ij).
        # For [H_i, Z^X_i] for i in 1,...N
        # For [H_i, id_i, id_j, Z^A_ij] for i in 1...N, j in 1...i-1
        
        adj_copy = self.adj_all[idx].clone()
        x_copy = self.x_all[idx].clone()

        g = self.g_all[idx]
        j = random.randint(0, self.len_all[idx] - 1)
        x_idx = np.array(nx.bfs_tree(g, j).nodes)
        
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        x_copy = x_copy[x_idx]
        
        initial_data = Data(x = x_copy[:1, :], edge_index=torch.LongTensor([[], []]))
        data_x = []
        y_x = []
        y_A = []
        data_A = []
        id_A = []
        lens = []
        for j in range(1, len(x_copy)):
            data_x.append(initial_data)
            if j < len(x_copy) - 1:
                y_x.append(torch.FloatTensor([x_copy[j][k] for k in range(self.node_dim)] + [1 + torch.rand(1).item()]))
            else:
                y_x.append(torch.FloatTensor([0 for k in range(self.node_dim)] + [torch.rand(1).item()]))
            initial_data = Data(x=torch.cat([initial_data.x, x_copy[j:j+1, :]], axis=0), edge_index = initial_data.edge_index)
            for l in range(j):
                id_A.append((j, l))
                lens.append(j+1)
                data_A.append(initial_data)
                y_A.append(adj_copy[j, l] + torch.rand(1))
            # update initial data:
            useA = adj_copy[j, :j]
            total_As = useA.sum()
            new_edge_index = torch.cat([initial_data.edge_index, 
                                        torch.LongTensor([torch.where(useA > 0)[0].tolist(), [j for l in range(int(total_As.item()))]]), 
                                        torch.LongTensor([[j for l in range(int(total_As.item()))], torch.where(useA > 0)[0].tolist()])], axis=1)
            initial_data = Data(x=initial_data.x, edge_index = new_edge_index)
        
        return {'data_x':data_x,
                'y_x': y_x,
                'y_A': y_A,
                'data_A': data_A, 
                'id_A': id_A, 
                'lens': lens}

    def sample(self, list_ids):
        dicts = [self[j] for j in list_ids]
        add_ids = np.array(sum([j['lens'] for j in dicts], []))
        add_ids = np.cumsum(add_ids) - add_ids
        id_i = np.array(sum([[l[0] for l in j['id_A']] for j in dicts], []))
        id_j = np.array(sum([[l[1] for l in j['id_A']] for j in dicts], []))
        info = {'data_x': Batch.from_data_list(sum([j['data_x'] for j in dicts], [])), \
                'y_x':torch.cat(sum([[k.view(1, -1) for k in j['y_x']] for j in dicts], []), axis=0), 
                'data_A': Batch.from_data_list(sum([j['data_A'] for j in dicts], [])), 
                'y_A':torch.FloatTensor(sum([j['y_A'] for j in dicts], [])), 
                'idi_A':torch.LongTensor(id_i + add_ids),
                'idj_A':torch.LongTensor(id_j + add_ids),
                }
        return info 

class GraphAF(nn.Module):
    def __init__(self, in_dim=2, convs=[128, 256], node_layers=[256, 512], edge_layers=[256, 512], out_dim=None, e_out_dim=None,
                 layer_act='tanh', add_bn=False):
        super(GraphAF, self).__init__()
        self.in_dim = in_dim
        if out_dim is None:
            out_dim = in_dim
        self.graph_conv = nn.ModuleList([])
        pre_dim = in_dim
        for j in convs:
            self.graph_conv.append(GraphConv(pre_dim, j))
            self.graph_conv.append(nn.LeakyReLU(0.01))
            pre_dim = j
        graph_dim = pre_dim
        self.add_bn = add_bn
        if add_bn:
            self.bn = nn.BatchNorm1d(graph_dim)
        
        self.node_layers = []
        for j in node_layers:
            self.node_layers.append(nn.Linear(pre_dim, j))
            if layer_act == 'tanh':
                self.node_layers.append(nn.Tanh())
            else:
                self.node_layers.append(nn.LeakyReLU(0.05))
            pre_dim = j
        self.node_layers.append(nn.Linear(pre_dim, out_dim*2))
        self.node_layers = nn.Sequential(*self.node_layers)
        self.out_dim = out_dim
        
        pre_dim = graph_dim + graph_dim * 2
        self.edge_layers = []
        for j in edge_layers:
            self.edge_layers.append(nn.Linear(pre_dim, j))
            if layer_act == 'tanh':
                self.edge_layers.append(nn.Tanh())
            else:
                self.edge_layers.append(nn.LeakyReLU(0.05))
            pre_dim = j
        self.edge_layers.append(nn.Linear(pre_dim, e_out_dim*2))
        self.e_out_dim = e_out_dim
        self.edge_layers = nn.Sequential(*self.edge_layers)
    
    def forward(self, x, verbose=False):
        data_x = x['data_x']
        y_x = x['y_x']
        data_A = x['data_A']
        y_A = x['y_A'].view(-1, 1)
        idi_A = x['idi_A']
        idj_A = x['idj_A']
        g_x = data_x.x
        g_A = data_A.x
        for j in self.graph_conv:
            if isinstance(j, GraphConv):
                g_x = j(g_x, edge_index=data_x.edge_index)
                g_A = j(g_A, edge_index=data_A.edge_index)
            else:
                g_x = j(g_x)
                g_A = j(g_A)
        add_xi_A = g_A[idi_A, :]
        add_xj_A = g_A[idj_A, :]
        if self.add_bn:
            g_x = self.bn(g_x)
            g_A = self.bn(g_A)
        read_g_x = global_add_pool(g_x, batch=data_x.batch)
        read_g_A = global_add_pool(g_A, batch=data_A.batch)
        use_x_A = torch.cat([read_g_A, add_xi_A, add_xj_A], axis=1)
        y_pred_x = self.node_layers(read_g_x)
        y_pred_A = self.edge_layers(use_x_A)
        a_x = torch.pow(y_pred_x[:, :self.out_dim], 2) + 1e-7
        mu_x = y_pred_x[:, self.out_dim:]
        a_A = torch.pow(y_pred_A[:, :self.e_out_dim], 2) + 1e-7
        mu_A = y_pred_A[:, self.e_out_dim:]
        eps_x = (y_x - mu_x)/a_x
        eps_A = (y_A - mu_A)/a_A
        if verbose:
            print ("============ FOR X's =================")
            print (eps_x)
            print (a_x)
            print (torch.pow(eps_x, 2).sum())
            print (torch.log(a_x).sum())
            print ("============ FOR A's =================")
            print (eps_A)
            print (y_A.shape, mu_A.shape, a_A.shape)
            print (y_A, mu_A, a_A)
            print (a_A)
        loss_x = torch.pow(eps_x, 2) + torch.log(a_x)
        loss_A = torch.pow(eps_A, 2) + torch.log(a_A)
        return loss_x, loss_A
        
    def load_outx(self, g):
        g_x = g.x
        for j in self.graph_conv:
            if isinstance(j, GraphConv):
                g_x = j(g_x, edge_index=g.edge_index)
            else:
                g_x = j(g_x)
        if self.add_bn:
            g_x = self.bn(g_x)
        read_g_x = global_add_pool(g_x, batch=g.batch)
        y_pred_x = self.node_layers(read_g_x)
        a_x = torch.pow(y_pred_x[:, :self.out_dim], 2) + 1e-7
        mu_x = y_pred_x[:, self.out_dim:]
        return a_x, mu_x

    def load_outA(self, g, idi, idj):
        g_A = g.x
        for j in self.graph_conv:
            if isinstance(j, GraphConv):
                g_A = j(g_A, edge_index=g.edge_index)
            else:
                g_A = j(g_A)
        add_xi_A = g_A[idi, :]
        add_xj_A = g_A[idj, :]
        if self.add_bn:
            g_A = self.bn(g_A)
        read_g_A = global_add_pool(g_A, batch=g.batch)
        use_x_A = torch.cat([read_g_A, add_xi_A.view(1, -1), add_xj_A.view(1, -1)], axis=1)
        y_pred_A = self.edge_layers(use_x_A)
        a_A = torch.pow(y_pred_A[:, :self.e_out_dim], 2) + 1e-7
        mu_A = y_pred_A[:, self.e_out_dim:]
        return a_A, mu_A
        
    def generate(self, count, zs, max_node_num=12):
        # To build graph based on trained self.
        results = []
        for j in range(count):
            initial_data = Data(x=zs[j:j+1], edge_index=torch.LongTensor([[], []]))
            for k in range(max_node_num - 1):
                a_x, mu_x = self.load_outx(Batch.from_data_list([initial_data]))
                one_more_x = torch.rand(1, self.out_dim)*a_x + mu_x
                if one_more_x[0, -1].item() < 1:
                    break
                initial_data = Data(x=torch.cat([initial_data.x, one_more_x[:, :-1]], axis=0), edge_index=initial_data.edge_index)
                if k >= 1:
                    false_point = True
                else:
                    false_point = False
                
                for l in range(k):
                    a_A, mu_A = self.load_outA(Batch.from_data_list([initial_data]), k, l)
                    one_more_A = torch.rand(1, self.e_out_dim)*a_A + mu_A
                    if one_more_A[0, 0].item() < 1:
                        continue
                    false_point = False
                    initial_data = Data(x=initial_data.x, edge_index=torch.cat([initial_data.edge_index, torch.LongTensor([[k, l], [l, k]])], axis=1))
                if false_point:
                    initial_data = Data(x=initial_data.x[:-1, :], edge_index=initial_data.edge_index)
                    break
            results.append(initial_data)
        return results
        
class trainerAF():
    def __init__(self, model, lr=1e-4, folder=None, batch_size=16):
        self.model = model
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            if 'weight' in name:
                try:
                    nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('relu'))
                except:
                    nn.init.normal(param, 0.5)
                    
        self.optimD = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.folder = folder
        if folder is not None:
            if not os.path.exists(folder):
                os.mkdir(folder)
        self.cnt_train = 0
        self.batch_size = batch_size
        
        
    def train(self, dataloader, datasampler, epoch=3000, save_cnt=500, verbose = 100, vverbose=False):
        ii = 0
        for j in range(epoch):
            print (f"Epoch {j} ===============")
            for k in dataloader:
                use_x = datasampler.sample(k)
                ii += 1
                self.optimD.zero_grad()
                lossX, lossA = self.model(use_x, verbose = ((ii % save_cnt == 1) & vverbose))
                (lossX.sum()/self.batch_size).backward()
                (lossA.sum()/self.batch_size).backward()
                self.optimD.step()
                if ii % verbose == 1:
                    print (f"At {ii}, loss X is {lossX.sum().item()/self.batch_size}; loss A is {lossA.sum().item()/self.batch_size}.")
                if ii % save_cnt == 0 and self.folder is not None:
                    torch.save(self.model, os.path.join(self.folder, f"graphAF_{self.cnt_train}.pt"))
                    self.cnt_train += 1