import networkx as nx
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time>1:
            y_result = torch.rand(y.size(0),y.size(1),y.size(2))
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = torch.rand(y.size(1), y.size(2))
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
        else:
            y_thresh = torch.rand(y.size(0),y.size(1),y.size(2))
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = torch.ones(y.size(0), y.size(1), y.size(2))*thresh
        y_result = torch.gt(y, y_thresh).float()
    return y_result



def encode_adj(adj, max_prev_node=10, is_full = False):
    '''
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full



class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw
    
class GraphRNN(nn.Module):
    def __init__(self, node_dim, 
                 embedding_size_edge,
                 embedding_size_graph,  
                 hidden_size_edge, 
                 hidden_size_graph, 
                 num_layers_edge, 
                 num_layers_graph, 
                 max_prev_node, 
                 max_node_num
                 ):
        super(GraphRNN, self).__init__()
        self.node_dim = node_dim
        self.num_layers_edge = num_layers_edge
        self.edge_RNN = GRU_plain(1, embedding_size_edge, hidden_size_edge, num_layers_edge, has_input=True, has_output=True, output_size=1)
        self.graph_RNN = GRU_plain(node_dim + max_prev_node, embedding_size_graph, hidden_size_graph, num_layers_graph, has_input=True, has_output=True, output_size=(node_dim + hidden_size_edge))
        self.n = max_node_num
        self.max_prev_node = max_prev_node


    def forward(self, data):
        x_edge_unsorted = data['x_edge'].float()
        y_edge_unsorted = data['y_edge'].float()
        x_graph_unsorted = data['x_graph'].float()
        y_graph_unsorted = data['y_graph'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_edge_unsorted = x_edge_unsorted[:, 0:y_len_max, :]
        y_edge_unsorted = y_edge_unsorted[:, 0:y_len_max, :]
        x_graph_unsorted = x_graph_unsorted[:, :y_len_max, :]
        y_graph_unsorted = y_graph_unsorted[:, :y_len_max, :]
        self.graph_RNN.hidden = self.graph_RNN.init_hidden(batch_size=x_graph_unsorted.size(0))

        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x_edge = torch.index_select(x_edge_unsorted,0,sort_index)
        y_edge = torch.index_select(y_edge_unsorted,0,sort_index)
        x_graph_origin = torch.index_select(x_graph_unsorted,0,sort_index)
        y_graph = torch.index_select(y_graph_unsorted,0,sort_index)

        x_graph = torch.cat([x_edge, x_graph_origin], axis=2)

        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y_edge, y_len, batch_first=True).data

        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)      
        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)

        output_y = y_reshape  
        
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y_edge.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        # x = Variable(x).cuda()
        # y = Variable(y).cuda()
        # output_x = Variable(output_x).cuda()
        # output_y = Variable(output_y).cuda()
                
        h = self.graph_RNN(x_graph, pack=True, input_len=y_len)
        node_y_pred = h[:, :, :self.node_dim]
        h = pack_padded_sequence(h[:, :, self.node_dim:], y_len, batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        h = h.index_select(0, idx)
        hidden_null = torch.zeros(self.num_layers_edge - 1, h.size(0), h.size(1))
        self.edge_RNN.hidden = torch.cat((h.view(1,h.size(0),h.size(1)), hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = self.edge_RNN(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = F.binary_cross_entropy(y_pred, output_y)
        loss_node = torch.pow(y_graph - node_y_pred, 2).sum(axis=(1, 2)).mean()
        return loss, loss_node
    
    def generate(self, num_nodes, batch_size=32, sample_data = None):
        _ = self.eval()
        G_pred_list = []
        for j in range(num_nodes // batch_size + 1):
            self.graph_RNN.hidden = self.graph_RNN.init_hidden(batch_size)
            y_pred_long = torch.zeros(batch_size, self.n, self.max_prev_node)
            node_pred_feature = torch.zeros(batch_size, self.n, self.node_dim)
            if sample_data is None:
                x_step = torch.cat([torch.ones(batch_size,1,self.max_prev_node), torch.rand(batch_size, 1, self.node_dim)], axis=2)
            else:
                x_step = torch.cat([torch.ones(batch_size,1,self.max_prev_node), sample_data[torch.randint(0, sample_data.size(0), (batch_size, ))].view(batch_size, 1, self.node_dim)], axis=2)
            node_pred_feature[:, :1, :] = x_step[:, :, -self.node_dim:]
            for i in range(self.n):
                h = self.graph_RNN(x_step)
                # output.hidden = h.permute(1,0,2)
                hidden_null = torch.zeros(self.num_layers_edge - 1, h.size(0), h.size(2) - self.node_dim)
                self.edge_RNN.hidden = torch.cat((h.permute(1,0,2)[:, :, self.node_dim:], hidden_null),
                                        dim=0)  # num_layers, batch_size, hidden_size
                x_step = torch.zeros(batch_size, 1, self.max_prev_node + self.node_dim)
                output_x_step = torch.ones(batch_size,1,1)
                for j in range(min(self.max_prev_node,i+1)):
                    output_y_pred_step = self.edge_RNN(output_x_step)
                    output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
                    x_step[:,:,j:j+1] = output_x_step
                    self.edge_RNN.hidden = self.edge_RNN.hidden.data
                y_pred_long[:, i:i + 1, :] = x_step[:, :, :-self.node_dim]
                self.graph_RNN.hidden = self.graph_RNN.hidden
                if i < self.n - 1:
                    node_pred_feature[:, i+1:i+2, :] = h[:, :, :self.node_dim]
                x_step[:, :, -self.node_dim:] = h[:, :, :self.node_dim]
            y_pred_long_data = y_pred_long.data.long()
            for i in range(batch_size):
                tests = y_pred_long_data[i].numpy()
                tests[-1, :] = 0
                adj_pred = decode_adj(y_pred_long_data[i].numpy())
                edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_pred)) # get a graph from zero-padded adj
                use_uid = np.unique(edge_index.numpy())
                G_pred_list.append(Data(x=node_pred_feature[i, use_uid, :], edge_index = edge_index))
        _ = self.train()
        return G_pred_list


class GraphSampler(torch.utils.data.Dataset):
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
        adj_copy = self.adj_all[idx].clone()
        x_copy = self.x_all[idx].clone()
        len_batch = len(adj_copy)
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        g = self.g_all[idx]
        j = random.randint(0, self.len_all[idx] - 1)
        x_idx = np.array(nx.bfs_tree(g, j).nodes)

        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        x_copy = x_copy[x_idx]

        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        x_s = np.zeros((self.n, self.node_dim))
        y_s = np.zeros((self.n, self.node_dim))
        x_s[:x_copy.shape[0], :] = x_copy[:x_copy.shape[0]]
        y_s[:x_copy.shape[0]-1, :] = x_copy[1:x_copy.shape[0]]
        return {'x_edge':x_batch,'y_edge':y_batch,  # This need further packed during the training process.
                'x_graph':x_s, 'y_graph':y_s,  
                'len':len_batch}


def generate_samples(all_g):
    # Just return x_list and adj_list
    x_list = []
    adj_list = []
    g_list = []
    for g in all_g:
        # First build graph
        adj = to_dense_adj(g.edge_index)
        x_list.append(g.x)
        adj_list.append(adj[0]) 
        h = nx.Graph()
        h.add_nodes_from(np.unique((g.edge_index[:, 0]).numpy()))
        h.add_edges_from(g.edge_index.numpy().T)
        g_list.append(h)
    return x_list, adj_list, g_list