import networkx as nx
import torch
from torch import FloatTensor, nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphConv, global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing, GINConv, unpool
import os

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_n = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight_s = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        # self.relu = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight_n) + torch.matmul(x, self.weight_s)
        return y

class SpatialGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, hidden_dims=[50, 50, 50]):
        super(SpatialGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        self.hidden_dims = hidden_dims
        self.h_dim = self.input_dim*3 + 3
        self.weight_1 = nn.Parameter(torch.FloatTensor(self.h_dim, hidden_dims[0]))
        self.bias_1 = nn.Parameter(torch.FloatTensor(1, hidden_dims[0]))
        self.lrelu_1 = nn.LeakyReLU(0.05)

        self.h_dim2 = self.input_dim*2 + hidden_dims[0] + 1
        self.weight_2 = nn.Parameter(torch.FloatTensor(self.h_dim2, hidden_dims[1]))
        self.bias_2 = nn.Parameter(torch.FloatTensor(1, hidden_dims[1]))
        self.lrelu_2 = nn.LeakyReLU(0.05)

        self.h_dim3 = self.input_dim + hidden_dims[1]
        self.weight_3 = nn.Parameter(torch.FloatTensor(self.h_dim3, output_dim))
        self.bias_3 = nn.Parameter(torch.FloatTensor(1, output_dim))
        self.lrelu_3 = nn.LeakyReLU(0.05)



    def forward(self, x, adj):
        batch_size = x.size(0)
        xi = x[:, :, None, :].repeat(1, 1, self.node_num, 1)
        xj = x[:, None, :, :].repeat(1, self.node_num, 1, 1)
        dis = torch.sqrt(torch.pow(xi - xj, 2).sum(axis=3) + 1e-10) # B*N*N

        rel_ij = dis[:, :, :, None, None].repeat(1,1,1,self.node_num,1)
        rel_jk = dis[:, None, :, :, None].repeat(1,self.node_num,1,1,1)
        rel_ik = dis[:, :, None, :, None].repeat(1,1,self.node_num,1,1)

        adj_rep1 = adj[:, :, :, None, None].repeat(1, 1, 1, self.node_num, 1)
        adj_rep2 = adj[:, None, :, :, None].repeat(1, self.node_num, 1, 1, 1)
        adj_3d = adj_rep1*adj_rep2

        x_x = x[:, :, None, None, :].repeat(1, 1, self.node_num, self.node_num, 1)
        x_y = x[:, None, :, None, :].repeat(1, self.node_num, 1, self.node_num, 1)
        x_z = x[:, None, None, :, :].repeat(1, self.node_num, self.node_num, 1, 1)
        m3 = torch.cat([x_x, x_y, x_z, rel_ij, rel_jk, rel_ik], dim=-1) #B*N*N*N*h'
        h_dim = self.input_dim*3 + 3
        m3 = m3.view(-1, h_dim)
        m3 = (torch.matmul(m3, self.weight_1) + self.bias_1).view(batch_size, self.node_num, self.node_num, self.node_num, self.hidden_dims[0])
        m3 = m3.transpose(3, 4) # B*N*N*h'*N
        m3_sum = torch.matmul(m3, adj_3d) # ->B, N, N, h', 1
        m3_sum = self.lrelu_1(m3_sum.view(batch_size, self.node_num, self.node_num, self.hidden_dims[0])) # B, N, N, h'


        x_x = x[:, :, None, :].repeat(1, 1, self.node_num, 1)
        x_y = x[:, None, :, :].repeat(1, self.node_num, 1, 1)
        
        m2 = torch.cat([x_x, x_y, dis[:, :, :, None], m3_sum], dim=-1) #B*N*N*N*h'
        h_dim = self.input_dim*2 + 1 + self.hidden_dims[0]
        m2 = m2.view(-1, h_dim)
        m2 = (torch.matmul(m2, self.weight_2) + self.bias_2).view(batch_size, self.node_num, self.node_num, self.hidden_dims[1])
        m2 = m2.transpose(2, 3) # B*N*h'*N
        m2_sum = torch.matmul(m2, adj[:, :, :, None]) # ->B, N, h', 1
        m2_sum = self.lrelu_2(m2_sum.view(batch_size, self.node_num, self.hidden_dims[1])) # 

        m1 = torch.cat([x, m2_sum], axis=2) # B, N, in_dim+hidden_dims[1]
        h_dim = self.input_dim + self.hidden_dims[1]
        m1 = m1.view(-1, h_dim)
        m1 = self.lrelu_3((torch.matmul(m1, self.weight_3) + self.bias_3)).view(batch_size, self.node_num, self.output_dim)

        return m1


class SpatialEncoder(nn.Module):
    def __init__(self,  in_dim, node_num, hidden_dim_spatial1=10, hidden_dim_spatial2=20, hidden_dim_linear=100):
        super(SpatialEncoder, self).__init__()
        self.in_dim = in_dim
        self.node_num = node_num
        self.hidden_dim_spatial1 = hidden_dim_spatial1
        self.hidden_dim_spatial2 = hidden_dim_spatial2
        self.part1 = nn.Sequential(nn.Conv1d(in_dim, hidden_dim_spatial1, 5, padding=2), 
                                             nn.ReLU(), 
                                             nn.Conv1d(hidden_dim_spatial1, hidden_dim_spatial1, 5, padding=2), 
                                             nn.ReLU(), 
                                             nn.Conv1d(hidden_dim_spatial1, hidden_dim_spatial2, 5, padding=2), 
                                             nn.ReLU(),
                                             ) 
        # Changed to (node_num) * hidden_dim_spatial2
        self.part2_init_dim = (node_num) * hidden_dim_spatial2
        self.hidden_dim_linear = hidden_dim_linear
        self.part2 = nn.Sequential(nn.Linear(self.part2_init_dim, hidden_dim_linear), nn.ReLU(), nn.Linear(hidden_dim_linear, hidden_dim_linear))
        self.output_dim = hidden_dim_linear
        
    def forward(self, x):
        x = self.part1(x.transpose(1, 2)).view(-1, self.part2_init_dim)
        return self.part2(x)


class JointEncoder(nn.Module):
    def __init__(self,  in_dim, node_num, hidden_dim_joint1=20, hidden_dim_joint2=50, hidden_dim_linear=200):
        super(JointEncoder, self).__init__()
        self.in_dim = in_dim
        self.node_num = node_num
        self.hidden_dim_joint1 = hidden_dim_joint1
        self.hidden_dim_joint2 = hidden_dim_joint2
        self.sm1 = SpatialGraphConv(in_dim, hidden_dim_joint1, node_num)
        self.relu1 = nn.LeakyReLU(0.05)
        self.sm2 = SpatialGraphConv(hidden_dim_joint1, hidden_dim_joint2, node_num)
        self.relu2 = nn.LeakyReLU(0.05)
        
        self.l1 = nn.Linear(hidden_dim_joint2, hidden_dim_linear)
        self.relu = nn.LeakyReLU(0.05)
        self.l2 = nn.Linear(hidden_dim_linear, hidden_dim_linear)
        self.output_dim = hidden_dim_linear
        
    def forward(self, x, adj):
        x = self.relu1(self.sm1(x, adj))
        x = self.relu2(self.sm2(x, adj))
        x = x.sum(axis=1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

class NetworkEncoder(nn.Module):
    def __init__(self,  in_dim, node_num, hidden_dim1=10, hidden_dim2=20, hidden_dim_linear=100):
        super(NetworkEncoder, self).__init__()
        self.in_dim = in_dim
        self.node_num = node_num
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.gcn1 = GraphConv(in_dim, hidden_dim1)
        self.gcn2 = GraphConv(hidden_dim1, hidden_dim2)
        self.l1 = nn.Linear(hidden_dim2, hidden_dim_linear)
        self.relu = nn.LeakyReLU(0.05)
        self.l2 = nn.Linear(hidden_dim_linear, hidden_dim_linear)
        self.output_dim = hidden_dim_linear
        
    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        x = x.sum(axis=1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class NetworkDecoder(nn.Module):
    def __init__(self, input_dim, node_num, hidden_dim=500, hidden_deconv_dims=[50, 20], hidden_layer_dim=50):
        super(NetworkDecoder, self).__init__()
        self.input_dim = input_dim
        self.node_num = node_num
        hidden_dim = hidden_dim // (node_num-6)//(node_num-6) * (node_num-6) * (node_num - 6)
        self.channel_dim = hidden_dim // (node_num-6)//(node_num-6)
        self.ln1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.LeakyReLU(0.05)
        self.deconvs = nn.Sequential(nn.ConvTranspose2d(self.channel_dim, hidden_deconv_dims[0], 5), 
                                    nn.LeakyReLU(0.05), 
                                    nn.ConvTranspose2d(hidden_deconv_dims[0], hidden_deconv_dims[1], 3), 
                                    nn.LeakyReLU(0.05), 
                                    )
        self.final_ln = nn.Sequential(
                                nn.Linear(hidden_deconv_dims[1], hidden_layer_dim),
                                nn.LeakyReLU(0.05), 
                                nn.Linear(hidden_layer_dim, 1),
                                nn.Sigmoid()
        )


    def forward(self, z):
        x = self.relu1(self.ln1(z)).view(z.size(0), self.channel_dim, (self.node_num-6), (self.node_num-6))
        x = self.deconvs(x).transpose(1, 3)
        return self.final_ln(x)

class SpatialDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, hidden_dim=500, hidden_conv_dims=[50, 20, 10]):
        super(SpatialDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        hidden_dim = hidden_dim //node_num * node_num
        channel_dim = hidden_dim //node_num
        self.channel_dim = channel_dim
        self.ln1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.LeakyReLU(0.05)
        self.convs = nn.Sequential(nn.Conv1d(channel_dim, hidden_conv_dims[0], 5, padding=2), 
                                   nn.LeakyReLU(0.05), 
                                   nn.Conv1d(hidden_conv_dims[0], hidden_conv_dims[1], 5, padding=2), 
                                   nn.LeakyReLU(0.05), 
                                   nn.Conv1d(hidden_conv_dims[1], hidden_conv_dims[2], 5, padding=2), 
                                   nn.LeakyReLU(0.05), 
                                   )
        self.final_layer = nn.Linear(hidden_conv_dims[2], output_dim)

    def forward(self, z):
        x = self.relu1(self.ln1(z)).view(z.size(0), self.channel_dim, self.node_num)
        return self.final_layer(self.convs(x).transpose(1, 2))

class SGDVAE(nn.Module):
    def __init__(self, in_dim, node_num, spatial_encoder_para={}, joint_encoder_para={}, network_encoder_para={}, 
                 network_decoder_para={}, spatial_decoder_para={}): 
        super(SGDVAE, self).__init__()
        self.in_dim = in_dim
        self.node_num = node_num
        self.spatial_encoder = SpatialEncoder(in_dim, node_num, **spatial_encoder_para)
        # self.zdim_s = self.spatial_encoder.hidden_dim_linear
        self.joint_encoder = JointEncoder(in_dim, node_num, **joint_encoder_para)
        self.network_encoder = NetworkEncoder(in_dim, node_num, **network_encoder_para)
        self.network_decoder = NetworkDecoder(self.network_encoder.output_dim + self.joint_encoder.output_dim, node_num, **network_decoder_para)
        self.spatial_decoder = SpatialDecoder(self.spatial_encoder.output_dim + self.joint_encoder.output_dim, in_dim, node_num, **spatial_decoder_para)
        self.linear_zs_mean = nn.Linear(self.spatial_encoder.output_dim, self.spatial_encoder.output_dim)
        self.linear_zs_std = nn.Linear(self.spatial_encoder.output_dim, self.spatial_encoder.output_dim)
        self.linear_zgs_mean = nn.Linear(self.joint_encoder.output_dim, self.joint_encoder.output_dim)
        self.linear_zgs_std = nn.Linear(self.joint_encoder.output_dim, self.joint_encoder.output_dim)
        self.linear_zg_mean = nn.Linear(self.network_encoder.output_dim, self.network_encoder.output_dim)
        self.linear_zg_std = nn.Linear(self.network_encoder.output_dim, self.network_encoder.output_dim)
        
    def get_z(self, z_mean_s, z_std_s, z_mean_sg, z_std_sg, z_mean_g, z_std_g):
        z_s = z_mean_s + torch.rand(z_mean_s.size(0), z_mean_s.size(1)) * torch.exp(torch.clip(z_std_s, -10, 10))
        z_sg = z_mean_sg + torch.rand(z_mean_sg.size(0), z_mean_sg.size(1)) * torch.exp(torch.clip(z_std_sg, -10, 10))
        z_g = z_mean_g + torch.rand(z_mean_g.size(0), z_mean_g.size(1)) * torch.exp(torch.clip(z_std_g, -10, 10))
        return z_s, z_sg, z_g
        
    def forward(self, x, adj):
        z_s = self.spatial_encoder(x)
        z_g = self.network_encoder(x, adj)
        z_sg = self.joint_encoder(x, adj)
        z_mean_s = self.linear_zs_mean(z_s)
        z_mean_sg = self.linear_zgs_mean(z_sg)
        z_mean_g = self.linear_zg_mean(z_g)
        z_std_s = self.linear_zs_std(z_s)
        z_std_sg = self.linear_zgs_std(z_sg)
        z_std_g = self.linear_zg_std(z_g)

        z_s, z_sg, z_g = self.get_z(z_mean_s, z_std_s, z_mean_sg, z_std_sg, z_mean_g, z_std_g)

        z_in_n = torch.cat([z_sg, z_g], axis=1)
        z_in_s = torch.cat([z_sg, z_s], axis=1)
        adj_out = self.network_decoder(z_in_n)
        x_out = self.spatial_decoder(z_in_s)
        loss_x = torch.sqrt(torch.pow(x - x_out, 2).sum(axis=(1, 2)))
        loss_adj = torch.nn.functional.binary_cross_entropy(adj_out, adj[:, :, :, None], reduction='none').sum(axis=(1, 2, 3))
        kl_s = -(0.5) * (1 + 2 * z_std_s - torch.square(z_mean_s) - torch.square(torch.exp(torch.clip(z_std_s, -10, 10)))).mean(axis=1) #actually kl
        kl_g = -(0.5) * (1 + 2 * z_std_g - torch.square(z_mean_g) - torch.square(torch.exp(torch.clip(z_std_g, -10, 10)))).mean(axis=1)
        kl_sg = -(0.5) * (1 + 2 * z_std_sg - torch.square(z_mean_sg) - torch.square(torch.exp(torch.clip(z_std_sg, -10, 10)))).mean(axis=1)
        loss_kl = kl_s + kl_g + kl_sg
        return loss_x, loss_adj, loss_kl
    
    
    def test_z(self, x, adj):
        z_s = self.spatial_encoder(x)
        z_g = self.network_encoder(x, adj)
        z_sg = self.joint_encoder(x, adj)
        z_mean_s = self.linear_zs_mean(z_s)
        z_mean_sg = self.linear_zgs_mean(z_sg)
        z_mean_g = self.linear_zg_mean(z_g)
        z_std_s = self.linear_zs_std(z_s)
        z_std_sg = self.linear_zgs_std(z_sg)
        z_std_g = self.linear_zg_std(z_g)

        z_s, z_sg, z_g = self.get_z(z_mean_s, z_std_s, z_mean_sg, z_std_sg, z_mean_g, z_std_g)
        return z_s, z_sg, z_g
    
    def generate(self, num_graph, z_s=None, z_g=None, z_sg=None, batch_size=64):
        # Produce those num_graph graphs.
        # return them.
        z_in_n_dim = self.network_encoder.output_dim + self.joint_encoder.output_dim
        z_in_s_dim = self.spatial_encoder.output_dim + self.joint_encoder.output_dim
        data_out = []
        for j in range(num_graph//batch_size):
            j_st = j*batch_size
            j_ed = (j+1)*batch_size
            if z_g is None or z_sg is None:
                z_in_n = torch.randn(batch_size, z_in_n_dim)
            else:
                z_in_n = torch.cat([z_sg[j_st:j_ed], z_g[j_st:j_ed]], axis=1)
            if z_s is None or z_sg is None:
                z_in_s = torch.randn(batch_size, z_in_s_dim)
            else:
                z_in_s = torch.cat([z_sg[j_st:j_ed], z_s[j_st:j_ed]], axis=1)
            adj_out = self.network_decoder(z_in_n)
            x_out = self.spatial_decoder(z_in_s)
            for k in range(batch_size):
                use_adj = (adj_out[k] > 0.5).type(torch.FloatTensor)
                data_out.append(Data(x=x_out[k], edge_index=dense_to_sparse(use_adj[:, :, 0])[0]))
        return data_out
        


class trainerSGD():
    def __init__(self, model, lr=1e-4, folder=None, batch_size=64):
        self.model = model
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                # nn.init.normal(param, 0.02)
                nn.init.constant(param, 0)
            if 'weight' in name:
                try:
                    nn.init.xavier_uniform(param, gain=nn.init.calculate_gain('relu')/10)
                    # nn.init.normal(param, 0.002)
                except:
                    nn.init.normal(param, std=0.002)
        nn.init.constant(self.model.linear_zg_std.bias, -1)
        nn.init.constant(self.model.linear_zs_std.bias, -1)
        nn.init.constant(self.model.linear_zgs_std.bias, -1)
        self.optimD = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.folder = folder
        if folder is not None:
            if not os.path.exists(folder):
                os.mkdir(folder)
        self.cnt_train = 0
        self.batch_size = batch_size
        
        
    def train(self, dataloader, epoch=3000, save_cnt=500, verbose = 100, vverbose=False):
        ii = 0
        for j in range(epoch):
            print (f"Epoch {j} ===============")
            for use_x in dataloader:
                ii += 1
                self.optimD.zero_grad()
                x = use_x['x']
                adj = use_x['adj']
                lossX, lossA, loss_kl = self.model(x, adj)
                (lossX.mean() + lossA.mean() + loss_kl.mean()).backward()
                self.optimD.step()
                if ii % verbose == 1:
                    print (f"At {ii}, loss X is {lossX.mean().item()}; loss A is {lossA.mean().item()}; loss KL is {loss_kl.mean().item()}")
                if ii % save_cnt == 0 and self.folder is not None:
                    torch.save(self.model, os.path.join(self.folder, f"graphSGD_{self.cnt_train}.pt"))
                    self.cnt_train += 1
