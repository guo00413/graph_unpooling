# Some Util functions for GNN.
# Including:
#  draw_graph, calculate_gp(QM9)/calculate_gp_enriched(ZINC),  
#  weight_initiate, generate_noise, convert_Batch_to_datalist

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd import grad
from torch_geometric.nn import global_add_pool


def process_data(i, dj):
    use_ids = dj.x[:, 0] == 0
    node_code_map = (torch.cumsum(use_ids, axis=0) - 1).type(torch.LongTensor)


    use_edges = use_ids[dj.edge_index].prod(axis=0).type(torch.BoolTensor)    
    newx = dj.x[use_ids, 1:]
    newz = dj.z[use_ids]
    newpos = dj.pos[use_ids]
    use_edge_index = dj.edge_index[:, use_edges]
    use_edge_attr = dj.edge_attr[use_edges]
    edge_type = ((torch.arange(0, 4).view(1, -1) * use_edge_attr).sum(axis=1)).type(torch.LongTensor)
    use_edge_attr = torch.cat((use_edge_attr, \
                                torch.pow(dj.pos[use_edge_index[0]] - dj.pos[use_edge_index[1]], 2).sum(axis=1).view(-1, 1)), 1)
    use_edge_index = node_code_map[use_edge_index]
    return Data(x=newx, y=dj.y.view(1, -1), z=newz, idx=dj.idx, pos=newpos, edge_index=use_edge_index, edge_attr=use_edge_attr, name=dj.name, edge_type=edge_type)

def handle_dataset(a, pre_processs=True):
    '''
    Input: a should be the dataset of QM9

    We will do:
    1. Make process of a.
    2. Clamp ys based on 0.001% percentile and 99.9 percentile.
    3. Return std of ys.
    '''
    data_list = [process_data(i, dj) for i, dj in enumerate(a)]
    if not pre_processs:
        return data_list
    all_ys = torch.cat([j.y.view(1, -1) for j in data_list], 0)
    mins, maxs =torch.FloatTensor(np.percentile(all_ys, 0.001, axis=0)).view(1, -1), torch.FloatTensor(np.percentile(all_ys, 99.9, axis=0)).view(1, -1)
    for j in range(all_ys.shape[1]):
        all_ys[:, j] = torch.clamp(all_ys[:, j], mins[0, j], maxs[0, j])
    use_std = all_ys.std(axis=0)

    for i, j in enumerate(data_list):
        j.y = all_ys[i, :].view(1, -1)
    return data_list, use_std


def draw_graph(data=None, x=None, edge_index=None, edge_attr=None, figsize=None, pos=None, \
               node_code=True, ax=None, cmaps='Reds', **more_options):
    """Given a torch_geometric.Data, draw it using networkx.

    Args:
        data (_type_, optional): _description_. Defaults to None.
        x (_type_, optional): _description_. Defaults to None.
        edge_index (_type_, optional): _description_. Defaults to None.
        edge_attr (_type_, optional): _description_. Defaults to None.
        figsize (_type_, optional): _description_. Defaults to None.
        pos (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.
        cmaps (str, optional): _description_. Defaults to 'Reds'.

    Returns:
        _type_: _description_
    """
    # Input x, edge_index, edge_attr to draw the figure.
    # print ('updated')
    if data is not None:
        x = data.x
        edge_index = data.edge_index
        try:
            edge_attr = data.edge_attr
        except:
            edge_attr = None
    if figsize is None:
        figsize = (6, 6)

    G = nx.Graph()
    G.add_nodes_from(range(len(x)))
    G.add_edges_from(np.array(edge_index).T)
    if pos is None:
        pos = nx.spring_layout(G)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    options = {
        "node_size": 500,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1.2,
        "width": 3,
    }
    options.update(more_options)
    if node_code:
        nx.draw(G, pos, ax=ax, **options)
    # code_map = {0:'H',  1:'C', 2:'N',  3:'O',  4:'F'}
    code_map = {0:'C', 1:'N',  2:'O',  3:'F'}
    edge_map = {0: 'S', 1:'D', 2:'T', 3:'A'}
    atom_code = [code_map[j.item()] for j in (x[:, :4].argmax(axis=1))]
    labels = dict(zip(range(len(x)), atom_code))
    if node_code:
        try:
            nx.draw_networkx_labels(G, pos, labels, font_size=15, fontweight='bold', ax=ax)
        except:
            nx.draw_networkx_labels(G, pos, labels, font_size=15, ax=ax)
        if edge_attr is not None:
            edge_labels = dict([((e[0].item(), e[1].item()), edge_map[int(at[:4].argmax())]) for e, at in zip(np.array(edge_index).T, edge_attr) if e[0].item() < e[1].item()])
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    else:
        val_map = dict(zip(range(len(x)), x[:, 0].detach()))
        values = [val_map.get(node, 0.25) for node in G.nodes()]

        nx.draw(G, cmap=plt.get_cmap(cmaps), node_color=values, with_labels=False, ax=ax)
    
    return fig, ax


class WeightedMSE():
    """
    Calculate the negative log-likelihood given sample X and mu/var
    for normal distribution.
    """
    def __init__(self, weights, device='cpu'):
        self.weights = (weights.view(1, -1) + 1e-7).to(device)

    def __call__(self, x, target):
        MSE = torch.pow(x - target, 2)*self.weights
        sample_nll = (MSE.sum(axis=1).mean())
        return sample_nll

def calculate_gp(real, out, device='cpu'):
    """Given Data (real) and output (out), return the penalty for gradient of out w.r.t real.x and real.edge_attr (following WGAN)
    Used for QM9

    Args:
        real (torch_geometric.Data): contain x and edge_attr, edge_index, batch, edge_index_batch.
        out (torch.Variable): output, can be loss function.
        device (str, optional): Device of the calculation. Defaults to 'cpu'.

    Returns:
        the penalty that is ||\nabla_data out| - 1|^2
    """
    v_x = Variable(real.x[:, :4], requires_grad=True).to(device)
    v_e = Variable(real.edge_attr[:,:3], requires_grad=True).to(device)
    prob = out(x=v_x, edge_index=real.edge_index, edge_attr=v_e, batch=real.batch, edge_batch=real.edge_index_batch).view(-1)
    gradients_x = grad(outputs=prob, inputs=v_x,
                               grad_outputs=torch.ones(prob.size()).to(device),
                               create_graph=True, retain_graph=True, allow_unused=True)[0]
    gradients_e = grad(outputs=prob, inputs=v_e,
                               grad_outputs=torch.ones(prob.size()).to(device),
                               create_graph=True, retain_graph=True, allow_unused=True)[0]
    gradients_x = gradients_x.view(len(real.x), -1)
    gx = global_add_pool(torch.sum(torch.pow(gradients_x, 2), dim=1), batch=real.batch)
    if len(real.edge_attr) > 0:
        gradients_e = gradients_e.view(len(real.edge_attr), -1)
        ge = global_add_pool(torch.sum(torch.pow(gradients_e, 2), dim=1), batch=real.edge_index_batch)
        if len(ge) < len(gx):
            ge = torch.cat([ge, torch.zeros(len(gx) - len(ge)).to(device)])
    else:
        ge = torch.zeros(len(gx)).to(device)
    penalty = torch.pow(torch.sqrt(gx + ge + 1e-9) - 1, 2).mean()
    return penalty


def calculate_gp_enriched(real, out, device='cpu'):
    """Given Data (real) and output (out), return the penalty for gradient of out w.r.t real.x and real.edge_attr (following WGAN)
    Used for ZINC

    Args:
        real (torch_geometric.Data): contain x and edge_attr, edge_index, batch, edge_index_batch.
        out (torch.Variable): output, can be loss function.
        device (str, optional): Device of the calculation. Defaults to 'cpu'.

    Returns:
        the penalty that is ||\nabla_data out| - 1|^2
    """
    v_x = Variable(real.x[:, :], requires_grad=True).to(device)
    v_e = Variable(real.edge_attr[:, :], requires_grad=True).to(device)
    prob = out(x=v_x, edge_index=real.edge_index, edge_attr=v_e, batch=real.batch, edge_batch=real.edge_index_batch).view(-1)
    gradients_x = grad(outputs=prob, inputs=v_x,
                               grad_outputs=torch.ones(prob.size()).to(device),
                               create_graph=True, retain_graph=True, allow_unused=True)[0]
    gradients_e = grad(outputs=prob, inputs=v_e,
                               grad_outputs=torch.ones(prob.size()).to(device),
                               create_graph=True, retain_graph=True, allow_unused=True)[0]
    gradients_x = gradients_x.view(len(real.x), -1)
    gx = global_add_pool(torch.sum(torch.pow(gradients_x, 2), dim=1), batch=real.batch)
    if len(real.edge_attr) > 0:
        gradients_e = gradients_e.view(len(real.edge_attr), -1)
        ge = global_add_pool(torch.sum(torch.pow(gradients_e, 2), dim=1), batch=real.edge_index_batch)
        if len(ge) < len(gx):
            ge = torch.cat([ge, torch.zeros(len(gx) - len(ge)).to(device)])
    else:
        ge = torch.zeros(len(gx)).to(device)
    penalty = torch.pow(torch.sqrt(gx + ge + 1e-9) - 1, 2).mean()
    return penalty


def weight_initiate(m):
    # Make inititiate with some variance.
    if(type(m) == nn.BatchNorm2d) or (type(m) == nn.modules.batchnorm.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif (type(m) == nn.BatchNorm1d) or (type(m) == nn.modules.batchnorm.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    else:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.normal_(m.bias.data, 0.0, 0.02)
        if hasattr(m, "root") and m.root is not None:
            nn.init.normal_(m.bias.data, 0.0, 0.02)

    

def generate_noise(rand_dim=32, rand_type='uniform', lr_cont_dim=30, cate_dim=5, cont_dim=3, batch=64, device='cpu'):
    """Return randomly generated noise

    Args:
        rand_dim (int, optional): Dimension of random vector. Defaults to 32.
        rand_type (str, optional): How to generate the randomness. Defaults to 'uniform'.
        lr_cont_dim (int, optional): NOT IN USE; those are for conditional generation.. Defaults to 17.
        cate_dim (int, optional): NOT IN USE; those are for conditional generation.. Defaults to 5.
        cont_dim (int, optional): NOT IN USE; those are for conditional generation.. Defaults to 5.
        batch (int, optional): NOT IN USE; those are for conditional generation.. Defaults to 64.
        device (str, optional): Using device, 'cpu' or 'cuda:i'. Defaults to 'cpu'.

    Returns:
        random vector, labeled continuous random vector, random categorical variables, random continuous vector (in latent space for unsupervised learning)
        # for molecular generation, we only use first output now...
    """
    if rand_type == 'uniform':
        out_rand = torch.rand(batch, rand_dim) * 2 - 1
    else:
        out_rand = torch.randn(batch, rand_dim)

    c_cate = torch.zeros(batch, cate_dim, device=device)
    idx = np.random.randint(cate_dim, size=batch)
    c_cate[torch.arange(0, batch), idx] = 1.0
    c_cate = c_cate.view(batch, cate_dim)

    lr_cont = torch.randn(batch, lr_cont_dim)

    c_cont = torch.randn(batch, cont_dim)
    out_rand, lr_cont, c_cate, c_cont = out_rand.to(device), lr_cont.to(device), c_cate.to(device), c_cont.to(device)
    # out_rand = torch.cat([out_rand, lr_cont, c_cate, c_cont], axis=1)
    # out_rand = out_rand.to(device)
    return out_rand, lr_cont, c_cate, c_cont


def convert_Batch_to_datalist(x, edge_index, edge_attr, batch, edge_batch):
    """Given a data batch from torch_geometric, return a list of Data.

    Args:
        x (tensor): All nodes features (n*d), n nodes, d dimension feature.
        edge_index (tensor of long): 2*m, m edges.
        edge_attr (tensor): m*p, m edges, p edge features.
        batch (tensor of long): n size.
        edge_batch (tensor of long): m size.

    Returns:
        a list of Data, each Data is torch_geometric.utils.Data.
    """
    result = []
    for j in range(max(batch) + 1):
        use_ids = (batch == j).cpu()
        use_edge_ids = (edge_batch == j).cpu()
        use_x = x[use_ids].cpu()
        use_edge_index = edge_index[:, use_edge_ids].cpu()
        use_edge_attr = edge_attr[use_edge_ids].cpu()
        remap_ids = torch.arange(len(x))[use_ids]
        new_ids = torch.arange(len(use_x))
        for k in range(len(new_ids)):
            use_edge_index[use_edge_index == remap_ids[k]] = new_ids[k].item()
        result.append(Data(x=use_x, edge_index=use_edge_index, edge_attr=use_edge_attr))
    return result

