# A support function for ADJ generator, to transfer the X/A to torch_geometric.Data (to keep all samples in the same fashion)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool, global_mean_pool, NNConv, MessagePassing
from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool import knn
from torch_geometric.data import Data, Batch
from torch.distributions.categorical import Categorical
from torch_geometric.utils import dense_to_sparse

def convertA_to_data_enriched(X, A, node_dim=17, atom_dim=10, chiral_dim=3, charge_dim=3, edge_dim=5, \
        batch=None, gumbel=True, tau=0.5, hard=True, device='cpu', eps=1e-7, obtain_connected=False, \
        without_ar=False):
    """
    Given X, A and batch, return a list of torch_geometric.Data
    
    here, X/A will have one more dimension than the required node/edge feature, the one more dimension is for NON-existing of the edge/node.

    Args:
        X (tensor): n*d tensor, n is number of nodes, d is node feature dimension.
        A (tensor): n*n*p tensor, n is number of nodes, p is edge feature dimension.
        node_dim (int, optional): Node dimension. Defaults to 17.
        atom_dim (int, optional): Atom types (one more than real number of atom types). Defaults to 10.
        chiral_dim (int, optional): Number of Chiral types. Defaults to 3.
        charge_dim (int, optional): number of formal charge types. Defaults to 3.
        edge_dim (int, optional): Number of edge dimension (one more than real types). Defaults to 5.
        gumbel (bool, optional): T/F if we do Gumbel Softmax. Defaults to True.
        tau (float, optional): tau for Gumbel softmax. Defaults to 0.5.
        hard (bool, optional): parameter hard for Gumbel softmax. Defaults to True.
        device (str, optional): Used device. Defaults to 'cpu'.
        eps (_type_, optional): epsilon in the program.. Defaults to 1e-7.
        obtain_connected (bool, optional): If we want to obtain the connected part. Defaults to False.
        without_ar (bool, optional): must to True for now.

    Return:
        useX, edge_index, edge_attr
    """
    if gumbel:
        use_X = F.gumbel_softmax(torch.log(X[:, :atom_dim] + eps), tau=tau, hard=hard)
        use_A = F.gumbel_softmax(torch.log(A + eps), tau=tau, hard=hard)
    else:
        use_X = X[:, :atom_dim]
        use_A = A
    adj = (use_A[:, :, :edge_dim - 1]*(torch.arange(1, edge_dim).to(device)).view(1, 1, -1)).sum(axis=-1)
    use_ids = (use_X[:, -1] < 1.0)
    final_X = use_X[use_ids, :atom_dim - 1]

    if gumbel:
        add_X = F.gumbel_softmax(torch.log(X[:, atom_dim:atom_dim + charge_dim] + eps), tau=tau, hard=hard)
    else:
        add_X = X[:, atom_dim:atom_dim + charge_dim]
    if gumbel:
        add2_X = F.gumbel_softmax(torch.log(X[:, atom_dim + charge_dim:atom_dim + charge_dim + chiral_dim] + eps), tau=tau, hard=hard)
    else:
        add2_X = X[:, atom_dim + charge_dim:atom_dim + charge_dim + chiral_dim]

    if without_ar:
        final_X = torch.cat([final_X, add_X[use_ids,:], add2_X[use_ids,:]], axis=1)
    else:
        if gumbel:
            add3_X = F.gumbel_softmax(torch.log(X[:, atom_dim + charge_dim + chiral_dim:] + eps), tau=tau, hard=hard)
        else:
            add3_X = X[:, atom_dim + charge_dim + chiral_dim:atom_dim + charge_dim + chiral_dim + 2]
        final_X = torch.cat([final_X, add_X[use_ids,:], add2_X[use_ids,:], add3_X[use_ids, :1]], axis=1)


    node_code_map = (torch.cumsum(use_ids, axis=0) - 1).type(torch.LongTensor)
    final_edge_index, final_edge_attr = dense_to_sparse(adj)
    use_edges = use_ids[final_edge_index].prod(axis=0).type(torch.BoolTensor)
    final_edge_index =  final_edge_index[:, use_edges]
    final_edge_attr = use_A[final_edge_index[0], final_edge_index[1]]
    # final_edge_attr = final_edge_attr[use_edges]
    final_edge_index =  node_code_map[final_edge_index]
    use_edge_ids = final_edge_index[0, :] < final_edge_index[1, :]
    final_edge_index = torch.cat([final_edge_index[:, use_edge_ids], \
                                torch.cat([final_edge_index[1, use_edge_ids].view(1, -1), final_edge_index[0, use_edge_ids].view(1, -1)], axis=0)], \
                                axis=1)
    final_edge_attr = torch.cat([final_edge_attr[use_edge_ids], final_edge_attr[use_edge_ids]], axis=0)
    # Obtain its best connected part.

    if obtain_connected and len(final_X) > 0:
        use_nodes = []
        node_ids = torch.arange(len(final_X))
        temp_x = (final_X[:, 0]*5 + final_X.sum(axis=1)).argmax()
        use_nodes.append(node_ids[temp_x])
        pre_len = len(use_nodes)
        for j in range(len(final_X)):
            for k in use_nodes:
                use_nodes.extend(final_edge_index[1, final_edge_index[0, :] == k].cpu())
                use_nodes = list(np.unique(use_nodes))
            if len(use_nodes) == pre_len:
                break
            pre_len = len(use_nodes)
        # get new ids
        newX = torch.zeros(final_X.shape).to(device)
        newX[use_nodes] = final_X[use_nodes]
        use_ids = (newX.sum(axis=1) >= 1.0)
        final_X = final_X[use_ids]
        node_code_map = (torch.cumsum(use_ids, axis=0) - 1).type(torch.LongTensor).to(device)
        use_edges = use_ids[final_edge_index].prod(axis=0).type(torch.BoolTensor)
        final_edge_index =  final_edge_index[:, use_edges]
        final_edge_attr = final_edge_attr[use_edges, :]
        final_edge_index =  node_code_map[final_edge_index]

    return final_X, final_edge_index, final_edge_attr[:, :edge_dim-1]


def convertA_to_data():
    pass



def obtain_connected_block(X, edge_index, edge_attr, batch, edge_batch, max_iter=9, device='cpu'):
    """
    Given X, edge_index, edge_attr, and batch, return the connected graph for each batch.
    """
    use_nodes = []
    node_ids = torch.arange(len(X))
    for j in np.unique(batch.cpu()):
        batch_X = X[batch == j]
        batch_ids = node_ids[batch == j]
        temp_x = (batch_X[:, 0]*5 + batch_X.sum(axis=1)).argmax()
        use_nodes.append(batch_ids[temp_x])
    use_nodes = list(np.unique(use_nodes))
    pre_len = len(use_nodes)
    for j in range(max_iter):
        for k in use_nodes:
            use_nodes.extend(edge_index[1, edge_index[0, :] == k].cpu())
            use_nodes = list(np.unique(use_nodes))
        if len(use_nodes) == pre_len:
            break
        pre_len = len(use_nodes)

    newX = torch.zeros(X.shape).to(device)
    newX[use_nodes] = X[use_nodes]
    use_ids = (newX.sum(axis=1) >= 1.0)
    final_X = newX[use_ids]
    node_code_map = (torch.cumsum(use_ids, axis=0) - 1).type(torch.LongTensor).to(device)
    use_edges = use_ids[edge_index].prod(axis=0).type(torch.BoolTensor)
    final_edge_index =  edge_index[:, use_edges]
    final_edge_attr = edge_attr[use_edges, :]
    final_edge_index =  node_code_map[final_edge_index]
    final_batch = batch[use_ids]
    final_edge_batch = edge_batch[use_edges]

    return final_X, final_edge_index, final_edge_attr, final_batch, final_edge_batch
