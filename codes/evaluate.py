# Some evaluation functions for random graph dataset and protein dataset.
# Including:
#  kld_evaluation (produce kld and wd's.)
#  graph_set_info (generate graph properties based on graphs.)
#  generate_samples_GraphVAE (generate graph samples for graphVAE.)

import torch
from scipy.special import rel_entr
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
import numpy as np
import pandas as pd
from networkx import average_clustering, clustering, average_node_connectivity, Graph
from scipy.stats import wasserstein_distance
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse



def convert_sample_to_space(samples, space):
    hist_res = [((samples >= space[j]) & (samples < space[j+1])).sum()/len(samples) for j in range(len(space) - 1)]
    return hist_res

def kld_evaluation(train_data, test_data, train_distributions=None, sample_cnt=20, node_feature_samples=50):
    # First generate distribution
    if train_distributions is None:
        node_cnts, edge_cnts, node_features, degrees, dense_edge, connectivities, clustering = graph_set_info(train_data, return_raw=True)
    else:
        node_cnts, edge_cnts, node_features, degrees, dense_edge, connectivities, clustering = train_distributions
    node_cnts_test, edge_cnts_test, node_features_test, degrees_test, dense_edge_test, connectivities_test, clustering_test = graph_set_info(test_data, return_raw=True)

    info = {}
    ref_samples = connectivities
    test_samples = connectivities_test

    space = np.linspace(np.min(ref_samples)-1e-5, np.max(ref_samples)+1e-5, sample_cnt)
    ref_hist = convert_sample_to_space(ref_samples, space)
    test_hist = convert_sample_to_space(test_samples, space)
    info['kl_connectivity'] = rel_entr(np.array(ref_hist) + 1e-7, np.array(test_hist) + 1e-7).sum()
    info['wd_connectivity'] = wasserstein_distance(ref_samples, test_samples)
    ref_samples = dense_edge
    test_samples = dense_edge_test

    space = np.linspace(np.min(ref_samples)-1e-5, np.max(ref_samples)+1e-5, sample_cnt)
    ref_hist = convert_sample_to_space(ref_samples, space)
    test_hist = convert_sample_to_space(test_samples, space)
    info['kl_edge_density'] = rel_entr(np.array(ref_hist) + 1e-7, np.array(test_hist) + 1e-7).sum()
    info['wd_edge_density'] = wasserstein_distance(ref_samples, test_samples)
    ref_samples = clustering
    test_samples = clustering_test

    space = np.linspace(np.min(ref_samples)-1e-5, np.max(ref_samples)+1e-5, sample_cnt)
    ref_hist = convert_sample_to_space(ref_samples, space)
    test_hist = convert_sample_to_space(test_samples, space)
    info['kl_clustering_coef'] = rel_entr(np.array(ref_hist) + 1e-7, np.array(test_hist) + 1e-7).sum()
    info['wd_clustering_coef'] = wasserstein_distance(ref_samples, test_samples)

    ref_samples = degrees
    test_samples = degrees_test

    space = np.linspace(np.min(ref_samples)-1e-5, np.max(ref_samples)+1e-5, sample_cnt)
    ref_hist = convert_sample_to_space(ref_samples, space)
    test_hist = convert_sample_to_space(test_samples, space)
    info['kl_avg_degrees'] = rel_entr(np.array(ref_hist) + 1e-7, np.array(test_hist) + 1e-7).sum()
    info['wd_avg_degrees'] = wasserstein_distance(ref_samples, test_samples)
    

    ref_samples = np.array(node_features)
    test_samples = np.array(node_features_test)

    kls = []
    wds = []
    if test_samples.shape[1] < ref_samples.shape[1]:
        print ("Node feature is not supported")
        return info
    for j in range(ref_samples.shape[1]):
        space = np.linspace(np.min(ref_samples[:, j])-1e-5, np.max(ref_samples[:, j])+1e-5, node_feature_samples)
        ref_hist = convert_sample_to_space(ref_samples[:, j], space)
        test_hist = convert_sample_to_space(test_samples[:, j], space)
        kls.append(rel_entr(np.array(ref_hist) + 1e-7, np.array(test_hist) + 1e-7).sum())
        wds.append(wasserstein_distance(ref_samples[:, j], test_samples[:, j]))
    info['kl_node_features'] = kls
    info['wd_node_features'] = wds
    return info

def generate_samples_GraphVAE(graph_vae, n, indim=128, verbose=False, z_inputs=None):
    _ = graph_vae.eval()
    use_data = []
    for jj in range(n//64 + 1):
        if z_inputs is None:
            y = graph_vae.vae.decode(torch.randn(64, indim))
        else:
            y = graph_vae.vae.decode(z_inputs[jj*64:(jj+1)*64])
        A = (torch.sigmoid(y[:, :graph_vae.edge_dim]) > 0.5).type(torch.FloatTensor)
        X = y[:, graph_vae.edge_dim:].view(64, graph_vae.max_num_nodes, graph_vae.input_dim)
        As = []
        for j in range(len(A)):
            recon_adj_lower = graph_vae.recover_adj_lower(A[j])
            recon_adj_tensor = graph_vae.recover_full_adj_from_lower(recon_adj_lower)
            
            use_id = recon_adj_tensor[torch.arange(graph_vae.max_num_nodes), torch.arange(graph_vae.max_num_nodes)] > 0.5
            useAs = recon_adj_tensor[use_id, :][:, use_id]
            useAs[torch.arange(use_id.sum()), torch.arange(use_id.sum())] = 0
            edge_index, edge_attr = dense_to_sparse(useAs)
            use_data.append(Data(x=X[j, use_id, :], edge_index=torch.LongTensor(edge_index)))
        if verbose:
            print (jj)

    return use_data
    
def graph_set_info(use_data, node_cnt_space=[5, 6, 7, 8, 9, 10, 11, 12], edge_cnt_space=range(4, 66), node_feature_space=np.linspace(0, 1, 1000), 
                   degrees_space=range(12), return_raw=False, \
                    dense_edge_space=np.linspace(0.15, 0.565, 20), \
                    connectivities_space=np.linspace(1, 3.75, 20), \
                    clustering_space=np.linspace(0, 0.87, 20)):
    # Let use_data to be a list of Data.
    total_graph = len(use_data)
    total_nodes = sum([j.x.size(0) for j in use_data])
    node_cnts = np.array([j.x.size(0) for j in use_data])
    edge_cnts = np.array([j.edge_index.size(1)//2 for j in use_data])
    node_features = torch.cat([j.x for j in use_data], axis=0).detach().numpy()
    degrees = [to_dense_adj(j.edge_index, torch.zeros(j.edge_index.max()+1, dtype=int)).sum(axis=(0, 1)).mean().item() for j in use_data]
    dense_edge = [to_dense_adj(j.edge_index, torch.zeros(j.edge_index.max()+1, dtype=int)).mean(axis=(1, 2)).item() for j in use_data]
    connectivities = []
    clustering = []
    for gg in use_data:
        g = Graph()
        # g.add_nodes_from(range(gg.edge_index.max() + 1))
        g.add_nodes_from(np.unique((gg.edge_index[:, 0]).numpy()))
        g.add_edges_from(gg.edge_index.numpy().T)
        clustering.append(average_clustering(g))
        connectivities.append(average_node_connectivity(g))
    return node_cnts, edge_cnts, node_features, degrees, dense_edge, connectivities, clustering

    # node_cnts = [(node_cnts == j).sum()/total_graph for j in node_cnt_space]
    # edge_cnts = [(edge_cnts == j).sum()/total_graph for j in edge_cnt_space]
    # node_features = [((node_features >= node_feature_space[j]) & (node_features < node_feature_space[j+1])).sum()/total_nodes/2 for j in range(len(node_feature_space) - 1)]
    # degrees = [(degrees == j).sum()/total_nodes for j in degrees_space]
    # dense_edge = np.array(dense_edge)
    # connectivities = np.array(connectivities)
    # clustering = np.array(clustering)    
    
    # dense_edge = [((dense_edge >= dense_edge_space[j]) & (dense_edge < dense_edge_space[j+1])).sum()/total_graph for j in range(len(dense_edge_space) - 1)]
    # connectivities = [((connectivities >= connectivities_space[j]) & (connectivities < connectivities_space[j+1])).sum()/total_graph for j in range(len(connectivities_space) - 1)]
    # clustering = [((clustering >= clustering_space[j]) & (clustering < clustering_space[j+1])).sum()/total_graph for j in range(len(clustering_space) - 1)]
    # return node_cnts, edge_cnts, node_features, degrees, dense_edge, connectivities, clustering
