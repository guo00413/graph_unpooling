# Some util function for molecule process.
# Useful ones including:
#  MolFromGraphs, MolFromTorchGraphData_enriched, evaluate, calculate_y_graph

from rdkit import Chem
import torch
import numpy as np
from util_richer import calculate_ys as calculate_y

def MolFromGraphs(node_list, adjacency_matrix):
    # Generate mol from nodes & Adj matrix.
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()            

    return mol

# 0:C, 1:N, 2:O, 3:F
atom_map = {0: 6, 1:7, 2:8, 3:9}
bond_map = {0:Chem.rdchem.BondType.SINGLE, 1:Chem.rdchem.BondType.DOUBLE, \
            2:Chem.rdchem.BondType.TRIPLE}

# atom_list = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']

atom_enriched_map = {0: 6, 1:7, 2:8, 3:9, 4:16, 5:17, 6:35, 7:53, 8:15}
chiral_enriched_map = {0:Chem.rdchem.ChiralType.CHI_UNSPECIFIED, \
                        1:Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, \
                        2:Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW}
charge_enriched_map = {0:0, 1:1, 2:-1}


bond_enriched_map = {0:Chem.rdchem.BondType.SINGLE, 1:Chem.rdchem.BondType.DOUBLE, \
            2:Chem.rdchem.BondType.TRIPLE, 3:Chem.rdchem.BondType.AROMATIC}
stere_enriched_map = {0:Chem.rdchem.BondStereo.STEREONONE, 1:Chem.rdchem.BondStereo.STEREOE, 2:Chem.rdchem.BondStereo.STEREOZ}


def MolFromTorchGraph(x, edge_index, edge_attr):
    """From a torch_geometric.Data (which contains x/edge_index/edge_attr), build a Chem.mol
    DONT USE, because it only considers atom type as node feature...
    Please use `MolFromTorchGraphData_enriched`

    Args:
        x (tensor): Node features (n*d), n is number of nodes, d is node dimension.
        edge_index (tensor of Long): Edges, (2*m), m is number of edges.
        edge_attr (tensor): (m*p), m is number of edges, p is edge dimension.

    Returns:
        Chem.Mol
    """
    mol = Chem.RWMol()
    n, p = x.shape
    for j in range(n):
        use_id = int((x[j]*torch.arange(p)).sum().item())
        a = Chem.Atom(atom_map[use_id])
        mol.AddAtom(a)
    m, d = edge_attr.shape
    for j in range(m):
        use_bond = int((edge_attr[j]*torch.arange(d)).sum().item())
        if edge_index[0][j] >= edge_index[1][j]:
            continue
        mol.AddBond(int(edge_index[0][j]), int(edge_index[1][j]), bond_map[use_bond])
    return mol
    # return mol

def MolFromTorchGraphData(data):
    return MolFromTorchGraph(data.x[:, :4], data.edge_index, data.edge_attr[:, :3])

def calculate_y_graph(graph, molType=False, **para):
    if molType:
        mol = graph
    else:
        mol = MolFromTorchGraphData_enriched(graph.to('cpu'), **para)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol is None:
        return [0, 0, 0]
    return calculate_y(mol)

def MolFromTorchGraphData_enriched(data, node_dim=9, charge_map=None, chiral_map=None, stereo_map=None, without_aroma=False, device='cpu'):
    """From a torch_geometric.Data (which contains x/edge_index/edge_attr), build a Chem.mol

    Args:
        data (torch_geometric.Data): Contains data.x, data.edge_index, data.edge_attr
        node_dim (int, optional): Dimension of node type (atom types). Defaults to 9.
        charge_map (_type_, optional): A map for charge type. Defaults to charge_enriched_map.
        chiral_map (_type_, optional): A map for chiral type. Defaults to chiral_enriched_map.
        stereo_map (_type_, optional): NOT IN USE. Defaults to None.
        without_aroma (bool, optional): T/F if we use aroma, WE ONLY USE THIS FOR TRUE. Defaults to False.
        device (str, optional): Device. Defaults to 'cpu'.

    Returns:
        Chem.Mol.
    """
    if chiral_map is None:
        chiral_map = chiral_enriched_map
    if charge_map is None:
        charge_map = charge_enriched_map
    if stereo_map is None:
        stereo_map = stere_enriched_map
    mol = Chem.RWMol()
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    n, p = x.shape
    for j in range(n):
        use_id = int((x[j, :node_dim]*torch.arange(node_dim).to(device)).sum().item())
        a = Chem.Atom(atom_enriched_map[use_id])
        use_id = int((x[j, node_dim:node_dim + len(charge_map)]* torch.arange(len(charge_map)).to(device)).sum().item())
        a.SetFormalCharge(charge_map[use_id])
        use_id = int((x[j, node_dim + len(charge_map):node_dim + len(charge_map) + len(chiral_map)]* torch.arange(len(chiral_map)).to(device)).sum().item())
        a.SetChiralTag(chiral_map[use_id])
        if not without_aroma:
            if x[j, node_dim + len(charge_map) + len(chiral_map)] > 0:
                a.SetIsAromatic(True)
        mol.AddAtom(a)
        
    m, d = edge_attr.shape
    for j in range(m):
        if edge_index[0][j] >= edge_index[1][j]:
            continue
        if without_aroma:
            use_id = int((edge_attr[j, :len(bond_enriched_map)-1]*torch.arange(len(bond_enriched_map)-1).to(device)).sum().item())
        else:
            use_id = int((edge_attr[j, :len(bond_enriched_map)]*torch.arange(len(bond_enriched_map)).to(device)).sum().item())
        use_bond = bond_enriched_map[use_id]

        id_e = mol.AddBond(int(edge_index[0][j]), int(edge_index[1][j]), use_bond)
        # bond = mol.GetBondWithIdx(id_e - 1)
        # use_id = int((edge_attr[j, len(bond_enriched_map):len(bond_enriched_map) + len(stereo_map)]*torch.arange(len(stereo_map))).sum().item())
        # bond.SetStereo(stereo_map[use_id])
        # if use_id > 0:
        #     bond.SetStereoAtoms(int(edge_index[0][j]), int(edge_index[1][j]))
    return mol
    # return mol

    pass


def MolFromTorchGraphBatch(x, edge_index, edge_attr, batch, edge_batch):
    # Given a data including batch/edge batch, return the list of mols.
    res_mols = []
    max_id = batch.max().item() + 1
    for j in range(max_id):
        use_id_xs = batch == j
        use_id_edges = edge_batch == j
        res_mols.append(MolFromTorchGraph(x[use_id_xs], edge_index[:, use_id_edges], edge_attr[use_id_edges]))
    return res_mols

# Save molecular:
# fig = Draw.MolToImage(m)
# fig.save('mol_rdkit.png')

def check_validation(data):
    """Check validity of data.

    Args:
        data (list of Chem.Mol)

    Returns:
        list of smiles of valid ones, valid rate.
    """
    smile_data = [Chem.MolFromSmiles(Chem.MolToSmiles(m)) for m in data]
    smile_data = [j for j in smile_data if j is not None]
    val_rate = len(smile_data)/len(data)
    return smile_data, val_rate

def check_uniqueness(val_smiles, eval_two=False):
    """Check uniqueness.. If eval_two=True, print unique rate of first 1000 and unique rate of first 10k..

    Args:
        val_smiles (list of smiles)
        eval_two (bool, optional): T/F if we do additional print. Defaults to False.

    Returns:
        an array of unique smiles, uniqueness rate.
    """
    # 
    if len(val_smiles) == 0:
        return [], 0
    smiles = [Chem.MolToSmiles(m) for m in val_smiles]
    if eval_two:
        print ("Uniqueness at 1k, 10k", len(np.unique(smiles[:1000]))/len(smiles[:1000]), len(np.unique(smiles[:10000]))/len(smiles[:10000]))
    return np.unique(smiles), len(np.unique(smiles))/len(smiles)

def check_novelty(smiles, mol_smiles):
    """Check novelty.. 

    Args:
        smiles (list of unique smiles)
        mol_smiles (list of smiles): The smiles from training data.

    Returns:
        novelty rate.
    """
    if len(smiles) == 0:
        return 0
    novel_data = [j for j in smiles if not j in mol_smiles]
    return len(novel_data)/len(smiles)

def evaluate(generated_data, mol_data, eval_two=False):
    """Given generated list of Chem.Mol and a list of smiles (training data),
    Return the 3 metrics.

    Args:
        generated_data (list of Chem.Mol): generated samples.
        mol_data (list of smiles): smiles of training data.
        eval_two (bool, optional): T/F if we print out unique rate of 1k and 10k. Defaults to False.

    Returns:
        validity rate, uniqueness rate, novelty rate.
    """
    val_smiles, val_rate = check_validation(generated_data)
    uniq_smiles, unique_rate = check_uniqueness(val_smiles, eval_two=eval_two)
    novel_rate = check_novelty(uniq_smiles, mol_data)
    return val_rate, unique_rate, novel_rate


from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import BCUT2D
from rdkit.Chem import Crippen
from rdkit import Chem

def calculate_ys(j, means=None, stds=None):
    """NOT IN USE, please use util_richer.calculate_ys for chemical properties.
    """
    temp = []
#     j = Chem.MolFromSmiles(Chem.MolToSmiles(jj))
    try:
        temp.extend(BCUT2D(j))
    except:
        try:
            temp.extend(BCUT2D(Chem.AddHs(j)))
        except:
            temp.extend([None]*8)
    try:
        temp.append(qed(j))
    except:
        try:
            temp.append(qed(Chem.AddHs(j)))
        except:
            temp.extend([None])
    try:
        temp.append(Crippen.MolLogP(j))
    except:
        try:
            temp.append(Crippen.MolLogP(Chem.AddHs(j)))
        except:
            temp.extend([None])
    try:
        temp.append(Crippen.MolMR(j))
    except:
        try:
            temp.append(Crippen.MolMR(Chem.AddHs(j)))
        except:
            temp.extend([None])    
    if means is None:
        temp = [j if j is not None else 0 for j in temp]
    else:
        temp = [(j-means[i])/stds[i] if j is not None else 0 for (i, j) in enumerate(temp)]
    return temp



