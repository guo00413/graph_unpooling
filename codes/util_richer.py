# Some additional functions for molecules' chemical properties.
# Including:
#  calculate_ys, dataFromSmile
# This function will be used by calculate_y_graph in util_molecular.py

from rdkit import Chem
import torch
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.data import Data
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import BCUT2D
from rdkit.Chem import Crippen
try:
    from scscore.utils.SA_Score import sascorer
except:
    import warnings
    warnings.warn("Don't have scscore package, please clone it from https://github.com/connorcoley/scscore and copy the `scscore` folder under codes folder. Currently, we will return 0 for SA score.")
    import sascorer_fake as sascorer
import numpy as np

atom_list = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']
charge_list = [0, 1, -1]
chiral_list = [0, 2, 1]
bond_list = [1, 2, 3, 12]
# stereo_list = [0, 3, 2]

logPmin = -2.12178879609
logPmax = 6.0429063424

samin = 5.0
samax = 1.5



def remap_clip(x, x_min, x_max):
    v = (x - x_min) / (x_max - x_min)
    return np.clip(v, 0.0, 1.0)

def calculate_ys(mol):
    return [qed(mol), \
            remap_clip(Crippen.MolLogP(mol), logPmin, logPmax), \
            remap_clip(sascorer.calculateScore(mol), samin, samax)\
                ]


def one_encoding(x, values):
    return [x == j for j in values]


def dataFromSmile(smile_str, without_aroma=False, node_dim=9):
    # Given a smiles string, 
    # Return a geometric data
    mol = Chem.MolFromSmiles(smile_str)
    if without_aroma:
        Chem.Kekulize(mol)
    cnt = 0
    xs = []
    idx_map = {}
    edge_index = torch.LongTensor(2, 0)
    for j in mol.GetAtoms():
        atom_t = j.GetSymbol()
        charge = j.GetFormalCharge()
        chiral = j.GetChiralTag()
        useid = j.GetIdx()
        if without_aroma:
            xs.append(torch.FloatTensor([one_encoding(atom_t, atom_list[:node_dim]) + \
                    one_encoding(charge, charge_list) + \
                    one_encoding(chiral, chiral_list)]))
        else:
            xs.append(torch.FloatTensor([one_encoding(atom_t, atom_list[:node_dim]) + \
                    one_encoding(charge, charge_list) + \
                    one_encoding(chiral, chiral_list) + [j.GetIsAromatic()]]))
        idx_map[useid] = cnt
        cnt += 1
    edge_attr = []
    for j in mol.GetBonds():
        st = j.GetBeginAtomIdx()
        ed = j.GetEndAtomIdx()
        st_j = idx_map[st]
        ed_j = idx_map[ed]
        edge_index = torch.cat([edge_index, torch.LongTensor([[st_j], [ed_j]])], axis=1)
        bond = j.GetBondType()
        if without_aroma:
            edge_attr.append(torch.FloatTensor([one_encoding(bond, bond_list[:-1])])) 
        else:
            edge_attr.append(torch.FloatTensor([one_encoding(bond, bond_list)])) 
    xs = torch.cat(xs, axis=0)
    edge_attr = torch.cat(edge_attr, axis=0)
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    # Add chemical properties:
    ys = torch.FloatTensor(calculate_ys(mol))
    return Data(x=xs, edge_index=edge_index, edge_attr=edge_attr, y=ys)

