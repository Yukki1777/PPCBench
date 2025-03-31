from Bio.PDB import PDBParser
from Bio.Data import IUPACData
from Bio.PDB.Polypeptide import standard_aa_names as AA_NAMES_3
from typing import List
import numpy as np
import torch
import os

parser = PDBParser(QUIET=True)
AA_NAMES_1 = tuple(IUPACData.protein_letters_3to1.values())
BACKBONE_ATOM = ['N', 'CA', 'C', 'O']
N_INDEX, CA_INDEX, C_INDEX, O_INDEX = 0, 1, 2, 3

### get index of the residue by 3-letter name
def aa_index_3(aa_name_3):
    return AA_NAMES_3.index(aa_name_3)

### get index of the residue by 1-letter name
def aa_index_1(aa_name_1):
    return AA_NAMES_1.index(aa_name_1)

### get 1-letter name of the residue by 3-letter name
def aa_3to1(aa_name_3):
    return AA_NAMES_1[aa_index_3(aa_name_3)]

### get 3-letter name of the residue by 1-letter name
def aa_1to3(aa_name_1):
    return AA_NAMES_3[aa_index_1(aa_name_1)]

def recursive_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [recursive_to_list(i) for i in obj]
    else:
        return obj

def base_pdb_parse(pdb_path):
    #print('pdb_parse: ' + pdb_path)
    filename = os.path.basename(pdb_path)
    pdb_id = filename[:4]
    structure = parser.get_structure(pdb_id, pdb_path)

    seq = {}            # sequence
    coord = {}          # coordinates, backbone only
    len = 0             # length

    for chain in structure[0]:
        chain_name = chain.get_id()
        assert chain_name != '', 'chain name is not valid.'
        seq[chain_name] = ''
        coord[chain_name] = []
        for residue in chain:
            # print(f"residue id: {residue.get_id()}")
            if (res_name := residue.get_resname()) in AA_NAMES_3:
                seq[chain_name] += aa_3to1(res_name)
                len += 1
                backbone_coord = []
                for bb in BACKBONE_ATOM:
                    if residue[bb]!= None:
                        backbone_coord.append(residue[bb].get_coord())
                coord[chain_name].append(backbone_coord)
        coord[chain_name] = np.asarray(coord[chain_name]).tolist()
        #coord[chain_name] = recursive_to_list(coord[chain_name])
        #print(coord)

    return seq, coord

class BaseComplex:
    def __init__(self, ligand_seq, receptor_seq, ligand_coord, receptor_coord):
        self.li_seq = ligand_seq
        self.re_seq = receptor_seq
        self.li_coord = ligand_coord
        self.re_coord = receptor_coord

    @classmethod
    def from_pdb(cls, ligand_path, receptor_path):
        #print("ligand_path",ligand_path)
        ligand_seq, ligand_coord = base_pdb_parse(ligand_path)
        receptor_seq, receptor_coord = base_pdb_parse(receptor_path)
        return cls(ligand_seq, receptor_seq, ligand_coord, receptor_coord)

    def ligand_seq(self) -> List:
        ligand_seq = []
        for _, chain_seq in self.li_seq.items():
            for res_name_1 in chain_seq:
                ligand_seq.append(aa_index_1(res_name_1))
        return ligand_seq

    def ligand_coord(self):
        ligand_coord = np.array([])
        for _, chain_coord in self.li_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not ligand_coord.shape[0]:
                ligand_coord = chain_coord
            elif ligand_coord.ndim == chain_coord.ndim:
                ligand_coord = np.concatenate((ligand_coord, chain_coord), axis=0)
            else:
                continue
        return ligand_coord

    def receptor_seq(self) -> List:
        receptor_seq = []
        for _, chain_seq in self.re_seq.items():
            for res_name_1 in chain_seq:
                receptor_seq.append(aa_index_1(res_name_1))
        return receptor_seq

    def receptor_coord(self):
        receptor_coord = np.array([])
        for _, chain_coord in self.re_coord.items():
            chain_coord = np.asarray(chain_coord)
            if not receptor_coord.shape[0]:
                receptor_coord = chain_coord
            elif receptor_coord.ndim == chain_coord.ndim:
                receptor_coord = np.concatenate((receptor_coord, chain_coord), axis=0)
            else:
                continue
        return receptor_coord

    def receptor_relative_pos(self) -> List:
        return [i for i in range(len(self.receptor_seq()))]

    def receptor_identity(self) -> List:
        receptor_id = []
        i = 0
        for _, chain_seq in self.re_seq.items():
            receptor_id.extend([i] * len(chain_seq))
            i += 1
        return receptor_id

    def ligand_relative_pos(self) -> List:
        return [i for i in range(len(self.ligand_seq()))]

    def ligand_identity(self) -> List:
        ligand_id = []
        i = 0
        for _, chain_seq in self.li_seq.items():
            ligand_id.extend([i] * len(chain_seq))
            i += 1
        return ligand_id

    # def find_keypoint(self, threshold=8.) -> np.ndarray:
    #     receptor_ca_coord = self.receptor_coord()[:, CA_INDEX]     # CA coordinates, (N, 3)
    #     ligand_ca_coord = self.ligand_coord()[:, CA_INDEX]
    #     abag_dist = cdist(receptor_ca_coord, ligand_ca_coord)
    #     ab_idx, ag_idx = np.where(abag_dist < threshold)
    #     keypoints = 0.5 * (receptor_ca_coord[ab_idx] + ligand_ca_coord[ag_idx])
    #     return keypoints


def test_complex_process(ligand_path, receptor_path):
    complex = BaseComplex.from_pdb(ligand_path, receptor_path)
    # receptor
    receptor_seq = complex.receptor_seq()
    receptor_bb_coord = complex.receptor_coord()  # backbone atoms, [N_re, 4, 3]
    receptor_rp = complex.receptor_relative_pos()
    receptor_id = complex.receptor_identity()

    # ligand
    ligand_seq = complex.ligand_seq()
    ligand_bb_coord = complex.ligand_coord()  # backbone atoms, [N_li, 4, 3]
    #print(f"ligand_bb_coord type: {type(ligand_bb_coord)}, shape: {np.array(ligand_bb_coord).shape}")
    ligand_rp = complex.ligand_relative_pos()
    ligand_id = complex.ligand_identity()
    
    # Ensure receptor_bb_coord and ligand_bb_coord are numpy arrays
    receptor_bb_coord = np.array(receptor_bb_coord)
    ligand_bb_coord = np.array(ligand_bb_coord)

    # center
    center = np.mean(ligand_bb_coord.reshape(-1, 3), axis=0)

    assert receptor_bb_coord.ndim == 3, f'invalid receptor coordinate dimension: {receptor_bb_coord.ndim}'
    assert ligand_bb_coord.ndim == 3, f'invalid ligand coordinate dimension: {ligand_bb_coord.ndim}'
    assert len(receptor_seq) == len(receptor_rp) and len(receptor_seq) == len(receptor_id) and len(receptor_seq) == \
           receptor_bb_coord.shape[0], 'receptor seq/coord/rp/id dimension mismatch'
    assert len(ligand_seq) == len(ligand_rp) and len(ligand_seq) == len(ligand_id) and len(ligand_seq) == \
           ligand_bb_coord.shape[0], 'ligand seq/coord/rp/id dimension mismatch'
    assert receptor_bb_coord.shape[1] == ligand_bb_coord.shape[1] and receptor_bb_coord.shape[2] == \
           ligand_bb_coord.shape[2], 'receptor and ligand coordinates mismatch'

    data = {
        'S': torch.tensor(np.array(receptor_seq + ligand_seq), dtype=torch.long),
        'X': torch.tensor(np.concatenate((receptor_bb_coord, ligand_bb_coord), axis=0), dtype=torch.float),
        'RP': torch.tensor(np.array(receptor_rp + ligand_rp), dtype=torch.long),
        'ID': torch.tensor(np.array(receptor_id + ligand_id), dtype=torch.long),
        ### segment, 0 for receptor and 1 for ligand
        'Seg': torch.tensor(
            np.array([0 for _ in range(len(receptor_seq))] + [1 for _ in range(len(ligand_seq))]),
            dtype=torch.long
        ),
        'center': torch.tensor(center, dtype=torch.float).unsqueeze(0),
        'bid': torch.tensor([0] * (len(receptor_seq) + len(ligand_seq)), dtype=torch.long),
    }

    return data