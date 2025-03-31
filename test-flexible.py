import argparse
import os
import json
import torch
from tqdm import tqdm
from Bio import PDB
import numpy as np
from evaluate.dataset import test_complex_process, BaseComplex, CA_INDEX
from evaluate.dockq import dockQ
from evaluate.tm_score import tm_score
from evaluate.rmsd import compute_crmsd, compute_irmsd, protein_surface_intersection

def monomer2complex(monomers, save_path):
    parser = PDB.PDBParser(QUIET=True)
    comp_writer = PDB.PDBIO()
    comp_model = PDB.Model.Model('annoym')
    for mon in monomers:
        structure = parser.get_structure('annoym', mon)
        for model in structure:
            for chain in model:
                comp_model.add(chain)
    comp_writer.set_structure(comp_model)
    comp_writer.save(save_path)

def main(args):
    work_path = '.'
    model_type = args.model
    dataset = args.dataset
    print(f'Model type: {model_type}')
    print(f'Dataset: {dataset}')
    test_path = '%s/dataset/%s'%(work_path, dataset)
    test_dict_path = '%s/%s.json'%(test_path, dataset)

    #if os.path.exists(test_dict_path):
    #    with open(test_dict_path, 'r') as f1:
    #        lines = f1.read().strip().split('\n')
    #    for line in lines:
    #        item = json.loads(line)
    #        test_dict[item['pdb']] = [item['rchain'], item['lchain']]
    #else:
    #    raise ValueError(f'Dataset {dataset}.json not found')
    
    pdbids, a_crmsds, a_irmsds, u_crmsds, u_irmsds, dockqs, tmscores, intersections = [], [], [], [], [], [], [], []
    with open(test_dict_path, 'r') as f1:
        lines = f1.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            pdbid = item['pdb']
            rchain_id = item['rchain']
            lchain_id = item['lchain']

            if '-g' in dataset:
                ligand_path = '%s/structures/%s/%s_l_b_g.pdb'%(test_path,pdbid,pdbid)
                receptor_path = '%s/structures/%s/%s_r_b_g.pdb'%(test_path,pdbid,pdbid)
            elif 'afm' in dataset:
                ligand_path = '%s/structures/%s/%s_l_b_afm.pdb'%(test_path,pdbid,pdbid)
                receptor_path = '%s/structures/%s/%s_r_b_afm.pdb'%(test_path,pdbid,pdbid)    
            elif 'af3' in dataset:
                ligand_path = '%s/structures/%s/%s_l_b_af3.pdb'%(test_path,pdbid,pdbid)
                receptor_path = '%s/structures/%s/%s_r_b_af3.pdb'%(test_path,pdbid,pdbid)                
            elif 'hawk' in dataset:
                ligand_path = '%s/structures/%s/%s_l_b_hawk.pdb'%(test_path,pdbid,pdbid)
                receptor_path = '%s/structures/%s/%s_r_b_hawk.pdb'%(test_path,pdbid,pdbid)                   
            else:
                ligand_path = '%s/structures/%s/%s_l.pdb'%(test_path,pdbid,pdbid)
                receptor_path = '%s/structures/%s/%s_r.pdb'%(test_path,pdbid,pdbid)            

            result_dir = '%s/results/%s/%s'%(work_path, dataset, model_type)
            #save_dir = args.save_dir
            save_dir = result_dir
            gt = test_complex_process(ligand_path, receptor_path)
            gt_X = gt['X'][:, CA_INDEX].numpy()
            pred_ligand_path = '%s/%s/%s_%s_id.pdb'%(save_dir, pdbid, pdbid, model_type)
            gt_complex_path = '%s/%s/%s_gt.pdb'%(save_dir, pdbid, pdbid)
            pred_complex_path = '%s/%s/%s_predicted.pdb'%(save_dir, pdbid, pdbid)
            pred_rec_path = '%s/%s/%s_%s_rec_id.pdb'%(save_dir, pdbid, pdbid, model_type)
            monomer2complex([receptor_path, ligand_path], gt_complex_path)
            #monomer2complex([receptor_path, pred_ligand_path], pred_complex_path)
            monomer2complex([pred_rec_path, pred_ligand_path], pred_complex_path)
            dock_X = BaseComplex.from_pdb(
                    pred_complex_path, ligand_path
                ).ligand_coord()[:, CA_INDEX]
            
            Seg = gt['Seg'].numpy()
            dock_X_re, dock_X_li = torch.tensor(dock_X[Seg == 0]), torch.tensor(dock_X[Seg == 1])
            assert dock_X.shape[0] == gt_X.shape[0], 'coordinates dimension mismatch'

            aligned_crmsd = compute_crmsd(dock_X, gt_X, aligned=False)
            aligned_irmsd = compute_irmsd(dock_X, gt_X, Seg, aligned=False)
            unaligned_crmsd = compute_crmsd(dock_X, gt_X, aligned=True)
            unaligned_irmsd = compute_irmsd(dock_X, gt_X, Seg, aligned=True)
            intersection = float(protein_surface_intersection(dock_X_re, dock_X_li).relu().mean() +
                protein_surface_intersection(dock_X_li, dock_X_re).relu().mean())
            a_crmsds.append(aligned_crmsd)
            a_irmsds.append(aligned_irmsd)
            u_crmsds.append(unaligned_crmsd)
            u_irmsds.append(unaligned_irmsd)
            #print(aligned_crmsd, aligned_irmsd)
            tmscores.append(tm_score(gt_complex_path, pred_complex_path))
            dockqs.append(dockQ(pred_complex_path, gt_complex_path,
                            rchain_id=rchain_id, lchain_id=lchain_id))
            
            intersections.append(intersection)
            pdbids.append(pdbid)
            os.remove(gt_complex_path)
            os.remove(pred_complex_path)
    
    data = {
        "pdbid": pdbids,
        "model_type": model_type.upper(),
        "IRMSD": a_irmsds,
        "CRMSD": a_crmsds,
        "TMscore": tmscores,
        "DockQ": dockqs,
        "intersection": intersections
    }
    data = json.dumps(data, indent=4)
    with open(os.path.join(save_dir, '%s.json'%model_type), 'w') as fp:
        fp.write(data)

    for name, val in zip(['CRMSD(aligned)', 'IRMSD(aligned)', 'TMscore', 'DockQ', 'CRMSD', 'IRMSD'],
                         [a_crmsds, a_irmsds, tmscores, dockqs, u_crmsds, u_irmsds]):
        print(f'{name} median: {np.median(val)}', end=' ')
        print(f'mean: {np.mean(val)}', end=' ')
        print(f'std: {np.std(val)}')

def parse():
    parser = argparse.ArgumentParser(description='Docking given flexible protein-protein complex')
    parser.add_argument('-d','--dataset',type=str, default='PPCBench30-af3', help='DB5-af3, DB5-afm, DB5-g-b, DB5-hawk-b')
    parser.add_argument('-m','--model', type=str, default='af3_1', help='af3, afm, geodock, hawkdock')
    parser.add_argument('-s','--save_dir', type=str, default=None, help='Directory to save results')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())