# PPCBench & AACBench
Revisiting Protein-protein Docking: A Systematic Evaluation Framework

### Requirements
```
biopython
pandas
numpy
pytorch
scipy
tqdm
```

### Datasets & Results
The full datasets and results can be found here: https://zenodo.org/records/15112025.

### Models
- [HDOCK](http://hdock.phys.hust.edu.cn/)
- [PatchDock](https://bioinfo3d.cs.tau.ac.il/PatchDock/)
- [PIPER](https://cluspro.bu.edu/home.php)
- [ZDOCK](https://zdock.wenglab.org/software/)
- [EquiDock](https://github.com/octavian-ganea/equidock_public/)
- [ElliDock](https://github.com/yaledeus/ElliDock)
- [EBMDock](https://github.com/wuhuaijin/EBMDock)
- [DiffDock-PP](https://github.com/biocad/DiffDock-PP/)
- [GeoDock](https://github.com/Graylab/GeoDock)
- [AlphaFold-Multimer](https://github.com/deepmind/alphafold)
- [AlphaFold3](https://github.com/google-deepmind/alphafold3)

### DockQ

```
cd evaluate && git clone --branch v1.0 https://github.com/bjornwallner/DockQ.git
```

### TM-Score

```
cd evaluate && g++ -O3 -o TMscore TMscore.cpp
```

### Examples for obatining metrics
___# If it is the result of a rigid docking model___
```
python test-rigid.py --dataset DB5 --model equidock
```
___# If it is the result of a flexible docking model___
```
python test-flexible.py --dataset DB5-g-u --model geodock
```

