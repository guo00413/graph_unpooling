# PyTorch implementation for _An Unpooling Layer for Graph Generation_
Accepted in AISTATS 2023

1. Notebooks are located in `./notebooks`
	- For Waxman random graph data:
		- To produce dataset, please use `RandomGraph_generation.ipynb`.
		- To draw the distributions, please use `Draw_WaxmanRandomGraph_distributions.ipynb`.
		- To reproduce training of UL GAN/ VAE in random graph, please use `UL_GAN_WaxmanRandomGraph_training.ipynb` and `UL_VAE_WaxmanRandomGraph_training.ipynb`.
	- For protein data:
		- To preprocessing data, please use `protein_preprocessing.ipynb`.
		- To reproduce training of UL GAN/UL VAE in random graph, please use `UL_GAN_protein_training.ipynb` and `UL_VAE_protein_training.ipynb`.
	- For molecule data:
		- To preprocessing smiles data of QM9/ZINC, please use `qm9_preprocessing.ipynb` and `ZINC_preprocessing.ipynb`.
		- To reproduce the evaluation of UL GAN in QM9/ZINC, please use `QM9_evaluation.ipynb` and `ZINC_evaluation.ipynb`.
		- To reproduce the training of UL GAN in QM9/ZINC, please use `UL_GAN_QM9_training.ipynb` and `UL_GAN_ZINC_training.ipynb`.
		- To reproduce the training of UL VAE in QM9, please use `UL_VAE_QM9_training.ipynb`.
		- To reproduce the training of UL GAN in QM9 for chemical property, please use `UL_GAN_optimize_chemical_property.ipynb`.
		- To reproduce the latent space exploration, please use `ULGAN_latent_space_exploration.ipynb`
2. Data (txt smiles) are located in `./data`
	- One need to use pre-processing notebooks (`qm9_preprocessing.ipynb` and `ZINC_preprocessing.ipynb`) to generate graph data from smiles.
	- The model training/evaluating uses the preprocessed graph data.
	- Protein data can be downloaded from https://graphgt.mathcs.emory.edu/datasets/Biology/Protein_dataset.zip
	- ZINC data can be downloaded from https://github.com/wengong-jin/icml18-jtnn/blob/master/data/zinc/all.txt
3. Some trained models are located in `./models`
	- For ease to use, the trained models are converted to `cpu` before saving.
	- The evaluation notebooks (e.g. `QM9_evaluation.ipynb`) load those trained models, report their performance metrics and draw some sample molecules.
4. Source codes are in `./codes/`
	- Unpooling layer's class is given in `./codes/unpool_layers_simple_v2.py`
	- Dependence:
```
		matplotlib==3.5.1
		networkx==2.6.3
		numpy==1.22.0
		pandas==1.3.5
		rdkit==2009.Q1-1
		torch==1.8.1+cu101
		torch_geometric==2.0.3
		torch_scatter==2.0.9
		torch_sparse==0.6.12
		# additional dependent packages:
		scscore # Clone from https://github.com/connorcoley/scscore
```
