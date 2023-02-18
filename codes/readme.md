# Brief introduction of files:
1. `unpool_layers_simple_v2.py`: Implementation of Unpooling Layer (see class `UnpoolLayerEZ`).
2. `gcn_model_sim_summ_2.py`: Some basic GCN layers that will be used in the work.
3. `gcn_model_sim_summ_qm.py`: Generator for QM9, including UL GAN and Adj GAN.
4. `ugcn_model_summ_2.py`: Discriminator for QM9.
5. `gcn_model_sim_summ_z2.py`: Generator for ZINC, including UL GAN and Adj GAN, and discriminator for ZINC.
6. `model_vae.py`: Encoder and decoder of UL VAE for QM9.
7. `model_graph.py`: UL GAN and UL VAE for Waxman random graph and for protein dataset.
8. `baseline_models`: Folder including baseline methods implementations for Waxman random graph and for protein dataset.
    1. `adjgan.py`: Implementation of Adj GAN.
    2. `graphAF.py`: GraphAF
    3. `graphCPN.py`: GCPN
    4. `graphRNN.py`: GraphRNN
    5. `graphSGD.py`: SGD-VAE
    6. `graphVAE.py`: GraphVAE
9. `evaluate.py`: Some functions used to evaluate graph generation.
10. `adj_generator.py`: Functions to convert adjacency matrix to graph data.
11. `sascorer_fake.py`: A trivial function to avoid error when scscore is not cloned.
12. `trainer.py`: Class for WGAN training process.
13. `train_vae.py`: Class for VAE training process.
14. `unpool_utils.py`: Some supporting functions for UL, including to assemble skip-z, convert Batch to list of Data.
15. `util_gnn.py`: Some supporting functions for GNN, including to calculate gradient penalty.
16. `util_molecular.py`: Some supporting functions for molecules, including to convert graph data to Mol (class defined by rdkit).
17. `util_richer.py`: Some additional supporting function for molecules, including to calculate chemical properties.
18. `util_seq_genereation.py`: Some supporting function for sequentially generating samples.
19. `verifyD.py`: Some classes to make validation, mostly used for testing.
