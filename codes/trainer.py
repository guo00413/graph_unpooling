# The trainer or UL-GAN or ADJ-GAN.
# Also, will do evaluation at every K steps and store trained generators.

import torch
import os
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm1d
from torch_geometric.data import batch
from torch_geometric.nn.glob.glob import global_add_pool
import torch.nn.functional as F
from torch import optim
from torch import nn
from util_gnn import weight_initiate, generate_noise
from util_gnn import calculate_gp, convert_Batch_to_datalist, calculate_gp_enriched
from util_molecular import MolFromTorchGraphData, evaluate, MolFromTorchGraphData_enriched, calculate_y_graph
from rdkit import Chem
from rdkit.Chem import Draw

class GANTrainer(object):
    """The UL-/ADJ- GAN trainer.

    Args:
        d (nn.Module): a discriminator
        g (nn.Module): a generator
        rand_dim (int): the random vector dimension for generator.
        train_folder (str): the folder where we dump intermediate models.
        tot_epoch_num (int, optional): Total epoch number. Defaults to 300.
        eval_iter_num (int, optional): Every XX iteration, we will evaluate the model. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): the use device.
        d_add (nn.Module, optional): Another discriminator to add (mostly used for ZINC) to capture some simple features. Defaults to None.
        learning_rate_g (float, optional): learning rate for generator. Defaults to 1e-3.
        learning_rate_d (float, optional): learning rate for discriminator. Defaults to 1e-4.
        lambda_g (float, optional): gradient's penalty coefficient. Following previous paper, just set to 10 if wanted. Defaults to 0
        max_train_G (int, optional): Maximal training runs for generator at each iteration. Defaults to 2.
        tresh_add_trainG (float, optional): The threshold when we want to add more training run for generator (i.e. when D can differentiate more than this threshold for True/Fake data, we add one more step for G). Defaults to 0.2.
        use_loss (string, optional): The type of loss we use, choose from 'wgan', 'bce', 'reward'. Defaults to 'wgan'
        lambda_rl (float, optional): Additional learning coefficient from RL (recommend to be small). Defaults to 1.0.
        lambda_noedges (float, optional): Penalty coefficient for no-edge existing, only applicable for ADJ generator. Defaults to 0..
        trainD (bool, optional): T/F if we want to train D, only for testing. Defaults to True.
        zinc (bool, optional): T/F if we use ZINC data. Defaults to False.
        qm9 (bool, optional): T/F if we use QM9 data. Defaults to False.
        without_ar (bool, optional): T/F if we don't use aroma types. Recommend to True. Defaults to False.
        reward_point (int, optional): 0 for QED, 1 for logP, 2 for Synthezisability. Defaults to None.
        initial_weight (bool, optional): _description_. Defaults to True.
    """
    def __init__(self, d, g, rand_dim, train_folder, tot_epoch_num=300, eval_iter_num=100, batch_size=64, \
        device=None, d_add=None, learning_rate_g=1e-3, learning_rate_d=1e-4, \
        lambda_g=0, max_train_G=2, tresh_add_trainG=0.2, \
        use_loss='wgan', \
        g_out_prob=False, lambda_rl=1.0, \
        lambda_nonodes = 0.,
        lambda_noedges = 0.,
        trainD=True, \
        zinc=False, \
        qm9=False, \
        without_ar=False, \
        reward_point=None, 
        initial_weight=True
        ):
        '''
        use_loss can be "wgan", or "bce", or "reward"
        reward_point is the integer where we need to make for the reward.
        #   0 for QED
        #   1 for logP
        #   2 for SAscore
        '''
        self.reward_point = reward_point
        if device is None:
            device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
        if trainD:
            gcn_model = d.to(device)
            if d_add is not None:
                gcn_add = d_add.to(device)
        else:
            gcn_model = d
            if d_add is not None:
                gcn_add = d_add
        generator = g.to(device)
        lr_d = learning_rate_d
        lr_g = learning_rate_g
        self.lambda_node = lambda_nonodes
        self.lambda_edge = lambda_noedges
        beta_d = (0.5, 0.999)
        beta_g = (0.5, 0.999)
        if trainD:
            if d_add is not None:
                optimD = optim.Adam([{'params': gcn_model.parameters()},
                                    {'params': gcn_add.parameters()}
                                    ], \
                                    lr=lr_d, betas=beta_d)
                if initial_weight:
                    gcn_add.apply(weight_initiate)

            else:
                optimD = optim.Adam([{'params': gcn_model.parameters()},
                                    ], \
                                    lr=lr_d, betas=beta_d)
            self.optimD = optimD
            if initial_weight:
                gcn_model.apply(weight_initiate)

        optimG = optim.Adam([{'params': generator.parameters()}, \
                            ], lr=lr_g, betas=beta_g)
        if initial_weight:
            generator.apply(weight_initiate)
        z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=rand_dim, device=device)
        self.fix_noise = z_rand
        self.rand_dim = rand_dim
        if d_add is not None:
            self.gcn_add = gcn_add
        else:
            self.gcn_add = None
        self.gcn_model = gcn_model
        self.generator = generator
        self.lambda_g = lambda_g
        self.lambda_rl = lambda_rl

        self.folder = train_folder
        self.tot_epoch_num = tot_epoch_num
        self.eval_iter_num = eval_iter_num
        self.batch_size = batch_size
        self.max_train_G = max_train_G
        self.tresh_add_trainG = tresh_add_trainG
        self.use_loss = use_loss
        self.evals = []
        self.error_rates = []
        self.optimG = optimG
        self.g_out_prob = g_out_prob
        self.criterionD = nn.BCELoss()
        self.device = device
        self.zinc = zinc
        self.qm9 = qm9
        self.without_ar = without_ar
        



    def call_g(self, data=None, usex=None, batch=None, edge_index=None, edge_attr=None, edge_batch=None):
        # Call the discriminator to classify the data.
        # Return the output of discriminator on this given data (probability of predicting real)
        if data is not None:
            usex = data.x
            batch = data.batch
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            edge_batch = data.edge_index_batch

        p1 = self.gcn_model(x=usex, \
                            edge_index=edge_index, \
                            edge_attr=edge_attr, \
                            batch=batch, edge_batch=edge_batch)
        if self.gcn_add is None:
            probs_real = p1
        else:
            p2 = self.gcn_add(x=usex, \
                            edge_index=edge_index, \
                            edge_attr=edge_attr, \
                            batch=batch, edge_batch=edge_batch)
            probs_real = p1 + p2
        return probs_real

    def train(self, data_loader, verbose=False, use_data_x = 4, use_data_edgeattr=3, \
        evaluate_num=1000, mol_data=None, pretrain_epoch=0, saveZ=False, trainD=True, \
            alter_trainer=False, NN=1000, save_images=None, use_node_dim=4, plot_para={}, \
                reward_fake_step=1, reinforce_acclerate=False, rl_only=False, useBN=False, \
                    acclerate_noscale=True):
        """Train the GAN framework

        Args:
            data_loader (Dataloader): This dataloader is producing true data.
            verbose (bool, optional): If we print additional information. Defaults to False.
            use_data_x (int, optional): Number of node dimension, it should be 10 for QM9, 15 for ZINC. Defaults to 4.
            use_data_edgeattr (int, optional): Numbre of edge dimension. Defaults to 3.
            evaluate_num(int, optional): Number of sample generated for each evaluation step.
            mol_data (list of smiles, optional): A list of all smiles strings from training data, used for evaluating if given. Defaults to None.
            pretrain_epoch (int, optional): Number of epoch for pretraining the discriminator. 
                                            This is ONLY used when optimizing a certain chemical property, 
                                            in this case, the discriminator is trying to predict 
                                            the given chemical property. 
                                            Defaults to 0.
            saveZ (bool, optional): If or not we save the random vector z's, only used for testing. Defaults to False.
            trainD (bool, optional): If we update discriminator, only used for testing. Defaults to True.
            alter_trainer (bool, optional): If we update based on GAN and REINFORCE alternatively, defaults to False. Recommend to True. 
            NN (int, optional): We will evaluate and save generator/discriminator every NN step. Defaults to 1000.
            save_images (str or None, optional): Folder of saving generated plots. If given, we will save generated plots in the folder. Defaults to None.
            use_node_dim (int, optional): Used node dimension for calculate_y_graph, which should be 4 for QM9. Defaults to 4.
            plot_para (dict, optional): Additional kwargs for plotting. Defaults to {}.
            reward_fake_step (int, optional): Only used when self.use_loss='reward', # of step when we use fake data to update reward of discriminator.
            reinforce_acclerate (bool, optional): If we use REINFORCE with baseline. Defaults to False.
            rl_only (bool, optional): If we only run REINFORCE step, only for testing. Defaults to False.
            useBN (bool, optional): If we used BN for REINFORCE with baseline. Defaults to False.
            acclerate_noscale(bool, optional): if we use REINFORCE with baseline without std, Recommend as True. Defaults to True.
        """
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        ii = 0
        import time
        real_label = 1.0
        fake_label = 0.0
        self.temp_loss_D = []
        self.temp_loss_G = []
        add_train_g = 1
        kk = 0
        self.all_zs = []
        label = torch.full((self.batch_size, ), real_label, device=self.device)
        if reinforce_acclerate and not hasattr(self, 'reinf_bn') and useBN:
            self.reinf_bn = BatchNorm1d(1, affine=False).to(self.device)
        if pretrain_epoch > 0:
            for epoch in range(pretrain_epoch):
                for i, data in enumerate(data_loader):
                    if max(data.batch) < self.batch_size - 1:
                        print (f'less than {self.batch_size}')
                        continue
                    data = data.to(self.device)
                    b_size = data.num_graphs
                    self.optimD.zero_grad()
                    label.fill_(real_label)
                    if use_data_x is not None:
                        usex = data.x[:, :use_data_x]
                    else:
                        usex = data.x
                    if use_data_edgeattr is not None:
                        use_edge_attr = data.edge_attr[:, :use_data_edgeattr]
                    else:
                        use_edge_attr = data.edge_attr
                    # Predicted results.
                    probs_real = self.call_g(usex=usex, batch=data.batch, edge_index=data.edge_index, \
                                            edge_attr=use_edge_attr, edge_batch=data.edge_index_batch)

                    if self.use_loss == 'wgan':
                        loss_real = -(probs_real.mean())
                    elif self.use_loss == 'reward':
                        all_ys = torch.FloatTensor([calculate_y_graph(data[k], node_dim=use_node_dim, without_aroma=self.without_ar) for k in range(data.num_graphs)])
                        # Collecting the target (y)
                        real_ys = all_ys[:, self.reward_point:self.reward_point + 1]
                        real_ys = real_ys.to(self.device)
                        loss_real = (torch.pow(probs_real - real_ys, 2).mean())
                    else:
                        loss_real = self.criterionD(probs_real, label)

                    loss_real.backward()
                    if self.use_loss != 'reward' or i % 2 == 0:
                        label.fill_(fake_label)
                        z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=self.rand_dim, device=self.device)
                        z = z_rand
                        if self.g_out_prob:
                            fake_data, prob = self.generator(z)
                        else:
                            fake_data = self.generator(z, obtain_connected=True)
                        probs_fake = self.call_g(fake_data)
                        # In the pre-train epoch, we only train discriminator.                        
                        if self.use_loss == 'wgan':
                            loss_fake = (probs_fake.mean())
                        elif self.use_loss == 'reward':
                            real_ys = torch.FloatTensor([calculate_y_graph(fake_data[k], node_dim=use_node_dim, without_aroma=self.without_ar) for k in range(fake_data.num_graphs)])[:, self.reward_point:self.reward_point + 1]
                            real_ys = real_ys.to(self.device)
                            loss_fake = (torch.pow(probs_fake - real_ys, 2).mean())
                        else:
                            loss_fake = self.criterionD(probs_fake, label)
                        loss_fake.backward()
                    if self.lambda_g > 0:
                        if self.zinc or self.qm9:
                            penalty1 = calculate_gp_enriched(data, self.gcn_model, device=self.device)
                            penalty2 = calculate_gp_enriched(fake_data, self.gcn_model, device=self.device)
                        else:
                            penalty1 = calculate_gp(data, self.gcn_model, device=self.device)
                            penalty2 = calculate_gp(fake_data, self.gcn_model, device=self.device)
                        gp = (penalty1 + penalty2) * self.lambda_g
                        gp.backward()
                    else:
                        pass
                    # In pre-train epoch, we only train D.
                    self.optimD.step()
                    if i % 100 == 0:
                        print (f'[{i}/{len(data_loader)}] Real loss: {loss_real.mean().item()}, fake loss:{loss_fake.mean().item()}.')
        if hasattr(self, 'eval_counts') and self.eval_counts:
            count_eval = True
            real_counts = []
            real_rewards = []
            fake_counts = []
            fake_rewards = []
        else:
            count_eval = False
        for epoch in range(self.tot_epoch_num):
            epoch_start_time = time.time()
            for i, data in enumerate(data_loader):
                if trainD:
                    if max(data.batch) < self.batch_size - 1:
                        print (f'less than {self.batch_size}')
                        continue
                    data = data.to(self.device)
                    b_size = data.num_graphs
                    self.optimD.zero_grad()
                    # Real data
                    if verbose:
                        print (1)
                    label.fill_(real_label)
                    if use_data_x is not None:
                        usex = data.x[:, :use_data_x]
                    else:
                        usex = data.x
                    if use_data_edgeattr is not None:
                        use_edge_attr = data.edge_attr[:, :use_data_edgeattr]
                    else:
                        use_edge_attr = data.edge_attr
                    probs_real = self.call_g(usex=usex, batch=data.batch, edge_index=data.edge_index, \
                                            edge_attr=use_edge_attr, edge_batch=data.edge_index_batch)
                    if count_eval:
                        real_rewards.extend(list(np.array(probs_real.detach().cpu().view(-1))))
                        real_counts.extend([data[jjj].x.size(0) for jjj in range(data.num_graphs)])
                    if self.use_loss == 'wgan':
                        loss_real = -(probs_real.mean())
                    elif self.use_loss == 'reward':
                        # [j[k] for k in range(j.num_graphs)]
                        real_ys = torch.FloatTensor([calculate_y_graph(data[k], node_dim=use_node_dim, without_aroma=self.without_ar) for k in range(data.num_graphs)])[:, self.reward_point:self.reward_point + 1]
                        # if self.reward_point == 2:
                        #     real_ys = 1./(real_ys+1e-7)
                        real_ys = real_ys.to(self.device)
                        loss_real = (torch.pow(probs_real - real_ys, 2).mean())
                    else:
                        loss_real = self.criterionD(probs_real, label)

                    loss_real.backward()

                    if self.use_loss != 'reward' or i % reward_fake_step == 0:
                        # If we are making objective (reward) target network, we only run fake data's every KK step.
                        
                        label.fill_(fake_label)
                        z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=self.rand_dim, device=self.device)
                        z = z_rand
                        if saveZ:
                            self.all_zs.append(('d', z, data))
                        if self.g_out_prob:
                            fake_data, prob = self.generator(z)
                        else:
                            fake_data = self.generator(z, obtain_connected=True)
                        probs_fake = self.call_g(fake_data)
                        if count_eval:
                            fake_rewards.extend(list(np.array(probs_fake.detach().cpu().view(-1))))
                            fake_counts.extend([fake_data[jjj].x.size(0) for jjj in range(fake_data.num_graphs)])
                        if self.use_loss == 'wgan':
                            loss_fake = (probs_fake.mean())
                        elif self.use_loss == 'reward':
                            # [j[k] for k in range(j.num_graphs)]
                            real_ys = torch.FloatTensor([calculate_y_graph(fake_data[k], node_dim=use_node_dim, without_aroma=self.without_ar) for \
                                        k in range(fake_data.num_graphs)])[:, self.reward_point:self.reward_point + 1]
                            # if self.reward_point == 2:
                            #     real_ys = 1./(real_ys+1e-7)
                            real_ys = real_ys.to(self.device)
                            loss_fake = (torch.pow(probs_fake - real_ys, 2).mean())
                        else:
                            loss_fake = self.criterionD(probs_fake, label)
                        loss_fake.backward()
                    else:
                        loss_fake = torch.FloatTensor([0])[0].to(self.device)

                    if self.lambda_g > 0:
                        if self.zinc or self.qm9:
                            penalty1 = calculate_gp_enriched(data, self.gcn_model, device=self.device)
                            penalty2 = calculate_gp_enriched(fake_data, self.gcn_model, device=self.device)
                        else:
                            penalty1 = calculate_gp(data, self.gcn_model, device=self.device)
                            penalty2 = calculate_gp(fake_data, self.gcn_model, device=self.device)
                        gp = (penalty1 + penalty2) * self.lambda_g
                        gp.backward()
                        D_loss = loss_real + loss_fake + gp
                    else:
                        gp = torch.zeros(1).mean()
                        D_loss = loss_real + loss_fake
                    if -loss_real.item() > loss_fake.item() + self.tresh_add_trainG:
                        # If D is too good in differentiating, we add more steps for G.
                        add_train_g = min(1+add_train_g, self.max_train_G)
                    else:
                        add_train_g = 1
                    # Update for D.
                    self.optimD.step()
                # torch.save(self.generator, os.path.join(self.folder, f'generator_stp_2.pt'))
                # torch.save(self.gcn_model, os.path.join(self.folder, f'gcn_model_stp_2.pt'))
                # Next is for G.
                if alter_trainer:
                    use_add_train_g = add_train_g*2
                else:
                    use_add_train_g = add_train_g
                for l in range(use_add_train_g):
                    self.optimG.zero_grad()
                    z_rand, z_lr_cont, z_cate, z_cont = generate_noise(rand_dim=self.rand_dim, device=self.device)
                    z = z_rand
                    if saveZ:
                        self.all_zs.append(('g', z))
                    if self.g_out_prob:
                        fake_data, prob = self.generator(z)
                    else:
                        fake_data = self.generator(z, obtain_connected=True)
                    probs_fake = self.call_g(fake_data)

                    label.fill_(real_label)
                    if self.use_loss == 'wgan':
                        gen_loss = -(probs_fake.mean())
                    elif self.use_loss == 'reward':
                        gen_loss = -(probs_fake).mean()
                    else:
                        gen_loss = self.criterionD(probs_fake, label)# -(probs_real.mean())
                    
                    if self.lambda_edge > 0 or self.lambda_node > 0:
                        useX, useA = self.generator.generate_XA(z)
                        if self.zinc:
                            gen_loss_nonodes = F.relu(useX[:, :, -1].sum(axis=1) - 16).mean()*self.lambda_node
                            gen_loss_edges = F.relu((1 - useX[:, :, -1]).sum(axis=1) - ((1 - useA[:, :, :, -1]).sum(axis=1).sum(axis=1) - 36)/2).mean()*self.lambda_edge
                        else:
                            gen_loss_nonodes = useX[:, :, -1].sum(axis=1).mean()*self.lambda_node
                            gen_loss_edges = F.relu((1 - useX[:, :, -1]).sum(axis=1) - ((1 - useA[:, :, :, -1]).sum(axis=1).sum(axis=1) - 9)/2).mean()*self.lambda_edge
                        gen_loss += gen_loss_nonodes + gen_loss_edges
                    if self.g_out_prob:
                        # If alter_train, we use half steps to train G based on RL
                        #                  another half steps to train G based on GAN's loss.
                        if alter_trainer and (l % 2 == 0) and (not rl_only):
                            G_loss = gen_loss
                        elif (alter_trainer and (l % 2 == 1)) or rl_only:
                            if reinforce_acclerate:
                                if useBN:
                                    usePF = self.reinf_bn(probs_fake)
                                    G_loss = - (prob * usePF.detach().view(-1)).mean() * self.lambda_rl
                                else:
                                    if acclerate_noscale:
                                        G_loss = - (prob * ((probs_fake - probs_fake.mean())).detach().view(-1)).mean() * self.lambda_rl
                                    else:
                                        G_loss = - (prob * ((probs_fake - probs_fake.mean())/(probs_fake.std() + 1e-3)).detach().view(-1)).mean() * self.lambda_rl
                            else:
                                G_loss = - (prob * probs_fake.detach().view(-1)).mean() * self.lambda_rl
                        else:
                            if reinforce_acclerate:
                                if useBN:
                                    usePF = self.reinf_bn(probs_fake)
                                    G_loss = gen_loss - (prob * usePF.detach().view(-1)).mean() * self.lambda_rl
                                else:
                                    if acclerate_noscale:
                                        G_loss = gen_loss - (prob * ((probs_fake - probs_fake.mean())).detach().view(-1)).mean() * self.lambda_rl
                                    else:
                                        G_loss = gen_loss - (prob * ((probs_fake - probs_fake.mean())/(probs_fake.std() + 1e-3)).detach().view(-1)).mean() * self.lambda_rl
                            else:
                                G_loss = gen_loss \
                                        - (prob * probs_fake.detach().view(-1)).mean() * self.lambda_rl
                    else:
                        G_loss = gen_loss
                    G_loss.backward()
                    # Update for G.
                    self.optimG.step()
                    # torch.save(self.generator, os.path.join(self.folder, f'generator_stp_3_{l}.pt'))
                    # torch.save(self.gcn_model, os.path.join(self.folder, f'gcn_model_stp_3_{l}.pt'))
                if trainD:
                    self.temp_loss_D.append(D_loss.item())
                else:
                    D_loss = torch.sum(torch.FloatTensor([0, 0]))
                    gp = torch.sum(torch.FloatTensor([0, 0]))
                    probs_real = torch.FloatTensor([0, 0])
                self.temp_loss_G.append(G_loss.item())
                if ii % 100 == 0:
                    # print out some information during training.
                    if save_images is not None:
                        self.plot(save_images, ii, **plot_para)
                    print(('[%d/%d][%d/%d]\t' 
                        'G Loss: %.4f;'
                        'D Loss: %.4f; GP: %.4f')
                        % (epoch+1, 200, i, len(data_loader), 
                            gen_loss.item(), \
                            D_loss.item() - gp.item(), gp.item()))
                    print ('now, we train G %d times with (prob fake = %.3f, prob real = %.3f)' % (add_train_g, \
                                                                                                    probs_fake.mean(), \
                                                                                                    probs_real.mean()))
                    print ('Mean x/edge attr: ', fake_data.x.mean(axis=0), fake_data.edge_attr.mean(axis=0), fake_data.edge_attr.sum(axis=0).sum()/self.batch_size)
                    if self.use_loss == 'reward':
                        with torch.no_grad():
                            real_ys = torch.FloatTensor([calculate_y_graph(fake_data[k], node_dim=use_node_dim, without_aroma=self.without_ar) for \
                                        k in range(fake_data.num_graphs)])[:, self.reward_point:self.reward_point + 1]
                            real_ys = real_ys.to(self.device)
                            print ("Property in fake data:", (real_ys > 0).sum(), real_ys[real_ys > 0].mean())
                    if count_eval:
                        rewards = np.array(real_rewards)
                        counts = np.array(real_counts)
                        print (f'REAL: Shape in 6: {(counts == 6).sum()}; Shape in 7: {(counts == 7).sum()}; Shape in 8: {(counts == 8).sum()}; Shape in 9: {(counts == 9).sum()}.')
                        print (f'REAL: Average reward in 6: {rewards[counts == 6].mean()}; Average reward in 7: {rewards[counts == 7].mean()}; Average reward in 8: {rewards[counts == 8].mean()}; Average reward in 9: {rewards[counts == 9].mean()}')
                        rewards = np.array(fake_rewards)
                        counts = np.array(fake_counts)
                        print (f'FAKE: Shape in 6: {(counts == 6).sum()}; Shape in 7: {(counts == 7).sum()}; Shape in 8: {(counts == 8).sum()}; Shape in 9: {(counts == 9).sum()}.')
                        print (f'FAKE: Average reward in 6: {rewards[counts == 6].mean()}; Average reward in 7: {rewards[counts == 7].mean()}; Average reward in 8: {rewards[counts == 8].mean()}; Average reward in 9: {rewards[counts == 9].mean()}')
                        real_rewards = []
                        real_counts = []
                        fake_rewards = []
                        fake_counts = []
                    if self.zinc or self.qm9:
                        print ('size x/ some distribution:', len(fake_data.batch)/self.batch_size, global_add_pool(torch.ones(len(fake_data.batch)).to(fake_data.batch.device), fake_data.batch))
                    if self.g_out_prob:
                        print ("Sample prob:", prob.mean().item())
                if ii % NN == 0:
                    # Evaluate generator and save models.
                    eval_fake_data, eval_res = self.eval(evaluate_num, mol_data)
                    if self.use_loss == 'reward':
                        with torch.no_grad():
                            # real_ys = torch.FloatTensor([calculate_y_graph(fake_data[k], node_dim=use_node_dim, without_aroma=self.without_ar) for \
                            #             k in range(fake_data.num_graphs)])[:, self.reward_point:self.reward_point + 1]
                            real_ys = torch.FloatTensor([calculate_y_graph(eval_fake_data[k], molType=True, node_dim=use_node_dim, without_aroma=self.without_ar) for \
                                        k in range(len(eval_fake_data))])[:, self.reward_point:self.reward_point + 1]
                            # eval_fake_data
                            # if self.reward_point == 2:
                            #     real_ys = 1./(real_ys+1e-7)
                            real_ys = real_ys.to(self.device)
                            print ('Current reward: ', real_ys[real_ys>0].mean().item())
                        eval_res = list(eval_res)
                        eval_res.append(real_ys[real_ys>0].mean().item())
                        eval_res = tuple(eval_res)
                    self.evals.append(eval_res)
                    torch.save(self.generator, os.path.join(self.folder, f'generator_{kk}.pt'))
                    torch.save(self.gcn_model, os.path.join(self.folder, f'gcn_model_{kk}.pt'))
                    self.all_zs = []
                    kk += 1

                ii += 1

    def plot(self, save_images, ii, without_aroma=True, enriched_ver=True):
        """Make plots.

        Args:
            save_images (string): folder to store plots
            ii (int): number of epoch
            without_aroma (bool, optional): If don't use aroma. Defaults to True.
            enriched_ver (bool, optional): If we use enricher version. Defaults to True.
        """
        if not os.path.exists(save_images):
            os.mkdir(save_images)
        self.generator.eval()
        evaluate_num = 64
        with torch.no_grad():
            eval_fake_data = []
            problem_info = []
            z_rand, z_lr_cont, z_cate, z_cont = generate_noise(device=self.device, rand_dim=self.rand_dim)
            z = z_rand
            fake_data, probs = self.generator(z)
            eval_fake_data.extend([fake_data[j] for j in range(fake_data.batch.max().item() + 1)])
            if enriched_ver:
                eval_fake_data = [MolFromTorchGraphData_enriched(j.to('cpu'), without_aroma=without_aroma) for j in eval_fake_data]
            else:
                eval_fake_data = [MolFromTorchGraphData(j.to('cpu')) for j in eval_fake_data]                

            eval_valid_data = [Chem.MolFromSmiles(Chem.MolToSmiles(j)) for j in eval_fake_data if Chem.MolFromSmiles(Chem.MolToSmiles(j)) is not None]
            print ([Chem.MolToSmiles(j) for j in eval_fake_data[:4]])
            print ([Chem.MolToSmiles(j) for j in eval_valid_data[:4]])
            for j in range(min(25, len(eval_valid_data))):
                fig = Draw.MolToImage(eval_valid_data[j])
                fig.save(os.path.join(save_images, 'Fake_graph_%d_epoch%d.png' % (ii, j)))

        self.generator.train()



    def eval(self, eval_num, mol_data):
        """Make evaluation, 1. generate fake data, 2. evaluate on metrics. 3. store it.

        Args:
            eval_num (int): number of fake data to generate.
            mol_data (list): list of smiles string as training data. used to obtain novelty.

        Returns:
            a list of fake data (in Chem.Mol)
            tuple of evaluated metrics.
        """
        self.generator.eval()
        if hasattr(self, 'eval_counts') and self.eval_counts:
            self.gcn_model.eval()
            if self.gcn_add is not None:
                self.gcn_add.eval()
            count_eval = True
        else:
            count_eval = False
        with torch.no_grad():
            eval_fake_data = []
            problem_info = []
            rewards = []
            counts = []
            for j in range(eval_num//self.batch_size + 1):
                z_rand, z_lr_cont, z_cate, z_cont = generate_noise(device=self.device, rand_dim=self.rand_dim)
                z = z_rand
                if self.g_out_prob:
                    fake_data, probs = self.generator(z)
                else:
                    fake_data = self.generator(z, obtain_connected=True)
                if count_eval:
                    rewards.extend(list(np.array(self.gcn_model(fake_data).detach().cpu())))
                    counts.extend([fake_data[j].x.size(0) for j in range(fake_data.num_graphs)])

                fake_data = convert_Batch_to_datalist(x=fake_data.x, 
                                                    edge_index=fake_data.edge_index, \
                                                    edge_attr=fake_data.edge_attr, \
                                                    batch=fake_data.batch, \
                                                    edge_batch=fake_data.edge_index_batch)
                if np.sum([j.x.isnan().sum().item() for j in fake_data]) > 0:
                    print ("Problem here")
                eval_fake_data.extend(fake_data)
            if hasattr(self, 'zinc') and self.zinc:
                eval_fake_data = [MolFromTorchGraphData_enriched(j.to('cpu'), without_aroma=self.without_ar) for j in eval_fake_data][:eval_num]
            elif hasattr(self, 'qm9') and self.qm9:
                eval_fake_data = [MolFromTorchGraphData_enriched(j.to('cpu'), node_dim=4, without_aroma=self.without_ar) for j in eval_fake_data][:eval_num]
            else:
                eval_fake_data = [MolFromTorchGraphData(j.to('cpu')) for j in eval_fake_data][:eval_num]
            if count_eval:

                rewards = np.array(rewards)
                counts = np.array(counts)
                print (f'Shape in 6: {(counts == 6).sum()}; Shape in 7: {(counts == 7).sum()}; Shape in 8: {(counts == 8).sum()}; Shape in 9: {(counts == 9).sum()}.')
                print (f'Average reward in 6: {rewards[counts == 6].mean()}; Average reward in 7: {rewards[counts == 7].mean()}; Average reward in 8: {rewards[counts == 8].mean()}; Average reward in 9: {rewards[counts == 9].mean()}')

            eval_res = evaluate(eval_fake_data, mol_data)
            print ("Validation, uniqueness, novelty: ", eval_res) 
        self.generator.train()
        if hasattr(self, 'eval_counts') and self.eval_counts:
            self.gcn_model.train()
            if self.gcn_add is not None:
                self.gcn_add.train()
        return eval_fake_data, eval_res




