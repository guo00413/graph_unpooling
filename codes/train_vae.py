from torch import optim
import torch
def train(dataloader, model, max_epoch, debug=False):
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=1e-3)

    model.train()
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            
            loss = model(data.x, data.edge_attr, data.edge_index, data.batch, data.edge_index_batch, debug=debug)
            if debug:
                print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss adj: ', loss[0].mean()\
                    , ', Loss E: ', loss[1].mean(), ', Loss X: ', loss[2].mean(), ', Loss KL: ', loss[3].mean()\
                        , ', Loss: ', (loss[0] + loss[1] + loss[2] + loss[3]).mean())
                loss = loss[0] + loss[1] + loss[2] + loss[3]
            else:
                print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss.mean())
            loss.mean().backward()

            optimizer.step()

def eval(decoder, ):
    pass


def train_ul(dataloader, model, max_epoch, device='cpu', debug=False, separate_update=False, verbose=False, \
    moving_ave=False, moving_mom=None, adj_coef=1, lr=1e-3):
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=lr)
    model.train()
    if not hasattr(model, 'losses'):
        model.losses = []
    moving_ave_use = None
    if moving_mom is None:
        moving_mom = 0.9
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            data = data.to(device)
            loss, prob = model(data.x, data.edge_attr, data.edge_index, data.batch, data.edge_index_batch, debug=debug, verbose=verbose)
            if debug:
                print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss adj: ', loss[0].mean()\
                    , ', Loss E: ', loss[1].mean(), ', Loss X: ', loss[2].mean(), ', Loss KL: ', loss[3].mean()\
                        , ', Loss: ', (loss[0] + loss[1] + loss[2] + loss[3]).mean())
                model.losses.append((loss[0] + loss[1] + loss[2] + loss[3]).mean().item())
                if separate_update:
                    if verbose:
                        print (loss[0], prob)
                        print ('!=============================!')
                    if moving_ave:
                        if moving_ave_use is None:
                            moving_ave_use = loss[0].detach().mean()
                        else:
                            moving_ave_use = moving_mom*moving_ave_use + (1 - moving_mom)*loss[0].detach().mean()
                    else:
                        moving_ave_use = loss[0].detach().mean()
                    ((loss[1] + loss[2] + loss[3]).mean() + adj_coef*((loss[0].detach() - moving_ave_use)*prob).mean()).backward()
                else:
                    loss = loss[0] + loss[1] + loss[2] + loss[3]
                    (loss.mean() + ((loss.detach() - loss.detach().mean())*prob).mean()).backward()
            else:
                print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss.mean())
                (loss.mean() + ((loss.detach() - loss.detach().mean())*prob).mean()).backward()
                model.losses.append(loss.mean().item())
            optimizer.step()
            # model.zero_grad()
            # loss, prob = model(data.x, data.edge_attr, data.edge_index, data.batch, data.edge_index_batch)
            # (loss.detach()*prob).mean().backward()
            # optimizer_rl.step()


def train_ul_v2(dataloader, model, max_epoch, device='cpu', debug=False, separate_update=False, verbose=False, \
    moving_ave=False, moving_mom=None, adj_coef=1):
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=1e-3)
    optimizer_RL = optim.Adam([j for j in model.decoder.convs.parameters()] + [j for j in model.decoder.unpools.parameters()] + [j for j in model.decoder.skipzs.parameters()], lr=1e-3*adj_coef)
    model.train()
    moving_ave_use = None
    if moving_mom is None:
        moving_mom = 0.9
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            data = data.to(device)
            loss, prob = model(data.x, data.edge_attr, data.edge_index, data.batch, data.edge_index_batch, debug=debug)
            if debug:
                print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss adj: ', loss[0].mean()\
                    , ', Loss E: ', loss[1].mean(), ', Loss X: ', loss[2].mean(), ', Loss KL: ', loss[3].mean()\
                        , ', Loss: ', (loss[0] + loss[1] + loss[2] + loss[3]).mean())
                if separate_update:
                    if verbose:
                        print (loss[0], prob)
                        print ('!=============================!')
                    if moving_ave:
                        if moving_ave_use is None:
                            moving_ave_use = loss[0].detach().mean()
                        else:
                            moving_ave_use = moving_mom*moving_ave_use + (1 - moving_mom)*loss[0].detach().mean()
                    else:
                        moving_ave_use = loss[0].detach().mean()
                    ((loss[1] + loss[2] + loss[3]).mean()).backward()
                    optimizer.step()
                    model.zero_grad()
                    loss, prob = model(data.x, data.edge_attr, data.edge_index, data.batch, data.edge_index_batch, debug=debug)
                    (((loss[0].detach() - moving_ave_use)*prob).mean()).backward()
                    optimizer_RL.step()
                else:
                    loss = loss[0] + loss[1] + loss[2] + loss[3]
                    (loss.mean() + ((loss.detach() - loss.detach().mean())*prob).mean()).backward()
            else:
                print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss.mean())
                (loss.mean() + ((loss.detach() - loss.detach().mean())*prob).mean()).backward()

            # optimizer.step()
            # model.zero_grad()
            # loss, prob = model(data.x, data.edge_attr, data.edge_index, data.batch, data.edge_index_batch)
            # (loss.detach()*prob).mean().backward()
            # optimizer_rl.step()
            

