"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
We adapt MeshNet to perform dilated convolutions by replacing the Stacked Dilated
Mesh Convolution (SDMC) block in place of its Mesh Convolution (MC) block.
This file trains this redesigned MeshNet model.
Note: For the ease of exposition and to keep this file coherent with the train.py
in the original MeshNet code, we do not add code comments to this file.
"""
import copy
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import ModelNet40
from models import MeshNet

dataset = 'ModelNet40'
cfg = get_train_config(dataset=dataset)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']

data_set = {
    x: ModelNet40(cfg=cfg[dataset], part=x) for x in ['train', 'test']
}

data_loader = {
    x: data.DataLoader(data_set[x],
                       batch_size=cfg['batch_size'],
                       num_workers=4,
                       shuffle=True,
                       pin_memory=False)
    for x in ['train', 'test']
}

def train_model(model, criterion, optimizer, scheduler, cfg):

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        for phrase in ['train', 'test']:

            if phrase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for (centers, corners, normals, neighbors, rings, targets) in data_loader[phrase]:

                optimizer.zero_grad()

                centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
                corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
                normals = Variable(torch.cuda.FloatTensor(normals.cuda()))

                for idx, ring in enumerate(rings):
                    rings[idx] = Variable(torch.cuda.LongTensor(ring.cuda()))

                targets = Variable(torch.cuda.LongTensor(targets.cuda()))

                with torch.set_grad_enabled(phrase == 'train'):
                    corners = corners - torch.cat([centers, centers, centers], 1)
                    outputs, _ = model(centers, corners, normals, neighbors, rings)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    if phrase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * centers.size(0)
                    running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / len(data_set[phrase])
            epoch_acc = running_corrects.double() / len(data_set[phrase])

            if phrase == 'train':
                scheduler.step()
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))

            if phrase == 'test':
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                if epoch % 10 == 0:
                    torch.save(copy.deepcopy(model.state_dict()),
                               cfg[dataset]['ckpt_root'] + '{}.pkl'.format(epoch))

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))

    return best_model_wts


if __name__ == '__main__':

    model_ft = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model_ft.cuda()
    model_ft = nn.DataParallel(model_ft)

    criterion_ft = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(),
                             lr=cfg['lr'],
                             momentum=cfg['momentum'],
                             weight_decay=cfg['weight_decay'])

    scheduler_ft = optim.lr_scheduler.MultiStepLR(optimizer_ft,
                                                  milestones=cfg['milestones'],
                                                  gamma=cfg['gamma'])

    best_model_wts = train_model(model=model_ft,
                                 criterion=criterion_ft,
                                 optimizer=optimizer_ft,
                                 scheduler=scheduler_ft,
                                 cfg=cfg)

    torch.save(best_model_wts, os.path.join(cfg[dataset]['ckpt_root'], 'MeshNet_best.pkl'))
