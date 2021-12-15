"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
We adapt MeshNet to perform dilated convolutions by replacing the Stacked Dilated
Mesh Convolution block in place of its Mesh Convolution block.
This file test this redesigned model after training.
Note: For the ease of exposition and to keep this file coherent with the train.py
in the original MeshNet code, we do not add code comments to this file.
"""
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelNet40
from models import MeshNet

dataset = 'ModelNet40'
cfg = get_test_config(dataset=dataset)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']

data_set = ModelNet40(cfg=cfg[dataset], part='test')
data_loader = data.DataLoader(data_set,
                              batch_size=1,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=False)

def test_model(model):
    correct_num = 0
    for (centers, corners, normals, neighbors, rings, targets) in data_loader:
        corners = corners - torch.cat([centers, centers, centers], 1)
        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))

        for idx, ring in enumerate(rings):
            rings[idx] = Variable(torch.cuda.LongTensor(ring.cuda()))

        targets = Variable(torch.cuda.LongTensor(targets.cuda()))

        outputs, _ = model(centers, corners, normals, neighbors, rings)
        _, preds = torch.max(outputs, 1)

        if preds[0] == targets[0]:
            correct_num += 1

    print('Accuracy: {:.4f}'.format(float(correct_num) / len(data_set)))


if __name__ == '__main__':

    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(False)

    model_ft = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model_ft.cuda()
    model_ft = nn.DataParallel(model_ft)
    model_ft.load_state_dict(torch.load(cfg[dataset]['load_model']))
    model_ft.eval()
    test_model(model_ft)
