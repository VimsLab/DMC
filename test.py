import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelNet40
from models import MeshNet
from utils import append_feature, calculate_map
import random

dataset = 'ModelNet40'
cfg = get_test_config(dataset=dataset)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


data_set = ModelNet40(cfg=cfg[dataset], part='test')
data_loader = data.DataLoader(data_set, batch_size=1, num_workers=4, shuffle=True, pin_memory=False)

def test_model(model):

    correct_num = 0
    ft_all, lbl_all = None, None
    for i, (centers, corners, normals, neighbors, rings, targets) in enumerate(data_loader):
        corners = corners - torch.cat([centers, centers, centers], 1)
        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        for idx, ring in enumerate(rings):
            rings[idx] = Variable(torch.cuda.LongTensor(ring.cuda()))

        targets = Variable(torch.cuda.LongTensor(targets.cuda()))

        outputs, feas = model(centers, corners, normals, neighbors, rings)
        _, preds = torch.max(outputs, 1)

        if preds[0] == targets[0]:
            correct_num += 1

        ft_all = append_feature(ft_all, feas.detach())
        lbl_all = append_feature(lbl_all, targets.detach(), flaten=True)

    print('Accuracy: {:.4f}'.format(float(correct_num) / len(data_set)))
    print('mAP: {:.4f}'.format(calculate_map(ft_all, lbl_all)))


if __name__ == '__main__':

    os.environ['PYTHONHASHSEED'] = str(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg[dataset]['load_model']))
    model.eval()

    test_model(model)
