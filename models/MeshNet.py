"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the MeshNet + SDMC model.
Note: For the ease of exposition and to keep this file coherent with the train.py
in the original MeshNet code, we do not add code comments to this file.
"""
import torch
import torch.nn as nn
from models import SpatialDescriptor, StructuralDescriptor, SDMC

class MeshNet(nn.Module):

    def __init__(self, cfg, require_fea=False):
        super(MeshNet, self).__init__()
        self.require_fea = require_fea

        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(cfg['structural_descriptor'])
        # Replaced the Mesh Convolution block in MeshNet with SDMC
        self.mesh_conv1 = SDMC(64, 131, 256, 256)
        self.mesh_conv2 = SDMC(256, 256, 512, 512)

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 40)
        )

    def forward(self, centers, corners, normals, neighbors, rings):
        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbors)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, rings)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, rings)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        fea = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))
        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        fea = self.classifier[:-1](fea)
        cls = self.classifier[-1:](fea)

        if self.require_fea:
            return cls, fea / torch.norm(fea)
        else:
            return cls
