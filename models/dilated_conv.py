"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the implementation logic and details of the Stacked Dilated
Mesh Convolution (SDMC) block.
"""
import math
import torch
import torch.nn as nn

class SDMC(nn.Module):
    """
    This class provides functions to perform mesh convolutions on the ring
    neighbors with different dilation rate and stack them to obtain a multi-scale
    response.
    """
    def __init__(self,
                 spatial_in_channel,
                 structural_in_channel,
                 spatial_out_channel,
                 structural_out_channel,
                ):
        """
        Args:
            - spatial_in_channel: int, number of input channels in the spatial features.

            - structural_in_channel: int, number of input channels in the structural features.

            - spatial_out_channel: int, number of output channels in the spatial features.

            - structural_out_channel: int, number of output channels in the structural features.
        """
        super(SDMC, self).__init__()

        self.spatial_in_channel = spatial_in_channel
        self.spatial_out_channel = spatial_out_channel
        self.structural_in_channel = structural_in_channel
        self.structural_out_channel = structural_out_channel

        # Combination MLP is required to maintain the MeshNet design
        self.combination_mlp = nn.Sequential(
            nn.Conv1d(self.spatial_in_channel + self.structural_in_channel,
                      self.spatial_out_channel, 1),
            nn.BatchNorm1d(self.spatial_out_channel),
            nn.ReLU(),
        )

        # MLP for Δ
        self.concat_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel = self.structural_in_channel

        # MLP for Δ1
        self.ring_1st_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel += self.structural_in_channel

        # MLP for Δ2
        self.ring_2nd_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel += self.structural_in_channel

        # MLP for Δ3
        self.ring_3rd_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel += self.structural_in_channel

        # MLP for Cat(Δ1, Δ2, and Δ3)
        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.aggregation_in_channel, self.structural_out_channel, 1),
            nn.BatchNorm1d(self.structural_out_channel),
            nn.ReLU(),
        )

    def DMC(self, structural_fea, ring, ring_mlp, s):
        """
        In this function, structural features in ring neighbors with a specific
        dilation rate are passed through an MLP. This process is referred to as
        Dilated Mesh Convolution (DMC) in the paper. Features refined through
        DMC are then max pooled.

        Args:
            - structural_fea: (num_meshes, structural_in_channel, max_faces),
                              MeshNet's structural features.

            - ring: (num_meshes, max_faces, ?), a ring neighbors around faces'

            - ring_mlp: MLP for the a ring neighbors around faces'

            - s: float, ring subsampling rate.

        Returns:
            - structural_fea_ring: (num_meshes, structural_in_channel, max_faces),
                                   refined structural features after DMC.
        """

        # Sub-sample faces in Δr to produce Δ'r
        num_meshes, num_features, max_faces = structural_fea.size()
        s = 1 - s
        sample_idx = torch.randperm(ring.shape[2])[:math.ceil(ring.shape[2]*s)]
        ring = ring[:, :, sample_idx]
        ring = ring.unsqueeze(1)
        ring = ring.expand(num_meshes, num_features, max_faces, -1)

        # For all meshes in a batch, sample features from Δ'r around each face fi
        # This is refered to as "Ring Sampling" in the paper.
        structural_fea_ring = structural_fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_features)[None, :, None, None],
            ring
        ]

        # Δ'r features are passed through an MLP to obtain A(Δ'r)
        structural_fea_ring = ring_mlp(structural_fea_ring)

        # Ap(Δ'r) as given in equation 5 in the paper
        structural_fea_ring = torch.max(structural_fea_ring, 3)[0]

        return structural_fea_ring

    def forward(self, spatial_fea, structural_fea, rings):
        """
        Below, num_meshes is the number of meshes in a batch and max_faces
        are the number of faces per mesh.

        Args:
            - spatial_fea: (num_meshes, spatial_in_channel, max_faces),
                           MeshNet's spatial features.

            - structural_fea: (num_meshes, structural_in_channel, max_faces),
                              MeshNet's structural features.

            - rings: list, list of ring neighbor(s) around faces'

        Returns:
            - spatial_fea: (num_meshes, spatial_out_channel, max_faces), refined
                           spatial features obtained through SDMC.

            - structural_fea: (num_meshes, structural_out_channel, max_faces),
                              refined spatial features obtained through SDMC.
        """
        # Combination MLP
        spatial_fea = self.combination_mlp(torch.cat([spatial_fea, structural_fea], 1))
        # structural_fea_copy = structural_fea

        ################################## Δ ###################################
        structural_fea = structural_fea.unsqueeze(3)
        structural_fea = self.concat_mlp(structural_fea)
        structural_fea = torch.max(structural_fea, 3)[0]

        ################################## Δr ##################################
        ring_1st, ring_2nd, ring_3rd = rings
        structural_fea_ring_1st = self.DMC(structural_fea,
                                           ring_1st,
                                           self.ring_1st_mlp,
                                           s=1/3)

        structural_fea_ring_2nd = self.DMC(structural_fea,
                                           ring_2nd,
                                           self.ring_2nd_mlp,
                                           s=1/3)

        structural_fea_ring_3rd = self.DMC(structural_fea,
                                           ring_3rd,
                                           self.ring_3rd_mlp,
                                           s=1/3)

        # Concatenated structural features of different ring neighbor
        # as per in equation 6 in the paper
        structural_fea = torch.cat([structural_fea,
                                    structural_fea_ring_1st,
                                    structural_fea_ring_2nd,
                                    structural_fea_ring_3rd], 1)

        structural_fea = self.aggregation_mlp(structural_fea)

        return spatial_fea, structural_fea
