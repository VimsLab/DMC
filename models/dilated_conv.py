import math
import torch
import torch.nn as nn

class SDMC(nn.Module):
    def __init__(self,
                 spatial_in_channel,
                 structural_in_channel,
                 spatial_out_channel,
                 structural_out_channel,
                ):

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

        self.concat_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel = self.structural_in_channel

        self.ring_1st_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel += self.structural_in_channel

        self.ring_2nd_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel += self.structural_in_channel

        self.ring_3rd_mlp = nn.Sequential(
            nn.Conv2d(self.structural_in_channel, self.structural_in_channel, 1),
            nn.BatchNorm2d(self.structural_in_channel),
            nn.ReLU(),
        )
        self.aggregation_in_channel += self.structural_in_channel

        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.aggregation_in_channel, self.structural_out_channel, 1),
            nn.BatchNorm1d(self.structural_out_channel),
            nn.ReLU(),
        )

    def DMC(self, structural_fea, ring, ring_mlp, subsampling_rate):
        # For all meshes in a batch, sample features from Δ1 around each face fi
        # This is refered to as "Ring Sampling" in the paper
        num_meshes, num_features, max_faces = structural_fea.size()
        subsampling_rate = 1 - subsampling_rate
        sample_idx = torch.randperm(ring.shape[2])[:math.ceil(ring.shape[2]*subsampling_rate)]
        ring = ring[:, :, sample_idx]
        ring = ring.unsqueeze(1)
        ring = ring.expand(num_meshes, num_features, max_faces, -1)
        structural_fea_ring = structural_fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_features)[None, :, None, None],
            ring
        ]
        # Refining sampled features
        structural_fea_ring = ring_mlp(structural_fea_ring)

        # AP(Δ'r) as given in equation 5 in the paper
        structural_fea_ring = torch.max(structural_fea_ring, 3)[0]

        return structural_fea_ring

    def forward(self, spatial_fea, structural_fea, rings):
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
                                           subsampling_rate=1/3)

        structural_fea_ring_2nd = self.DMC(structural_fea,
                                           ring_2nd,
                                           self.ring_2nd_mlp,
                                           subsampling_rate=1/3)

        structural_fea_ring_3rd = self.DMC(structural_fea,
                                           ring_3rd,
                                           self.ring_3rd_mlp,
                                           subsampling_rate=1/3)

        structural_fea = torch.cat([structural_fea,
                                    structural_fea_ring_1st,
                                    structural_fea_ring_2nd,
                                    structural_fea_ring_3rd], 1)

        # Concatenated Structural features passed through MLP
        structural_fea = self.aggregation_mlp(structural_fea)

        return spatial_fea, structural_fea
