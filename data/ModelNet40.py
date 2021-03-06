"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file contains the data loader for the ModelNet40 data set.
"""
import numpy as np
import os
import torch
import torch.utils.data as data

type_to_index_map = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39
}

class ModelNet40(data.Dataset):
    """ This class contains functions to load the training set and the test set
    of the pre-processed ModelNet40 dataset in the dataset/processed directory.
    """
    def __init__(self, cfg, part='train'):
        """
        Args:
            - cfg: dict, configuration details of the pre-processed ModelNet40
                   data set to to load it from the disk correctly.

            - part: str, the training set or the testing set of ModelNet40
        """
        # directory where the pre-processed ModelNet40 dataset is stored on disk
        self.root = cfg['processed']
        # boolean flag in the case of data augmentations are to be performed
        # Note: No data augmentations are performed for MeshNet + SDMC
        self.augment_data = cfg['augment_data']
        # The number of faces needs to be the same for training
        # MeshNet and its variants with a batch size greater than one.
        self.max_faces = cfg['max_faces']
        # the training set or the testing set of ModelNet40
        self.part = part
        # Read data from directory
        self.data = []
        for type in os.listdir(self.root):
            type_index = type_to_index_map[type]
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append((os.path.join(type_root, filename), type_index))

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)

        # Ideally, the face variable would be named face_features but is still
        # named as face to conform to the original MeshNet code
        face = data['face_features']

        ring_1st = data['ring_1st']
        ring_1st = torch.from_numpy(ring_1st).long()
        assert ring_1st.shape == (self.max_faces, 3)

        ring_2nd = data['ring_2nd']
        ring_2nd = torch.from_numpy(ring_2nd).long()
        assert ring_2nd.shape == (self.max_faces, 6)

        ring_3rd = data['ring_3rd']
        ring_3rd = torch.from_numpy(ring_3rd).long()
        assert ring_3rd.shape == (self.max_faces, 12)

        # to tensor
        face = torch.from_numpy(face).float()

        target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()

        centers, corners, normals = face[:3], face[3:12], face[12:]

        neighbors = ring_1st
        rings = [ring_1st, ring_2nd, ring_3rd]
        return centers, corners, normals, neighbors, rings, target

    def __len__(self):
        return len(self.data)
