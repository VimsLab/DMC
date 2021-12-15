"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh.
Copyright (c) VIMS Lab and its affiliates.
This file loads and pre-processes the .obj files in the "raw" dataset.
Pre-processing is performed to derive mesh attributes (e.g., faces, rings, etc.)
from the "raw" mesh. Derived attributes are saved as .npz files.
"""
import sys
import os
import os.path as osp
from tqdm import tqdm
from numpy import concatenate as cat
from numpy import savez
from config import get_train_config
from utils import get_file_paths, Mesh

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Use: python preprocess.py <dataset>')
        print('<dataset> can be one of the following: ')
        print('ModelNet40')
        print('For example:')
        print('python preprocess.py ModelNet40')
        exit(0)
    elif len(sys.argv) == 2:
        if str(sys.argv[1]) not in ['ModelNet40']:
            raise ValueError('Invalid <dataset>!'
                             'Valid datasets are: ModelNet40.')
        else:
            dataset = str(sys.argv[1])

    cfg = get_train_config('./config/train_config.yaml', dataset)
    max_faces = cfg[dataset]['max_faces']

    # All file paths with the .obj extension in ".datasets/raw/dataset".
    file_paths_raw = get_file_paths(dir_root=cfg[dataset]["raw"],
                                    file_type=".obj")

    print('Pre-processing dataset...')
    # Get and save mesh attributes required by MeshNet + SDMC.
    for file_path_raw in tqdm(file_paths_raw):
        mesh = Mesh()
        # Load the mesh from file path.
        mesh.load(file_path=file_path_raw)
        # Normalize the mesh.
        mesh.normalize()

        # Get the mesh attributes.
        verts = mesh.get_verts()
        faces = mesh.get_faces()
        centers = mesh.get_face_centers()
        corners = mesh.get_face_corners()
        corners = corners.reshape(-1, 9)
        normals = mesh.get_face_normals()
        ring_1st = mesh.get_face_neighbors(ring_name="1st Ring")
        ring_2nd = mesh.get_face_neighbors(ring_name="2nd Ring")
        ring_3rd = mesh.get_face_neighbors(ring_name="3rd Ring")

        # Check the shape of the mesh attributes.
        # The shape of all the attributes needs to be the same for training
        # MeshNet and its variants with a batch size greater than one.
        assert faces.shape == (max_faces, 3)
        assert centers.shape == (max_faces, 3)
        assert corners.shape == (max_faces, 9)
        assert normals.shape == (max_faces, 3)
        assert ring_1st.shape == (max_faces, 3)
        assert ring_2nd.shape == (max_faces, 6)
        assert ring_3rd.shape == (max_faces, 12)

        faces_features = cat([centers, corners, normals], axis=1)

        # Save mesh attributes in ".datasets/processed/dataset".
        file_path_processed = file_path_raw.replace(cfg[dataset]["raw"],
                                                    cfg[dataset]["processed"])
        dir_processed = file_path_processed[:file_path_processed.rindex(os.sep)]
        if not osp.exists(dir_processed):
            os.makedirs(dir_processed)
        file_path_processed = file_path_processed.replace('.obj', '.npz')
        savez(file_path_processed,
              verts=verts,
              faces=faces,
              faces_features=faces_features,
              ring_1st=ring_1st,
              ring_2nd=ring_2nd,
              ring_3rd=ring_3rd)
    print('Success!')
