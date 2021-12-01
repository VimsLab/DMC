"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh
Copyright (c) VIMS Lab and its affiliates
This file contains miscellaneous function/s pertaining to files paths
"""
import os
import os.path as osp

def get_file_paths(dir_root="", file_type=".obj"):
    """
    Get all file paths in a directory belonging to a particular file type.

    Args:
        - dir_root: str, a existing directory on the system.
        - file_type: str, a file type, e.g.: .obj, .npz.

    Returns:
        - file_paths: (n), file paths in dir_root belonging to file_type.
    """
    if not osp.exists(dir_root):
        raise ValueError('Directory does NOT exist!')

    if not osp.isdir(dir_root):
        raise ValueError('NOT a directory!')

    if file_type not in ['.obj', '.npz']:
        raise ValueError('Invalid file type! '
                         'Valid file_type are: .obj and .npz!')

    file_paths = []
    for root, dirs, files in os.walk(dir_root, topdown=False):
        for file_name in files:
            if file_name.endswith(file_type):
                if os.path.exists(os.path.join(root, file_name)):
                    file_paths.append(os.path.join(root, file_name))

    file_paths = sorted(file_paths)

    return file_paths
