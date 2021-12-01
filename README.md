# DMC
Official PyTorch Implementation for [Mesh Classification With Dilated Mesh Convolutions](https://ieeexplore.ieee.org/document/9506311).


## Download data set
Download the pre-processed ModelNet40 data set from [here](https://drive.google.com/drive/folders/1y-8m-GRErxCMkuJJf6t8yYSHztlUO0xF?usp=sharing). Files are in the OBJ file format, and all mesh model consists of precisely 1024 faces. 

## Setup Environment
Install the Anaconda package management platform if you haven't already.
1. Install tools on Ubuntu
sudo apt install curl

2. Install Anaconda Python Distribution on Ubuntu
```
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sha256sum Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
conda list

Notes:
If sha256sum command prints anything besides 2751ab3d678ff0277ae80f9e8a74f218cfc70fe9a9cdc7bb1c137d7e47e33d53, either the installer was not downloaded correctly
or it might be corrupted. (Source: https://docs.anaconda.com/anaconda/install/hashes/Anaconda3-2021.05-Linux-x86_64.sh-hash/)

Ideally after running the bash command, replying "yes" to the prompts should successfully install conda.
However, if conda is not installed successfully, take a look at the following links:
a. Anaconda3 install location is not prepended to PATH in your .bashrc?
   https://askubuntu.com/questions/908827/variable-path-issue-conda-command-not-found
b. Not enough disk space or RAM?
   https://github.com/conda/conda/issues/9125
Listed above are some known issues while installing conda and how they can be resolved.

Other versions of the Anaconda Python Distribution can be found on https://repo.anaconda.com/archive/.
```

cd DMC
conda env create -f environment.yml
conda activate dmc


## Pre-process data set
To pre-process the downloaded data set run the following command:
```
python preprocess.py 
```

## Train MeshNet+SDMC

## Test MeshNet+SDMC

## Custom data sets
The original [ModelNet40](http://modelnet.cs.princeton.edu/) data set contains non-manifold meshes. Pre-processing them to have a fixed number of faces is non-trivial. In most cases, [Watertight Manifold](https://github.com/hjwdzh/Manifold) decimated meshes to 1024 faces. However, for a few meshes, we utilized some functionalities in Blender and MeshLab before using Watertight Manifold. If you want our pre-preprocessing code for your own data sets, you can send an email to vinitvs@udel.edu.

## Citation
If you found this work helpful for your research, please consider citing us.
```
@inproceedings{singh2021mesh,
  title={Mesh Classification with Dilated Mesh Convolutions},
  author={Singh, Vinit Veerendraveer and Sheshappanavar, Shivanand Venkanna and Kambhamettu, Chandra},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={3138--3142},
  year={2021},
  organization={IEEE}
}
```


