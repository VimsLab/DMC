# DMC
Official PyTorch Implementation for [Mesh Classification With Dilated Mesh Convolutions](https://ieeexplore.ieee.org/document/9506311).
The code has been implemented and tested on the Ubuntu operating system only.

## Download data set
Download the pre-processed ModelNet40 data set from [here](https://drive.google.com/drive/folders/1y-8m-GRErxCMkuJJf6t8yYSHztlUO0xF?usp=sharing). Files are in the OBJ file format, and all mesh model consists of precisely 1024 faces. 

## Install tools on Ubuntu
```
sudo apt install curl
```
## Setup Environment
Install the Anaconda Python Distribution if you haven't already.
```
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sha256sum Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
conda list
```

cd back to DMC

```
conda env create -f environment.yml
conda activate dmc
```

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


