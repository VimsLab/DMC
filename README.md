# DMC
Official PyTorch implementation for [Mesh Classification With Dilated Mesh Convolutions](https://ieeexplore.ieee.org/document/9506311).
The code has been implemented and tested on the Ubuntu operating system only.

![Alt text](doc/Figure2.png?raw=true)

## Install CUDA Toolkit and cuDNN
Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and the [cuDNN library](https://developer.nvidia.com/rdp/cudnn-archive) matching the version of your Ubuntu operating system. Installation of the [Anaconda Python Distribution](https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh) is required as well.

## Install tools on Ubuntu
```
sudo apt install curl
sudo apt install unzip
```

## Setup Environment
Make sure the Anaconda Python Distribution is installed and cuda. Then run the following commands:
```
cd DMC
conda env create -f environment.yml
conda activate dmc
pip install gdown
```


## Download data set
Download the [pre-processed ModelNet40](https://drive.google.com/drive/folders/1y-8m-GRErxCMkuJJf6t8yYSHztlUO0xF?usp=sharing) data set in the datasets/raw/ directory. Files are in the OBJ file format (.obj), and all mesh model consists of precisely 1024 faces. Run the following commands:
```
cd datasets/raw/
gdown 'https://drive.google.com/uc?id=1YrOkSoAMrfNxW3xpHB1xxVQ01SzE-1LX'
unzip ModelNet40.zip
rm ModelNet40.zip
cd ../..
```

## Pre-process data set
Pre-processing is performed to derive mesh attributes (e.g., faces, rings, etc.) from the .obj files. The derived attributes are saved as .npz files. To pre-process the downloaded data set run the following command:
```
python preprocess.py ModelNet40
```
The preprocess.py file imports the Mesh class from utils/mesh_utils.py. The programming details and logic to derive the mesh attributes is declared in this mesh_utils.py file.

To visualize the derived mesh attributes run the following command (browser required):
```
jupyter notebook demo.ipynb
```

In case the browser isn't present run the following command on the remote server:
```
jupyter notebook --no-browser --port=8889
```
Copy the URL (http://localhost:8889/?token=...) .

And run the following command on the client side with a browser:
```
ssh -L 8889:localhost:8889 <REMOTE_USER>@<REMOTE_IP_ADDRESS>
```
Paste the URL.

## Train MeshNet+SDMC
We adapt MeshNet to perform dilated convolutions by replacing our Stacked Dilated Mesh Convolution block in place of its Mesh Convolution block.
To train this redesigned model (MeshNet+SDMC) to classify meshes in ModelNet40 run the following command:
```
python train.py
```
SDMC in the released code uses neighborhoods with a dilation rate of 1, 2, and 3. The subsampling rate is set to 0.33.
However, it is easy to change these configurations.

## Test MeshNet+SDMC
To test MeshNet+SDMC to classify meshes in ModelNet40 run the following command:
```
python test.py
```

## Test on pre-trained MeshNet+SDMC
Download the [pre-trained weights](https://drive.google.com/drive/folders/1y-8m-GRErxCMkuJJf6t8yYSHztlUO0xF?usp=sharing) in the ckpt_root/ModelNet40 directory.
To test MeshNet+SDMC to classify meshes in ModelNet40 run the following command:
```
cd ckpt_root/ModelNet40
gdown https://drive.google.com/uc?id=1r-ACZ0JI1-Gyw8TFygKfQ2Yd-w5RsiFo
cd ../..
python test.py
```
Note that retraining MeshNet+SDMC will over-write the pre-trained weights.

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
