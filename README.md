# DMC
Official PyTorch Implementation for [Mesh Classification With Dilated Mesh Convolutions](https://ieeexplore.ieee.org/document/9506311).

Download the pre-processed ModelNet40 data set [here](https://drive.google.com/drive/folders/1y-8m-GRErxCMkuJJf6t8yYSHztlUO0xF?usp=sharing). Files are in the OBJ file format and each mesh model is comprised of exactly 1024 faces. The orginal [ModelNet40](http://modelnet.cs.princeton.edu/) data set contains non-manifold meshes and pre-processing them to have a fixed number of faces is non-trivial. In most cases, [Watertight Manifold](https://github.com/hjwdzh/Manifold) decimated meshes to 1024 faces. However, for a few meshes, we utilized some functionalities in Blender and MeshLab prior to using [Watertight Manifold](https://github.com/hjwdzh/Manifold). If you want our pre-preprocessing code for your own data sets you can send an email at vinitvs@udel.edu.

Official PyTorch Implementation for [Mesh Classification With Dilated Mesh Convolutions](https://ieeexplore.ieee.org/document/9506311).

Download the pre-processed ModelNet40 data set from [here](https://drive.google.com/drive/folders/1y-8m-GRErxCMkuJJf6t8yYSHztlUO0xF?usp=sharing). Files are in the OBJ file format, and each mesh model consists of precisely 1024 faces. The original [ModelNet40](http://modelnet.cs.princeton.edu/) data set contains non-manifold meshes. Pre-processing them to have a fixed number of faces is non-trivial. In most cases, [Watertight Manifold](https://github.com/hjwdzh/Manifold) decimated meshes to 1024 faces. However, for a few meshes, we utilized some functionalities in Blender and MeshLab before using Watertight Manifold. If you want our pre-preprocessing code for your own data sets, you can send an email to vinitvs@udel.edu.

