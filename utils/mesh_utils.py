"""
Code for the paper "Mesh Classification with Dilated Mesh Convolutions."
published in 2021 IEEE International Conference on Image Processing.
Code Author: Vinit Veerendraveer Singh
Copyright (c) VIMS Lab and its affiliates
The programming details and logic to derive the mesh attributes from meshes in
OBJ file format is declared in this file.
"""
import os.path as osp
import warnings
from numpy import argwhere, where, arange, stack, delete, asarray
from numpy import concatenate as cat
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency

class Mesh:
    """
    This class provides functions to load a mesh, normalize it, and get its
    attributes.
    """
    def __init__(self):
        self.file_path = ""
        self.mesh = None
        self.device = torch.device('cpu:0')
        self.num_verts = 0
        self.num_faces = 0

    def load(self, file_path=""):
        """
        Read mesh from a file path as PyTorch3D Meshes object.

        Args:
            - file_path: str, system path to the mesh
        """
        if not osp.exists(file_path):
            raise ValueError('File path does NOT exist on the system!')

        if not file_path.endswith('.obj'):
            raise ValueError('File path NOT in .obj format!')

        self.file_path = file_path

        self.mesh = load_objs_as_meshes(files=[self.file_path],
                                        device=self.device)

        if not self.is_valid():
            self.mesh = None
            raise ValueError('Invalid mesh!')

    def is_valid(self):
        """
        Check mesh is NOT empty, and its vertices and face normals are valid.

        Returns:
            - validity: bool, the validity of the mesh.
        """
        validity = True

        # Check mesh is not empty
        if self.mesh.isempty():
            validity = False

        # Check mesh vertices are finite and NOT nan
        verts = self.mesh.verts_packed()
        if not torch.isfinite(verts).all() or torch.isnan(verts).all():
            validity = False

        self.num_verts = verts.shape[0]

        # Check mesh face normals are finite and NOT nan
        normals = self.mesh.faces_normals_packed()
        if not torch.isfinite(normals).all() or torch.isnan(normals).all():
            validity = False

        self.num_faces = normals.shape[0]

        return validity

    def normalize(self):
        """
        Normalize and center mesh to fit in a unit sphere centered at (0,0,0).
        """
        if self.mesh is None:
            raise ValueError('Load mesh prior to normalizing!')

        verts = self.mesh.verts_packed()
        faces = self.mesh.faces_packed()
        verts = verts - verts.mean(0)
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        self.mesh = Meshes(verts=[verts], faces=[faces])

    def get_verts(self):
        """
        Get vertices of the mesh.

        Returns:
            - verts: (num_verts, 3), the spatial coordinates of mesh vertices.
        """
        verts = self.mesh.verts_packed().squeeze(0)
        return verts

    def get_faces(self):
        """
        Get faces of the mesh.

        Returns:
            - faces: (num_faces, 3), the indices of mesh vertices that connect
                     to form the mesh faces.
        """
        faces = self.mesh.faces_packed().squeeze(0)
        return faces

    def get_edges(self):
        """
        Get edges of the mesh.

        Returns:
            - edges: (num_edges, 2), the indices of mesh vertices that connect
                     to form the mesh edges.
        """
        # PyTorch3D 0.5.0 throws warning with PyTorch 1.9.0 while computing
        # edge_packed() due to floor division.
        warnings.filterwarnings("ignore")
        edges = self.mesh.edges_packed().squeeze(0)
        return edges

    def get_face_normals(self):
        """
        Get face normals of the mesh.

        Returns:
            - normals: (num_faces, 3), the normals that lie upon the surface of
                       the mesh faces.
        """
        face_normals = self.mesh.faces_normals_packed().squeeze(0)
        return face_normals

    def get_face_corners(self):
        """
        Get corners of the mesh faces.

        Returns:
            - corners: (num_faces, 3, 3), the spatial coordinates of the mesh
                       vertices that connect to form the mesh faces.
        """
        verts = self.get_verts()
        faces = self.get_faces()
        face_corners = verts[faces.long()]
        # Each face has exactly three corners
        assert face_corners.shape == (self.num_faces, 3, 3)

        return face_corners

    def get_face_centers(self):
        """
        Get centers of the mesh faces.

        Returns:
            - centers: (num_faces, 3), the spatial coordinates of points lying
                       at the center of mesh faces' surface.
        """
        verts = self.get_verts()
        faces = self.get_faces()
        face_corners = verts[faces.long()]
        # Each face has exactly one center
        face_centers = torch.sum(face_corners, axis=1)/3
        assert face_centers.shape == (self.num_faces, 3)

        return face_centers

    def get_face_neighbors(self, ring_name=""):
        """
        Get dilated ring neighborhood around each mesh face by ring name.
        Currently, only 1st, 2nd, and 3rd rings are supported.

        Args:
            ring_name: str, the ring names are in accordance with the paper.

        Returns:
            One of the following:
            - ring_1st: (num_faces, 3), the ring neighbors around faces' for a
                        dilation rate of 1.

            - ring_2nd: (num_faces, 6), the ring neighbors around faces' for a
                        dilation rate of 2.

            - ring_3rd: (num_faces, 12), the ring neighbors around faces' for a
                        dilation rate of 3.
        """
        if ring_name not in ['1st Ring', '2nd Ring', '3rd Ring']:
            raise ValueError('Invalid ring name! '
                             'Valid ring_name: 1st Ring, 2nd Ring, 3rd Ring.')
        verts = self.get_verts()
        faces = self.get_faces().permute(1, 0)
        edges = self.get_edges().permute(1, 0)

        ################################## Δ1 ##################################
        # Refer to equation 1 and Fig.3 in the paper.
        trimesh = to_trimesh(Data(pos=verts, edge_index=edges, face=faces))

        # trimesh.graph.face_adjacency() returns an (n,2) list of face indices.
        # Each pair of faces in the list shares an edge, making them adjacent.
        adjacency = face_adjacency(faces=faces, mesh=trimesh)

        ring_1st = []

        # For each face "fi", get its 1st ring neighborhood.
        for fi in range(self.num_faces):
            fi_ring_1st = cat([adjacency[:, 0][argwhere(adjacency[:, 1] == fi)],
                               adjacency[:, 1][argwhere(adjacency[:, 0] == fi)]])

            ring_1st.insert(fi, fi_ring_1st)

        ring_1st = asarray(ring_1st).squeeze(2)

        # Each face is connected to three other faces in Δ1.
        assert ring_1st.shape == (self.num_faces, 3)

        if ring_name == "1st Ring":
            return ring_1st

        ################################## Δ2 ##################################
        # Refer to equation 2 in the paper.
        ring_0th = arange(self.num_faces)
        ring_2 = ring_1st[ring_1st]
        ring_0 = stack([ring_0th]*3, axis=1)
        ring_0 = stack([ring_0]*3, axis=2)

        # Refer to Fig.3 in the paper where r = 2.
        dilation_mask = ring_2 != ring_0
        ring_2nd = ring_2[dilation_mask]
        ring_2nd = ring_2nd.reshape(self.num_faces, -1)

        # For each face, there are six neighboring faces in Δ2.
        assert ring_2nd.shape == (self.num_faces, 6)

        if ring_name == "2nd Ring":
            return ring_2nd

        ################################## Δ3 ##################################
        # Refer to equation 3 in the paper.
        ring_3 = ring_2nd[ring_1st]
        ring_3 = ring_3.reshape(self.num_faces, -1)

        # Refer to Fig.3 in the paper where r = 3.
        # For each face "fi", get its 3rd ring neighborhood.
        ring_3rd = []
        for fi in range(self.num_faces):
            fi_ring_3 = ring_3[fi]
            for neighbor in range(ring_1st.shape[1]):
                fi_ring_1st = ring_1st[fi, neighbor]
                dilation_mask = delete(
                    arange(fi_ring_3.shape[0]),
                    where(fi_ring_3 == fi_ring_1st)[0][0:2])
                fi_ring_3 = fi_ring_3[dilation_mask]
            fi_ring_3rd = fi_ring_3
            ring_3rd.insert(fi, fi_ring_3rd)

        # For each face, there are twelve neighboring faces in Δ3.
        ring_3rd = asarray(ring_3rd)
        assert ring_3rd.shape == (self.num_faces, 12)

        if ring_name == "3rd Ring":
            return ring_3rd

    def __str__(self):
        """
        Provide file path of the mesh
        """
        if self.file_path == "":
            raise ValueError("Load mesh using the .load() function!")
        else:
            return 'Mesh located at: {0}'.format(self.file_path)
