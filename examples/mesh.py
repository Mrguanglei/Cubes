# 文件: mesh.py 或在 optimization_1.py 的顶部
import torch

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.normals = None
        self.compute_normals()

    def compute_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        normals = torch.cross(v1 - v0, v2 - v0)
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
        self.normals = normals
