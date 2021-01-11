#Based on https://github.com/google/neuroglancer/issues/272
import trimesh
import struct
import json
import numpy as np
import tempfile
import subprocess

from functools import cmp_to_key
from math import floor

def get_unique_rows(a1, a2):
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    return np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

if __name__ == "__main__":

    nyz, nxz, nxy = np.eye(3)
    mesh = trimesh.load(f'test/mito_obj_meshes_s2/345809856042.obj')
    current_min = mesh.vertices.min(axis=0)
    current_max = mesh.vertices.max(axis=0)
    halfsies = (current_max[1]+current_min[1])/2
    print(halfsies)
    meshes = []
    unique_rows = []
    for y in range(0,2):
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=nxz, plane_origin=nxz*y*halfsies)
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nxz, plane_origin=nxz*(y+1)*halfsies)
        unique_rows.append(get_unique_rows(mesh_x.vertices, mesh.vertices))

        meshes.append(mesh_x)

    #for new vertices, find all triangles sharing the edge that was split, and redefine em
    output_mesh = trimesh.util.concatenate(meshes)
    #print(get_unique_rows(output_mesh.vertices, mesh.vertices))
    # new_vertices = get_unique_rows(output_mesh.vertices, mesh.vertices)
    # delta=1
    # for i in range(0,len(new_vertices)):
    #     for j in range(i+1,len(new_vertices)):
    #         v1 = new_vertices[i]
    #         v2 = new_vertices[j]
    #         diff = v1-v2
    #         current_delta = sum(diff*diff)
    #         if current_delta<0.05:
    #             print(f"{v1} {v2} {current_delta}")
    #             delta=current_delta

    output_mesh.export("test.obj")

    