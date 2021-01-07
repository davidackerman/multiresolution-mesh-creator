#Based on https://github.com/google/neuroglancer/issues/272
import trimesh
import struct
import json
import numpy as np
import tempfile
import subprocess

from functools import cmp_to_key
from math import floor

def cmp_zorder(lhs, rhs) -> bool:
    def less_msb(x: int, y: int) -> bool:
        return x < y and x < (x ^ y)

    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]

def less_msb(x: int, y: int) -> bool:
    return x < y and x < (x ^ y)

def append_to_submeshes(submeshes, nodes, mesh, node):

    if node not in nodes:
        nodes.append(node)
        submeshes.append(mesh)
    else:
        idx = nodes.index(node)
        submeshes[idx] = trimesh.util.concatenate( [ submeshes[idx], mesh ]);

    return nodes, submeshes

def generate_mesh_decomposition(verts, faces, nodes_per_dim, max_nodes_per_dim):
    # Scale our coordinates.
    scale = nodes_per_dim/(verts.max(axis=0) - verts.min(axis=0))
    verts_scaled = scale*(verts - verts.min(axis=0))
    
    # Define plane normals and create a trimesh object.
    nyz, nxz, nxy = np.eye(3)
    mesh = trimesh.Trimesh(vertices=verts_scaled, faces=faces)

    submeshes = []
    nodes = []
    ratio = nodes_per_dim/max_nodes_per_dim
    for x in range(0, max_nodes_per_dim):
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=nyz, plane_origin=nyz*x*ratio)
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1)*ratio)
        for y in range(0, max_nodes_per_dim):
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y*ratio)
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1)*ratio)
            for z in range(0, max_nodes_per_dim):
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z*ratio)
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1)*ratio)
                
                if len(mesh_z.vertices) > 0:                    
                    node = [floor(node_position*nodes_per_dim/max_nodes_per_dim) for node_position in [x,y,z]]
                    nodes, submeshes = append_to_submeshes(submeshes, nodes, mesh_z, node )
                    #submeshes.append(mesh_z)
                    #nodes.append([x,y,z])
    
    # Sort in Z-curve order
    submeshes, nodes = zip(*sorted(zip(submeshes, nodes), key=cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1]))))
    print(len(submeshes))
    return nodes, submeshes

def my_export_draco(mesh, fragment_origin):
    with tempfile.NamedTemporaryFile(suffix='.obj') as temp_obj:
        mesh.export(temp_obj.name)
        with tempfile.NamedTemporaryFile(suffix='.drc') as encoded:
            try:
                subprocess.check_output(f"./dracoC/draco_encoder_custom \
                                         {temp_obj.name} \
                                         {encoded.name} \
                                         {str(fragment_origin[0])} \
                                         {str(fragment_origin[1])} \
                                         {str(fragment_origin[2])} \
                                         ", 
                                         shell=True,
                                         stderr=subprocess.STDOUT)

            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            encoded.seek(0)
            data = encoded.read()
    return data

if __name__ == "__main__":

    quantization_bits = 10
    lods = np.array([0, 1, 2])

    mesh = trimesh.load(f'test/mito_obj_meshes_s2/345809856042.obj')
    verts = mesh.vertices
    faces = mesh.faces

    chunk_shape = (verts.max(axis=0) - verts.min(axis=0))/2**lods.max()
    grid_origin = verts.min(axis=0)
    lod_scales = np.array([2**lod for lod in lods])

    num_lods = len(lod_scales)
    vertex_offsets = np.array([[0.,0.,0.] for _ in range(num_lods)])

    fragment_offsets = []
    fragment_positions = []
    with open('test/multiresolutionTrimesh/345809856042', 'wb') as f:
        for lod in lods:
            scale = lod_scales[lod]
            nodes_per_dim = 2**(max(lods)-lod)

            mesh = trimesh.load(f'test/mito_obj_meshes_s{lod}/345809856042.obj')
            verts = mesh.vertices
            faces = mesh.faces

            lod_offsets = []
            nodes, submeshes = generate_mesh_decomposition(verts.copy(), faces.copy(), nodes_per_dim, max(lod_scales))
            
            for node,mesh in zip(nodes,submeshes):
                draco = my_export_draco(mesh, node)
                f.write(draco)
                lod_offsets.append(len(draco))
            
            fragment_positions.append(np.array(nodes))
            fragment_offsets.append(np.array(lod_offsets))

    num_fragments_per_lod = np.array([len(nodes) for nodes in fragment_positions])

    with open('test/multiresolutionTrimesh/345809856042.index', 'wb') as f:
        f.write(chunk_shape.astype('<f').tobytes())
        f.write(grid_origin.astype('<f').tobytes())
        f.write(struct.pack('<I', num_lods))
        f.write(lod_scales.astype('<f').tobytes())
        f.write(vertex_offsets.astype('<f').tobytes(order='C'))
        f.write(num_fragments_per_lod.astype('<I').tobytes())
        for frag_pos, frag_offset in zip(fragment_positions, fragment_offsets):
            f.write(frag_pos.T.astype('<I').tobytes(order='C'))
            f.write(frag_offset.astype('<I').tobytes(order='C'))

    with open('test/multiresolutionTrimesh/info', 'w') as f:
        info = {
            '@type': 'neuroglancer_multilod_draco',
            'vertex_quantization_bits': quantization_bits,
            'transform': [1,0,0,0,0,1,0,0,0,0,1,0],
            'lod_scale_multiplier': 1
        }
        
        json.dump(info, f)