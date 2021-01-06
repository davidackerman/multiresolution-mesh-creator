#Based on https://github.com/google/neuroglancer/issues/272
import trimesh
import struct
import json
import numpy as np

from functools import cmp_to_key


class Quantize():
    def __init__(self, fragment_origin, fragment_shape, input_origin, quantization_bits):
        self.upper_bound = np.iinfo(np.uint32).max >> (np.dtype(np.uint32).itemsize*8 - quantization_bits)
        self.scale = self.upper_bound / fragment_shape
        self.offset = input_origin - fragment_origin + 0.5/self.scale
    
    def __call__(self, v_pos):
        output = np.minimum(self.upper_bound, np.maximum(0, self.scale*(v_pos + self.offset))).astype(np.uint32)
        return output

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

def generate_mesh_decomposition(verts, faces, nodes_per_dim, bits):
    # Scale our coordinates.
    scale = nodes_per_dim/(verts.max(axis=0) - verts.min(axis=0))
    verts_scaled = scale*(verts - verts.min(axis=0))
    
    # Define plane normals and create a trimesh object.
    nyz, nxz, nxy = np.eye(3)
    mesh = trimesh.Trimesh(vertices=verts_scaled, faces=faces)
    
    # create submeshes. 
    if nodes_per_dim == 1:
        return [[0,0,0]], [mesh]

    submeshes = []
    nodes = []
    for x in range(0, nodes_per_dim):
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=nyz, plane_origin=nyz*x)
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
        for y in range(0, nodes_per_dim):
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
            for z in range(0, nodes_per_dim):
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))
                
                # Initialize Quantizer.
                quantize = Quantize(
                    fragment_origin=np.array([x, y, z]), 
                    fragment_shape=np.array([1, 1, 1]), 
                    input_origin=np.array([0,0,0]), 
                    quantization_bits=bits
                )
    
                if len(mesh_z.vertices) > 0:
                    mesh_z.vertices = quantize(mesh_z.vertices)
                
                    submeshes.append(mesh_z)
                    nodes.append([x,y,z])
    
    # Sort in Z-curve order
    submeshes, nodes = zip(*sorted(zip(submeshes, nodes), key=cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1]))))
            
    return nodes, submeshes

if __name__ == "__main__":

    quantization_bits = 10
    lods = np.array([0, 1 , 2])

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
        for lod in lods[::-1]:
            scale = lod_scales[lod]
            #nodes_per_dim = 2**(max(lods)-lod)

            mesh = trimesh.load(f'test/mito_obj_meshes_s{lod}/345809856042.obj')
            verts = mesh.vertices
            faces = mesh.faces

            lod_offsets = []
            nodes, submeshes = generate_mesh_decomposition(verts.copy(), faces.copy(), scale, quantization_bits)
            
            for mesh in submeshes:
                draco = trimesh.exchange.ply.export_draco(mesh, bits=quantization_bits)
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