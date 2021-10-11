import trimesh
import numpy as np
from io_utils import stdout_redirected
from dvidutils import encode_faces_to_custom_drc_bytes
import dask
import utils
import time

from collections import namedtuple
import openmesh as om
import sys
import os
import pyfqmr
from dask.distributed import Client, progress
from utils import Fragment


def get_face_indices_in_range(mesh, face_mins, stop):
    max_edge_length = np.max(mesh.edges_unique_length)
    rows = np.where(face_mins[:, 0] < stop+max_edge_length)
    return rows[0]


def renumber_vertex_indices(faces, vertex_indices_in_range):
    def renumber_indicies(a, val_old, val_new):
        arr = np.empty(a.max()+1, dtype=val_new.dtype)
        arr[val_old] = val_new
        return arr[a]

    faces = np.reshape(faces, -1)
    faces = renumber_indicies(
        faces, vertex_indices_in_range, np.arange(len(vertex_indices_in_range)))

    return np.reshape(faces, (-1, 3))


def my_slice_faces_plane(v, f, plane_normal, plane_origin):
    # Wrapper for trimesh slice_faces_plane, checks that there are vertices and faces and catches an error that happens if the whole mesh is to one side

    if len(v) > 0 and len(f) > 0:
        try:
            v, f = trimesh.intersections.slice_faces_plane(
                v, f, plane_normal=plane_normal, plane_origin=plane_origin)
        except ValueError as e:
            if str(e) != "input must be 1D integers!":
                raise
            else:
                pass

    return v, f


def update_dict(combined_fragments_dictionary, fragment_origin, v, f, lod_0_fragment_position):
    if fragment_origin in combined_fragments_dictionary:
        [v_combined, f_combined,
            lod_0_fragment_positions_combined] = combined_fragments_dictionary[fragment_origin]

        f_combined = np.vstack((f_combined, f+len(v_combined)))
        v_combined = np.vstack((v_combined, v))
        lod_0_fragment_positions_combined.append(lod_0_fragment_position)

        combined_fragments_dictionary[fragment_origin] = [
            v_combined, f_combined, lod_0_fragment_positions_combined]
    else:
        combined_fragments_dictionary[fragment_origin] = [
            v, f, [lod_0_fragment_position]]

    return combined_fragments_dictionary


@dask.delayed
def generate_mesh_decomposition(v, f, lod_0_box_size, start_fragment, end_fragment, x_start, x_end, current_lod, starting_lod):
    combined_fragments_dictionary = {}
    fragments = []

    nyz, nxz, nxy = np.eye(3)

    if current_lod != starting_lod:
        sub_box_size = lod_0_box_size*2**(current_lod-1 - starting_lod)
        start_fragment *= 2  # since want it to be divisible by 2x2x2 subregions
        end_fragment *= 2
        x_start *= 2
        x_end *= 2
    else:
        sub_box_size = lod_0_box_size

    for x in range(x_start, x_end):
        vx, fx = my_slice_faces_plane(
            v, f, plane_normal=-nyz, plane_origin=nyz*(x+1)*sub_box_size)

        for y in range(start_fragment[1], end_fragment[1]):
            vy, fy = my_slice_faces_plane(
                vx, fx, plane_normal=-nxz, plane_origin=nxz*(y+1)*sub_box_size)

            for z in range(start_fragment[2], end_fragment[2]):
                vz, fz = my_slice_faces_plane(
                    vy, fy, plane_normal=-nxy, plane_origin=nxy*(z+1)*sub_box_size)

                lod_0_fragment_position = tuple(np.asarray([x, y, z]))
                if current_lod != starting_lod:
                    fragment_position = tuple(np.asarray([x, y, z]) // 2)
                else:
                    fragment_position = lod_0_fragment_position

                combined_fragments_dictionary = update_dict(
                    combined_fragments_dictionary, fragment_position, vz, fz, list(lod_0_fragment_position))

                vy, fy = my_slice_faces_plane(
                    vy, fy, plane_normal=nxy, plane_origin=nxy*(z+1)*sub_box_size)

            vx, fx = my_slice_faces_plane(
                vx, fx, plane_normal=nxz, plane_origin=nxz*(y+1)*sub_box_size)

        v, f = my_slice_faces_plane(
            v, f, plane_normal=nyz, plane_origin=nyz*(x+1)*sub_box_size)

    # return combined_fragments_dictionary
    for fragment_origin, [v, f, lod_0_fragment_positions] in combined_fragments_dictionary.items():
        current_box_size = lod_0_box_size*2**(current_lod-starting_lod)
        draco_bytes = encode_faces_to_custom_drc_bytes(
            v, np.zeros(np.shape(v)), f, np.asarray(3*[current_box_size]), np.asarray(fragment_origin)*current_box_size, position_quantization_bits=10)

        if len(draco_bytes) > 12:
            fragment = Fragment(draco_bytes, np.asarray(
                fragment_origin), len(draco_bytes), np.asarray(lod_0_fragment_positions))
            fragments.append(fragment)

    return fragments


def pyfqmr_decimate(v, f, fraction):
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(v, f)
    mesh_simplifier.simplify_mesh(
        target_count=len(f)//2, aggressiveness=7, preserve_border=False, verbose=0)
    v, f, _ = mesh_simplifier.getMesh()
    return v, f


def decimate(v, f, fraction):
    target = max(4, int(fraction * len(v)))

    try:
        sys.stderr.fileno()
    except:
        # Can't redirect stderr if it has no file descriptor.
        # Just let the output spill to wherever it's going.
        m = om.TriMesh(v, f)
    else:
        # Hide stderr, since OpenMesh construction is super noisy.
        with stdout_redirected(stdout=sys.stderr):
            m = om.TriMesh(v, f)

        h = om.TriMeshModQuadricHandle()
        d = om.TriMeshDecimater(m)
        d.add(h)
        d.module(h).unset_max_err()
        d.initialize()

        print(
            f"Attempting to decimate to {target} (Reduce by {len(v) - target})")
        eliminated_count = d.decimate_to(target)
        print(f"Reduced by {eliminated_count}")
        m.garbage_collection()

    v = m.points().astype(np.float32)
    f = m.face_vertex_indices().astype(np.uint32)

    return v, f


def generate_neuroglancer_meshes(input_path, output_path, id, client):

    os.system(f"rm -rf {output_path}/{id}*")
    mesh = trimesh.load(f"{input_path}/{id}.obj")

    nyz, nxz, nxy = np.eye(3)
    num_workers = 16
    num_lods = 6

    lods = list(range(num_lods))

    v_whole = mesh.vertices.astype(np.float32)
    f_whole = mesh.faces.astype(np.uint32)

    mesh = []

    lod_0_box_size = 64*4
    results = []

    for idx, current_lod in enumerate(lods):

        t = time.time()
        current_box_size = lod_0_box_size * 2**(current_lod-lods[0])
        start_fragment = np.maximum(np.min(v_whole, axis=0).astype(
            int) // current_box_size - 1, np.array([0, 0, 0]))
        end_fragment = np.max(v_whole, axis=0).astype(
            int) // current_box_size + 1
        x_stride = int(
            np.ceil(1.0*(end_fragment[0]-start_fragment[0])/num_workers))

        v = v_whole
        f = f_whole

        if len(lods) > 1:  # decimate here so don't have to keep original v_whole, f_whole around
            v_whole, f_whole = pyfqmr_decimate(v_whole, f_whole, 4)

        for x in range(start_fragment[0], end_fragment[0]+1, x_stride):
            vx, fx = my_slice_faces_plane(
                v, f, plane_normal=-nyz, plane_origin=nyz*(x+x_stride)*current_box_size)
            if len(vx) > 0:
                results.append(generate_mesh_decomposition(
                    client.scatter(vx), client.scatter(fx), lod_0_box_size, start_fragment, end_fragment, x, x+x_stride, current_lod, lods[0]))
                v, f = my_slice_faces_plane(
                    v, f, plane_normal=nyz, plane_origin=nyz*(x+x_stride)*current_box_size)
        dask_results = dask.compute(* results)
        fragments = [
            fragment for fragments in dask_results for fragment in fragments]
        results = []
        dask_results = []
        utils.write_files(output_path, f"{id}", fragments, current_lod, lods[:idx+1], np.asarray(
            [lod_0_box_size, lod_0_box_size, lod_0_box_size]))


if __name__ == "__main__":
    client = Client(threads_per_worker=4,
                    n_workers=16)
    t0 = time.time()
    for id in [1, 3]:  # range(1, 158): # range(1, 158):
        t_id_start = time.time()
        generate_neuroglancer_meshes(
            "/groups/cosem/cosem/ackermand/meshesForWebsite/res1decimation0p1/jrc_hela-1/er_seg/", "/groups/cosem/cosem/ackermand/meshesForWebsite/res1decimation0p1/jrc_hela-1/test_simpler_multires/", id, client)
        t_id_end = time.time()
        print(id, t_id_end-t_id_start, t_id_end-t0)
