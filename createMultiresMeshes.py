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
from dask.distributed import Client, progress, get_client, worker_client
from utils import Fragment
import logging
from numba import jit


@jit
def get_faces_within_slice(v, f, fv, slice_min, slice_max, axis, max_edge_length):
    slice_min -= max_edge_length
    slice_max += max_edge_length
    if axis == "x":
        c = 0
    elif axis == "y":
        c = 1
    elif axis == "z":
        c = 2

    faces_in_range = np.where((fv[:, 0+c] >= slice_min) & (fv[:, 0+c] <= slice_max) &
                              (fv[:, 3+c] >= slice_min) & (fv[:, 3+c] <= slice_max) &
                              (fv[:, 6+c] >= slice_min) & (fv[:, 6+c] <= slice_max))[0]
    f = f[faces_in_range].reshape(-1)

    v_unq = np.unique(f)  # Numba doesn't support return_inverse argument
    v_renumbering_dict = {}
    for idx, v_unq_idx in enumerate(v_unq):
        v_renumbering_dict[v_unq_idx] = idx

    f = np.array([v_renumbering_dict[f_idx] for f_idx in f])

    v = v[v_unq]

    f = f.reshape(-1, 3)
    return v, f


def my_fast_slice_faces_plane(v, f, fv, slice_min, slice_max, axis, max_edge_length, plane_normal, plane_origin):
    if len(v) > 0:
        v, f = get_faces_within_slice(
            v, f, fv, slice_min, slice_max,  axis, max_edge_length)
        v, f = my_slice_faces_plane(
            v, f, plane_normal=plane_normal, plane_origin=plane_origin)
    return v, f


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
def generate_mesh_decomposition(v, f, lod_0_box_size, start_fragment, end_fragment, x_start,  x_end, current_lod):
    combined_fragments_dictionary = {}
    fragments = []

    nyz, nxz, nxy = np.eye(3)

    if current_lod != 0:
        sub_box_size = lod_0_box_size*2**(current_lod-1)
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
                if current_lod != 0:
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
        current_box_size = lod_0_box_size*2**current_lod
        draco_bytes = encode_faces_to_custom_drc_bytes(
            v, np.zeros(np.shape(v)), f, np.asarray(3*[current_box_size]), np.asarray(fragment_origin)*current_box_size, position_quantization_bits=10)

        if len(draco_bytes) > 12:
            fragment = Fragment(draco_bytes, np.asarray(
                fragment_origin), len(draco_bytes), np.asarray(lod_0_fragment_positions))
            fragments.append(fragment)

    return fragments


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


@dask.delayed
def pyfqmr_decimate(input_path, output_path, id, lod, ext):
    mesh = trimesh.load(f"{input_path}/{id}.{ext}")
    desired_faces = max(len(mesh.faces)//(2**lod),  1)

    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(
        target_count=desired_faces, aggressiveness=7, preserve_border=False, verbose=False)
    v, f, _ = mesh_simplifier.getMesh()

    mesh = trimesh.Trimesh(v, f)
    mesh.export(f"{output_path}/s{lod}/{id}.stl")


def generate_multiscale_meshes(input_path, output_path, lods, ids, ext):
    for current_lod in lods:
        if current_lod == 0:
            os.makedirs(f"{output_path}", exist_ok=True)
            # link existing to s0
            if not os.path.exists(f"{output_path}/s0"):
                os.system(f"ln -s {input_path}/ {output_path}/s0")
        else:
            os.makedirs(f"{output_path}/s{current_lod}", exist_ok=True)
            for id in ids:
                results.append(pyfqmr_decimate(
                    input_path, output_path, id, current_lod, ext))

    dask.compute(*results)


def generate_all_neuroglancer_meshes(output_path, num_workers, ids, lods, original_ext):
    results = []
    for id in ids:
        results.append(generate_single_neuroglancer_mesh(
            output_path, num_workers, id, lods, original_ext))

    dask.compute(*results)


@dask.delayed
def generate_single_neuroglancer_mesh(output_path, num_workers, id, lods, original_ext):

    with worker_client() as client:
        os.system(f"rm -rf {output_path}/{id} {output_path}/{id}.index")

        nyz, _, _ = np.eye(3)

        mesh = []

        lod_0_box_size = 64*4
        results = []
        max_edge_length = mesh.edges_unique_length.max
        for idx, current_lod in enumerate(lods):
            if current_lod == 0:
                mesh = trimesh.load(
                    f"{output_path}/s{current_lod}/{id}.{original_ext}")
            else:
                mesh = trimesh.load(f"{output_path}/s{current_lod}/{id}.stl")

            v = mesh.vertices
            f = mesh.faces

            current_box_size = lod_0_box_size * 2**current_lod
            start_fragment = np.maximum(
                v.min(axis=0) // current_box_size - 1, np.array([0, 0, 0])).astype(int)
            end_fragment = (v.max(axis=0) // current_box_size + 1).astype(int)
            x_stride = int(
                np.ceil(1.0*(end_fragment[0]-start_fragment[0])/num_workers))

            for x in range(start_fragment[0], end_fragment[0]+1, x_stride):
                vx, fx = my_fast_slice_faces_plane(
                    v, f, plane_normal=-nyz, plane_origin=nyz*(x+x_stride)*current_box_size)

                if len(vx) > 0:
                    results.append(generate_mesh_decomposition(
                        client.scatter(vx), client.scatter(fx), lod_0_box_size, start_fragment, end_fragment, x, x+x_stride, current_lod))

                    v, f = my_fast_slice_faces_plane(
                        v, f, plane_normal=nyz, plane_origin=nyz*(x+x_stride)*current_box_size)

            dask_results = dask.compute(* results)

            fragments = [
                fragment for fragments in dask_results for fragment in fragments]
            results = []
            dask_results = []
            utils.write_mesh_files(output_path, f"{id}", fragments, current_lod, lods[:idx+1], np.asarray(
                [lod_0_box_size, lod_0_box_size, lod_0_box_size]))

    print(f"Completed id {id}!")


if __name__ == "__main__":

    # If more than 1 thread per worker, run into issues with decimation?

    num_workers = 20  # for creating meshes
    client = Client(threads_per_worker=1,
                    n_workers=num_workers,
                    memory_limit='16GB')
    t0 = time.time()

    #
    num_lods = 6
    input_path = "/groups/cosem/cosem/ackermand/meshesForWebsite/res1decimation0p1/jrc_hela-1/er_seg/"
    output_path = "/groups/cosem/cosem/ackermand/meshesForWebsite/res1decimation0p1/jrc_hela-1/test_simpler_multires/"
    results = []
    ids = list(range(1,  2))
    lods = list(range(num_lods))

    t0 = time.time()
    generate_all_neuroglancer_meshes(
        output_path, num_workers, ids, lods, "obj")
    print(time.time()-t0)
    # dask_results = dask.compute(*results)
    # utils.write_segment_properties_file(output_path)
    # utils.write_info_file(output_path)

    # generate_multiscale_meshes(
    #     input_path, output_path, lods, ids, 'obj')

    print("end", time.time()-t0)
