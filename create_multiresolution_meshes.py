import trimesh
from trimesh.intersections import slice_faces_plane
import numpy as np
from dvidutils import encode_faces_to_custom_drc_bytes
import dask
import utils
import time
import os
from os import listdir
from os.path import isfile, join, splitext
import pyfqmr
from dask.distributed import Client, worker_client
from utils import CompressedFragment
from numba import jit
import multiprocessing
from datetime import datetime
import argparse


class Fragment:
    """Fragment class used to store and update fragment chunk vertices, faces
    and corresponding lod 0 fragments
    """
    def __init__(self, vertices, faces, lod_0_fragment_pos):
        self.vertices = vertices
        self.faces = faces
        self.lod_0_fragment_pos = lod_0_fragment_pos

    def update_faces(self, new_faces):
        self.faces = np.vstack((self.faces, new_faces + len(self.vertices)))

    def update_vertices(self, new_vertices):
        self.vertices = np.vstack((self.vertices, new_vertices))

    def update_lod_0_fragment_pos(self, new_lod_0_fragment_pos):
        self.lod_0_fragment_pos.append(new_lod_0_fragment_pos)

    def update(self, new_vertices, new_faces, new_lod_0_fragment_pos):
        self.update_faces(new_faces)
        self.update_vertices(new_vertices)
        self.update_lod_0_fragment_pos(new_lod_0_fragment_pos)


@jit(nopython=True)
def get_faces_within_slice(vertices, faces, triangles, max_edge_length,
                           plane_normal, plane_origin):
    """Numba optimized function to get mesh faces to the positive normal side
    of plane_origin with a padding of max_edge_length. A face is defined to be
    within this region if at least one of its vertices is within the region.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        triangle: Array of mesh face vertices
        max_edge_length: Maximum edge length for mesh
        plane_normal: Normal of plane
        plane_origin: Origin of plane

    Returns:
        vertices, faces: Vertices and faces within region
    """

    axis = np.where(plane_normal != 0)[0][0]
    plane_origin = plane_origin[axis]

    if np.any(plane_normal < 0):
        plane_origin = plane_origin + max_edge_length
        faces_in_range = np.where((triangles[:, 0 + axis] <= plane_origin)
                                  | (triangles[:, 3 + axis] <= plane_origin)
                                  | (triangles[:, 6 + axis] <= plane_origin))
    else:
        plane_origin = plane_origin - max_edge_length
        faces_in_range = np.where((triangles[:, 0 + axis] >= plane_origin)
                                  | (triangles[:, 3 + axis] >= plane_origin)
                                  | (triangles[:, 6 + axis] >= plane_origin))

    faces_in_range = faces_in_range[0]
    faces = faces[faces_in_range].reshape(-1)

    vertices_unq = np.unique(faces)  # Numba doesn't support return_inverse
    vertices_renumbering_dict = {}
    for idx, vertex_unq_idx in enumerate(vertices_unq):
        vertices_renumbering_dict[vertex_unq_idx] = idx

    faces = np.array([vertices_renumbering_dict[v_idx] for v_idx in faces])
    vertices = vertices[vertices_unq]

    faces = faces.reshape(-1, 3)
    return vertices, faces


def my_fast_slice_faces_plane(vertices, faces, triangles, max_edge_length,
                              plane_normal, plane_origin):
    """`slice_faces_plane` but with numba optimized prestep to get faces
    within slice. Ideally would speed up code.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        triangles: Array of mesh face vertices
        max_edge_length: Maximum edge length for mesh
        plane_normal: Normal of plane
        plane_origin: Origin of plane

    Returns:
        v, f: Vertices and faces
    """

    if len(vertices) > 0:
        vertices, faces = get_faces_within_slice(vertices, faces, triangles,
                                                 max_edge_length, plane_normal,
                                                 plane_origin)

        vertices, faces = my_slice_faces_plane(vertices, faces, plane_normal,
                                               plane_origin)

    return vertices, faces


def my_slice_faces_plane(vertices, faces, plane_normal, plane_origin):
    """Wrapper for trimesh slice_faces_plane to cath error that happens if the
    whole mesh is to one side of the plane.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        plane_normal: Normal of plane
        plane_origin: Origin of plane

    Returns:
        vertices, faces: Vertices and faces
    """

    if len(vertices) > 0 and len(faces) > 0:
        try:
            vertices, faces = slice_faces_plane(vertices, faces, plane_normal,
                                                plane_origin)
        except ValueError as e:
            if str(e) != "input must be 1D integers!":
                raise
            else:
                pass

    return vertices, faces


def update_fragment_dict(dictionary, fragment_pos, vertices, faces,
                         lod_0_fragment_pos):
    """Update dictionary (in place) whose keys are fragment positions and
    whose values are `Fragment` which is a class containing the corresponding
    fragment vertices, faces and corresponding lod 0 fragment positions.

    This is necessary since each fragment (above lod 0) must be divisible by a
    2x2x2 grid. So each fragment is technically split into many "subfragments".
    Thus the dictionary is used to map all subfragments to the proper parent
    fragment. The lod 0 fragment positions are used when writing out the index
    files because if a subfragment is empty it still needs to be included in
    the index file. By tracking all the corresponding lod 0 fragments of a
    given lod fragment, we can ensure that all necessary empty fragments are
    included.

    Args:
        dictionary: Dictionary of fragment pos keys and fragment info values
        fragment_pos: Current lod fragment position
        vertices: Vertices
        faces: Faces
        lod_0_fragment_pos: Corresponding lod 0 fragment positions
                            corresponding to fragment_pos
    """

    if fragment_pos in dictionary:
        fragment = dictionary[fragment_pos]
        fragment.update(vertices, faces, lod_0_fragment_pos)
        dictionary[fragment_pos] = fragment
    else:
        dictionary[fragment_pos] = Fragment(vertices, faces,
                                            [lod_0_fragment_pos])


@dask.delayed
def generate_mesh_decomposition(vertices, faces, lod_0_box_size,
                                start_fragment, end_fragment, x_start, x_end,
                                current_lod):
    """Dask delayed function to decompose a mesh, provided as vertices and
    faces, into fragments of size lod_0_box_size * 2**current_lod. Each
    fragment is also subdivided by 2x2x2. This is performed over a limited
    xrange in order to parallelize via dask.

    Args:
        vertices: Vertices
        faces: Faces
        lod_0_box_size: Base chunk shape
        start_fragment: Start fragment position (x,y,z)
        end_fragment: End fragment position (x,y,z)
        x_start: Starting x position for this dask task
        x_end: Ending x position for this dask task
        current_lod: The current level of detail

    Returns:
        fragments: List of `CompressedFragments` (named tuple)
    """

    combined_fragments_dictionary = {}
    fragments = []

    nyz, nxz, nxy = np.eye(3)

    if current_lod != 0:
        # Want each chunk for lod>0 to be divisible by 2x2x2 region,
        # so multiply coordinates by 2
        start_fragment *= 2
        end_fragment *= 2
        x_start *= 2
        x_end *= 2

        # 2x2x2 subdividing box size
        sub_box_size = lod_0_box_size * 2**(current_lod - 1)
    else:
        sub_box_size = lod_0_box_size

    for x in range(x_start, x_end):
        plane_origin_yz = nyz * (x + 1) * sub_box_size
        vertices_yz, faces_yz = my_slice_faces_plane(vertices, faces, -nyz,
                                                     plane_origin_yz)

        for y in range(start_fragment[1], end_fragment[1]):
            plane_origin_xz = nxz * (y + 1) * sub_box_size
            vertices_xz, faces_xz = my_slice_faces_plane(
                vertices_yz, faces_yz, -nxz, plane_origin_xz)

            for z in range(start_fragment[2], end_fragment[2]):
                plane_origin_xy = nxy * (z + 1) * sub_box_size
                vertices_xy, faces_xy = my_slice_faces_plane(
                    vertices_xz, faces_xz, -nxy, plane_origin_xy)

                lod_0_fragment_position = tuple(np.array([x, y, z]))
                if current_lod != 0:
                    fragment_position = tuple(np.array([x, y, z]) // 2)
                else:
                    fragment_position = lod_0_fragment_position

                update_fragment_dict(combined_fragments_dictionary,
                                     fragment_position, vertices_xy, faces_xy,
                                     list(lod_0_fragment_position))

                vertices_xz, faces_xz = my_slice_faces_plane(
                    vertices_xz, faces_xz, nxy, plane_origin_xy)

            vertices_yz, faces_yz = my_slice_faces_plane(
                vertices_yz, faces_yz, nxz, plane_origin_xz)

        vertices, faces = my_slice_faces_plane(vertices, faces, nyz,
                                               plane_origin_yz)

    # return combined_fragments_dictionary
    for fragment_pos, fragment in combined_fragments_dictionary.items():
        current_box_size = lod_0_box_size * 2**current_lod
        draco_bytes = encode_faces_to_custom_drc_bytes(
            fragment.vertices,
            np.zeros(np.shape(fragment.vertices)),
            fragment.faces,
            np.asarray(3 * [current_box_size]),
            np.asarray(fragment_pos) * current_box_size,
            position_quantization_bits=10)

        if len(draco_bytes) > 12:
            fragment = CompressedFragment(
                draco_bytes, np.asarray(fragment_pos), len(draco_bytes),
                np.asarray(fragment.lod_0_fragment_pos))
            fragments.append(fragment)

    return fragments


@dask.delayed
def pyfqmr_decimate(input_path, output_path, id, lod, ext, decimation_factor):
    """Mesh decimation using pyfqmr.

    Decimation is performed on a mesh located at `input_path`/`id`.`ext`. The
    target number of faces is 1/2**`lod` of the current number of faces. The
    mesh is written to an stl file in `output_path`/s`lod`/`id`.stl. This
    utilizes `dask.delayed`.

    Args:
        input_path [`str`]: The input path for s0 meshes
        output_path [`str`]: The output path
        id [`int`]: The object id
        lod [`int`]: The current level of detail
        ext [`str`]: The extension of the s0 meshes.
        decimation_factor [`int`]: The factor by which we decimate faces,
                                   scaled by 2**lod
    """

    vertices, faces = utils.mesh_loader(f"{input_path}/{id}{ext}")
    desired_faces = max(len(faces) // (decimation_factor**lod), 1)
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(vertices, faces)
    mesh_simplifier.simplify_mesh(target_count=desired_faces,
                                  aggressiveness=7,
                                  preserve_border=False,
                                  verbose=False)
    vertices, faces, _ = mesh_simplifier.getMesh()

    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(f"{output_path}/s{lod}/{id}.stl")


def generate_decimated_meshes(input_path, output_path, lods, ids, ext,
                              decimation_factor):
    """Generate decimatated meshes for all ids in `ids`, over all lod in `lods`.

    Args:
        input_path (`str`): Input mesh paths
        output_path (`str`): Output mesh paths
        lods (`int`): Levels of detail over which to have mesh
        ids (`list`): All mesh ids
        ext (`str`): Input mesh formats.
        decimation_fraction [`int`]: The factor by which we decimate faces,
                                     scaled by 2**lod
    """

    results = []
    for current_lod in lods:
        if current_lod == 0:
            os.makedirs(f"{output_path}/mesh_lods/", exist_ok=True)
            # link existing to s0
            if not os.path.exists(f"{output_path}/mesh_lods/s0"):
                os.system(
                    f"ln -s {os.path.abspath(input_path)}/ {os.path.abspath(output_path)}/mesh_lods/s0"
                )
        else:
            os.makedirs(f"{output_path}/mesh_lods/s{current_lod}",
                        exist_ok=True)
            for id in ids:
                results.append(
                    pyfqmr_decimate(input_path, f"{output_path}/mesh_lods", id,
                                    current_lod, ext, decimation_factor))

    dask.compute(*results)


@dask.delayed
def generate_neuroglancer_multires_mesh(output_path, num_workers, id, lods,
                                        original_ext, lod_0_box_size):
    """Dask delayed function to generate multiresolution mesh in neuroglancer
    mesh format using prewritten meshes at different levels of detail.

    This function generates the neuroglancer mesh for a single mesh, and
    parallelizes the mesh creation over `num_workers` by splitting the mesh in
    the x-direciton into `num_workers` fragments, each of which is sent to a
    a worker to be further subdivided.

    Args:
        output_path (`str`): Output path to write out neuroglancer mesh
        num_workers (`int`): Number of workers for dask
        id (`int`): Mesh id
        lods (`list`): List of levels of detail
        original_ext (`str`): Original mesh file extension
        lod_0_box_size (`int`): Box size in lod 0 coordinates
    """

    with worker_client() as client:
        os.makedirs(f"{output_path}/multires", exist_ok=True)
        os.system(
            f"rm -rf {output_path}/multires/{id} {output_path}/multires/{id}.index"
        )

        nyz, _, _ = np.eye(3)

        results = []

        for idx, current_lod in enumerate(lods):
            if current_lod == 0:
                vertices, faces = utils.mesh_loader(
                    f"{output_path}/mesh_lods/s{current_lod}/{id}{original_ext}"
                )
            else:
                vertices, faces = utils.mesh_loader(
                    f"{output_path}/mesh_lods/s{current_lod}/{id}.stl")

            current_box_size = lod_0_box_size * 2**current_lod
            start_fragment = np.maximum(
                vertices.min(axis=0) // current_box_size - 1,
                np.array([0, 0, 0])).astype(int)
            end_fragment = (vertices.max(axis=0) // current_box_size +
                            1).astype(int)
            x_stride = int(
                np.ceil(1.0 * (end_fragment[0] - start_fragment[0]) /
                        num_workers))

            for x in range(start_fragment[0], end_fragment[0] + 1, x_stride):
                plane_origin_yz = nyz * (x + x_stride) * current_box_size
                vertices_yz, faces_yz = my_slice_faces_plane(
                    vertices, faces, -nyz, plane_origin_yz)

                if len(vertices_yz) > 0:
                    results.append(
                        generate_mesh_decomposition(
                            client.scatter(vertices_yz),
                            client.scatter(faces_yz), lod_0_box_size,
                            start_fragment, end_fragment, x, x + x_stride,
                            current_lod))

                    vertices, faces = my_slice_faces_plane(
                        vertices, faces, nyz, plane_origin_yz)

            dask_results = dask.compute(*results)

            fragments = [
                fragment for fragments in dask_results
                for fragment in fragments
            ]

            results = []
            dask_results = []
            utils.write_mesh_files(
                f"{output_path}/multires", f"{id}", fragments, current_lod,
                lods[:idx + 1],
                np.asarray([lod_0_box_size, lod_0_box_size, lod_0_box_size]))

    print(f"Completed creation of neuroglancer mesh for mesh {id}!")


def generate_all_neuroglancer_multires_meshes(output_path, num_workers, ids,
                                              lods, original_ext,
                                              lod_0_box_size):
    """Generate all neuroglancer multiresolution meshes for `ids`. Calls dask
    delayed function `generate_neuroglancer_multires_mesh` for each id.

    Args:
        output_path (`str`): Output path to write out neuroglancer mesh
        num_workers (`int`): Number of workers for dask
        ids (`list`): List of mesh ids
        lods (`list`): List of levels of detail
        original_ext (`str`): Original mesh file extension
        lod_0_box_size (`int`): Box size in lod 0 coordinates
    """

    results = []
    for id in ids:
        results.append(
            generate_neuroglancer_multires_mesh(output_path, num_workers, id,
                                                lods, original_ext,
                                                lod_0_box_size))

    dask.compute(*results)


def print_with_datetime(output):
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{now}: {output}")


if __name__ == "__main__":

    # If more than 1 thread per worker, run into issues with decimation?

    parser = argparse.ArgumentParser()
    required_name = parser.add_argument_group('Required named arguments')
    required_name.add_argument("-i",
                               "--input_path",
                               help="Path to lod 0 meshes",
                               type=str,
                               required=True)
    required_name.add_argument("-o",
                               "--output_path",
                               help="Path to write out multires meshes",
                               type=str,
                               required=True)
    required_name.add_argument("-n",
                               "--num_lods",
                               help="Number of levels of detail",
                               type=int,
                               required=True)
    required_name.add_argument("-b",
                               "--box_size",
                               help="lod 0 box size",
                               type=int,
                               required=True)
    parser.add_argument(
        "-s",
        "--skip_decimation",
        help="(Optional) flag to skip mesh decimation if meshes exist",
        action="store_true",
        required=False)
    parser.add_argument(
        "-d",
        "--decimation_factor",
        default=2,
        help=
        "(Optional) factor by which to decimate faces at each lod, ie factor**lod; default is 2",
        type=int,
        required=False)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    num_lods = args.num_lods
    lod_0_box_size = args.box_size
    skip_decimation = args.skip_decimation
    decimation_factor = args.decimation_factor

    lods = list(range(num_lods))
    mesh_files = [
        f for f in listdir(input_path) if isfile(join(input_path, f))
    ]
    mesh_ids = [splitext(mesh_file)[0] for mesh_file in mesh_files]
    mesh_ext = splitext(mesh_files[0])[1]

    t0 = time.time()

    print_with_datetime("Starting dask...")
    num_workers = multiprocessing.cpu_count()
    client = Client(threads_per_worker=1,
                    n_workers=num_workers,
                    memory_limit='16GB')
    print_with_datetime(
        "Dask started! Check http://localhost:8787/status for status")

    #  Mesh decimation
    if not skip_decimation:
        print_with_datetime("Generating decimated meshes...")
        generate_decimated_meshes(input_path, output_path, lods, mesh_ids,
                                  mesh_ext, decimation_factor)
        print_with_datetime("Generated decimated meshes!")

    # Create multiresolution meshes
    print_with_datetime("Generating multires meshes...")
    generate_all_neuroglancer_multires_meshes(output_path, num_workers,
                                              mesh_ids, lods, mesh_ext,
                                              lod_0_box_size)
    print_with_datetime("Generated multires meshes!")

    # Writing out top-level files
    print_with_datetime("Writing info and segment properties files...")
    output_path = f"{output_path}/multires"
    utils.write_segment_properties_file(output_path)
    utils.write_info_file(output_path)
    print_with_datetime("Wrote info and segment properties files!")

    print_with_datetime(f"Complete! Elapsed time: {time.time() - t0}")
