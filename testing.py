import trimesh
import numpy as np
import sys
from io_utils import stdout_redirected
import time
from dvidutils import encode_faces_to_custom_drc_bytes


def simplify_openmesh(self, fraction):
    """
    Simplify this mesh in-place, by the given fraction (of the original vertex count).
    Uses OpenMesh to perform the decimation.
    This has similar performance to our default simplify() method,
    but does not require a subprocess or conversion to OBJ.
    Therefore, it can be faster in cases where I/O is the major bottleneck,
    rather than the decimation procedure itself.
    (For example, when lightly decimating a large mesh, I/O is the bottleneck.)
    """
    if len(self.vertices_zyx) == 0:
        return

    target = max(4, int(fraction * len(self.vertices_zyx)))
    if fraction is None or fraction == 1.0:
        if len(self.normals_zyx) == 0:
            self.recompute_normals(True)
        return

    import openmesh as om

    # Mesh construction in OpenMesh produces a lot of noise on stderr.
    # Send it to /dev/null
    try:
        sys.stderr.fileno()
    except:
        # Can't redirect stderr if it has no file descriptor.
        # Just let the output spill to wherever it's going.
        m = om.TriMesh(self.vertices_zyx[:, ::-1], self.faces)
    else:
        # Hide stderr, since OpenMesh construction is super noisy.
        with stdout_redirected(stdout=sys.stderr):
            m = om.TriMesh(self.vertices_zyx[:, ::-1], self.faces)

    h = om.TriMeshModQuadricHandle()
    d = om.TriMeshDecimater(m)
    d.add(h)
    d.module(h).unset_max_err()
    d.initialize()

    logger.debug(
        f"Attempting to decimate to {target} (Reduce by {len(self.vertices_zyx) - target})")
    eliminated_count = d.decimate_to(target)
    logger.debug(f"Reduced by {eliminated_count}")
    m.garbage_collection()

    self.vertices_zyx = m.points()[:, ::-1].astype(np.float32)
    self.faces = m.face_vertex_indices().astype(np.uint32)

    # Force normal reomputation to eliminate possible degenerate faces
    # (Can decimation produce degenerate faces?)
    self.recompute_normals(True)


def simplify_open3d(self, fraction):
    import open3d
    print("get as open3d")
    as_open3d = open3d.geometry.TriangleMesh(
        vertices=open3d.utility.Vector3dVector(self.vertices_zyx[:, ::-1]),
        triangles=open3d.utility.Vector3iVector(self.faces))
    print(f"got as open3d {len(self.faces)} {int(fraction*len(self.faces))}")
    resulting_faces = int(fraction*len(self.faces))
    if(resulting_faces > 5):
        simple = as_open3d.simplify_quadric_decimation(
            resulting_faces, boundary_weight=1E9)
        print(f"simplified {resulting_faces}, {len(simple.triangles)}")
        self.faces = np.asarray(simple.triangles).astype(np.uint32)
        print(
            f"max {np.amax(np.asarray(simple.vertices)[:,::-1])} {np.amax(self.vertices_zyx)}")
        self.vertices_zyx = np.asarray(simple.vertices)[
            :, ::-1].astype(np.float32)


def simplify_pySimplify(self, fraction):
    # https://github.com/Kramer84/Py_Fast-Quadric-Mesh-Simplification
    num_faces = len(self.faces)
    if (num_faces > 4):
        mesh = trimesh.Trimesh(self.vertices_zyx[:, ::-1], self.faces)
        simplify = pySimplify()
        simplify.setMesh(mesh)
        simplify.simplify_mesh(target_count=int(
            num_faces*fraction), aggressiveness=7, preserve_border=True, verbose=0)
        mesh_simplified = simplify.getMesh()
        self.vertices_zyx = mesh_simplified.vertices[:,
                                                     ::-1].astype(np.float32)
        self.faces = mesh_simplified.faces.astype(np.uint32)
        self.box = np.array([self.vertices_zyx.min(axis=0),
                             np.ceil(self.vertices_zyx.max(axis=0))]).astype(np.int32)


# def generate_mesh_decomposition(verts, faces, nodes_per_dim, max_nodes_per_dim, minimum_coordinates, maximum_coordinates):
def generate_mesh_decomposition(mesh):
    # Scale our coordinates.
    # scale = nodes_per_dim/(maximum_coordinates-minimum_coordinates)
    # verts_scaled = scale*(verts - minimum_coordinates)

    # # Define plane normals and create a trimesh object.
    # mesh = trimesh.Trimesh(vertices=verts_scaled, faces=faces)

    # submeshes = []
    # nodes = []
    # ratio = nodes_per_dim/max_nodes_per_dim

    nyz, nxz, nxy = np.eye(3)
    ratio = 64*4
    max_nodes_per_dim = 234

    v = mesh.vertices
    f = mesh.faces
    for x in range(21, max_nodes_per_dim):
        t = time.time()
        total_blocks = 0
        sz = 0
        vx, fx = trimesh.intersections.slice_faces_plane(
            v, f, plane_normal=-nyz, plane_origin=nyz*(x+1)*ratio)
        for y in range(0, max_nodes_per_dim):
            vy, fy = trimesh.intersections.slice_faces_plane(
                vx, fx, plane_normal=-nxz, plane_origin=nxz*(y+1)*ratio)
            # vy, fy = trimesh.intersections.slice_faces_plane(
            #     vy, fy, plane_normal=nxz, plane_origin=nxz*y*ratio)
            for z in range(0, max_nodes_per_dim):
                vz, fz = trimesh.intersections.slice_faces_plane(
                    vy, fy, plane_normal=-nxy, plane_origin=nxy*(z+1)*ratio)
                if len(vz) > 0:
                    normals = np.zeros(np.shape(vz))
                    draco_bytes = normals
                    #draco_bytes = encode_faces_to_custom_drc_bytes(
                    #    vz, normals, fz, np.asarray(3*[ratio]), np.asarray([0, 0, 0]))

                if (len(vz) > 0):
                    total_blocks += 1
                    sz = max(sz, len(draco_bytes))

                vy, fy = trimesh.intersections.slice_faces_plane(
                    vy, fy, plane_normal=nxy, plane_origin=nxy*(z+1)*ratio)

            vx, fx = trimesh.intersections.slice_faces_plane(
                vx, fx, plane_normal=nxz, plane_origin=nxz*(y+1)*ratio)

        v, f = trimesh.intersections.slice_faces_plane(
            v, f, plane_normal=nyz, plane_origin=nyz*(x+1)*ratio)

        # if len(mesh_z.vertices) > 0:
        #     node = [floor(node_position*nodes_per_dim/max_nodes_per_dim)
        #             for node_position in [x, y, z]]
        #     nodes, submeshes = append_to_submeshes(
        #         submeshes, nodes, mesh_z, node)
        print(x, y, z, total_blocks, time.time()-t, sz)


if __name__ == "__main__":

    mesh = trimesh.load(
        "/groups/cosem/cosem/ackermand/meshesForWebsite/res1decimation0p1/jrc_hela-1/er_seg/17.obj")
    print("read")
    print(np.min(mesh.vertices, axis=0))
    print(np.max(mesh.vertices, axis=0))
    #trimesh.repair.fix_winding(mesh)
    generate_mesh_decomposition(mesh)
    print("fixed")
    fraction = 0.25
    import openmesh as om

    for i in range(1, 7):
        target = max(4, int(fraction * len(mesh.vertices)))

        try:
            sys.stderr.fileno()
        except:
            # Can't redirect stderr if it has no file descriptor.
            # Just let the output spill to wherever it's going.
            m = om.TriMesh(mesh.vertices, mesh.faces)
        else:
            # Hide stderr, since OpenMesh construction is super noisy.
            with stdout_redirected(stdout=sys.stderr):
                m = om.TriMesh(mesh.vertices, mesh.faces)
    h = om.TriMeshModQuadricHandle()
    d = om.TriMeshDecimater(m)
    d.add(h)
    d.module(h).unset_max_err()
    d.initialize()

    print(
        f"Attempting to decimate to {target} (Reduce by {len(mesh.vertices) - target})")
    eliminated_count = d.decimate_to(target)
    print(f"Reduced by {eliminated_count}")
    m.garbage_collection()

    mesh.vertices = m.points().astype(np.float32)
    mesh.faces = m.face_vertex_indices().astype(np.uint32)

    print(f"{i} simplified")
    _ = mesh.export(
        f"/groups/cosem/cosem/ackermand/meshesForWebsite/res1decimation0p1/jrc_hela-1/test/17_{i}.obj")
    print(f"{i} finished")
