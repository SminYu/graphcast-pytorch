import torch
from typing import NamedTuple, Sequence, List, Tuple, Optional
import itertools
import numpy as np
from scipy.spatial import transform, cKDTree
from scipy.spatial import Delaunay  # For Delaunay triangulation

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class TriangularMesh(NamedTuple):
    vertices: torch.Tensor
    faces: torch.LongTensor
    edges: torch.LongTensor

def merge_meshes(
        mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
    for mesh_i, mesh_ip1 in pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=torch.cat([mesh.faces for mesh in mesh_list], dim=0),
        edges=torch.cat([mesh.edges for mesh in mesh_list], dim=0),
    )

def get_hierarchy_of_triangular_meshes_for_sphere(
        splits: int) -> List[TriangularMesh]:
    current_mesh = get_icosahedron()

    output_meshes = [current_mesh]
    for _ in range(splits):
        current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
        output_meshes.append(current_mesh)
    return output_meshes

def get_icosahedron() -> TriangularMesh:
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1., -1.]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.))
            vertices.append((0., c1, c2))
            vertices.append((c2, 0., c1))

    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1., phi])

    faces = [(0, 1, 2),
             (0, 6, 1),
             (8, 0, 2),
             (8, 4, 0),
             (3, 8, 2),
             (3, 2, 7),
             (7, 2, 1),
             (0, 4, 6),
             (4, 11, 6),
             (6, 11, 5),
             (1, 5, 7),
             (4, 10, 11),
             (4, 8, 10),
             (10, 8, 3),
             (10, 3, 9),
             (11, 10, 9),
             (11, 9, 5),
             (5, 9, 7),
             (9, 3, 7),
             (1, 6, 5),
             ]

    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)

    e1, e2 = faces_to_edges(faces=torch.tensor(faces, dtype=torch.long))
    edges = torch.stack([e1, e2], dim=0).T

    return TriangularMesh(vertices=torch.tensor(vertices, dtype=torch.float32),
                          faces=torch.tensor(faces, dtype=torch.long),
                          edges=edges.to(torch.long))


def _two_split_unit_sphere_triangle_faces(
        triangular_mesh: TriangularMesh) -> TriangularMesh:
    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

    new_faces = []
    for ind1, ind2, ind3 in triangular_mesh.faces:
        ind1, ind2, ind3 = int(ind1), int(ind2), int(ind3)
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
        new_faces.extend([[ind1, ind12, ind31],
                          [ind12, ind2, ind23],
                          [ind31, ind23, ind3],
                          [ind12, ind23, ind31],
                          ])
    
    e1, e2 = faces_to_edges(faces=torch.tensor(new_faces, dtype=torch.long))
    new_edges = torch.stack([e1, e2], dim=0).T

    return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(),
                          faces=torch.tensor(new_faces, dtype=torch.long),
                          edges=new_edges)


class _ChildVerticesBuilder(object):
    def __init__(self, parent_vertices):
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
        self._all_vertices_list = [v for v in parent_vertices]

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))

    def _create_child_vertex(self, parent_vertex_indices):
        child_vertex_position = self._parent_vertices[
            list(parent_vertex_indices)].mean(0)
        child_vertex_position = child_vertex_position / torch.linalg.norm(child_vertex_position)

        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(
            self._all_vertices_list)
        self._all_vertices_list.append(child_vertex_position)

    def get_new_child_vertex_index(self, parent_vertex_indices):
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]

    def get_all_vertices(self):
        return torch.stack(self._all_vertices_list)


def faces_to_edges(faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert faces.ndim == 2
    assert faces.shape[-1] == 3
    senders = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def get_last_triangular_mesh_for_sphere(splits: int) -> TriangularMesh:
    return get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)[-1]


def _grid_lat_lon_to_coordinates(
        grid_latitude: np.ndarray, grid_longitude: np.ndarray) -> np.ndarray:
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude),
        np.deg2rad(90 - grid_latitude))

    return np.stack(
        [np.cos(phi_grid) * np.sin(theta_grid),
         np.sin(phi_grid) * np.sin(theta_grid),
         np.cos(theta_grid)], axis=-1)


def k_query_indices(grid_latitude: np.ndarray,
                    grid_longitude: np.ndarray,
                    mesh: TriangularMesh,
                    k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    
    return query_indices(option='k', 
                        grid_latitude=grid_latitude,
                        grid_longitude=grid_longitude,
                        mesh=mesh,
                        k=k,
                        radius=None)

def r_query_indices(grid_latitude: np.ndarray,
                    grid_longitude: np.ndarray,
                    mesh: TriangularMesh,
                    radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    
    return query_indices(option='r', 
                                grid_latitude=grid_latitude,
                                grid_longitude=grid_longitude,
                                mesh=mesh,
                                radius=radius,
                                k=None)

def query_indices(
        *,
        option: str,
        grid_latitude: np.ndarray,
        grid_longitude: np.ndarray,
        mesh: TriangularMesh,
        radius: Optional[float],
        k:Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude).reshape([-1, 3])

    mesh_positions = mesh.vertices.cpu().numpy()
    kd_tree = cKDTree(mesh_positions)

    if option == 'k':
        _, query_indices = kd_tree.query(x=grid_positions, k=k, workers=4) 
    elif option == 'r':
        query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius, workers=4)
    else:
        raise ValueError('Not an appropriate option (use k or r)')

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        if (len(query_indices.shape) == 1):
            grid_edge_indices.append(grid_index)
        else:
            grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))

        mesh_edge_indices.append(mesh_neighbors)

    if (len(query_indices.shape) == 1):
        grid_edge_indices = np.array(grid_edge_indices)
        mesh_edge_indices = np.array(mesh_edge_indices)
    else:
        grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
        mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    grid_edge_tensor = torch.from_numpy(grid_edge_indices)
    mesh_edge_tensor = torch.from_numpy(mesh_edge_indices)

    return torch.stack([grid_edge_tensor, mesh_edge_tensor], dim=0).to(torch.long)



def in_mesh_triangle_indices(
        *,
        grid_latitude: np.ndarray,
        grid_longitude: np.ndarray,
        mesh: TriangularMesh) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replaces trimesh's closest_point functionality using a combination of
    scipy.spatial.Delaunay for triangulation and barycentric coordinate checks.
    """

    # [num_grid_points=num_lat_points * num_lon_points, 3]
    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude).reshape([-1, 3])

    # Use Delaunay triangulation (requires 2D points, so we project to a plane)
    # Project onto the xy-plane (could choose a better projection for a sphere)
    mesh_vertices_2d = mesh.vertices.cpu().numpy()[:, :2]  # Keep as NumPy for scipy
    tri = Delaunay(mesh_vertices_2d)

    # Find the simplex (triangle) containing each grid point (projected to 2D)
    grid_positions_2d = grid_positions[:, :2]
    simplex_indices = tri.find_simplex(grid_positions_2d)

    # Get the vertices of the containing triangles
    containing_simplices = tri.simplices[simplex_indices]  # Indices into tri.points (mesh_vertices_2d)

    # Calculate barycentric coordinates
    transform = tri.transform[simplex_indices, :2]
    barycentric_coords = np.einsum('ijk,ik->ij', transform, grid_positions_2d - tri.transform[simplex_indices, 2])
    barycentric_coords = np.c_[barycentric_coords, 1 - barycentric_coords.sum(axis=1)]

    # Filter out points outside any triangle (simplex_indices == -1)
    valid_indices = simplex_indices != -1
    valid_grid_indices = np.arange(grid_positions.shape[0])[valid_indices]
    valid_barycentric_coords = barycentric_coords[valid_indices]
    valid_containing_simplices = containing_simplices[valid_indices]


    # Create edges:  Each grid point connects to the 3 vertices of its triangle
    grid_edge_indices = np.repeat(valid_grid_indices, 3)
    mesh_edge_indices = valid_containing_simplices.flatten()

    # Convert to PyTorch tensors
    return torch.tensor(grid_edge_indices, dtype=torch.long), torch.tensor(mesh_edge_indices, dtype=torch.long)


def coordinates_to_lat_lon(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts Cartesian (x, y, z) coordinates on a unit sphere to latitude and longitude.

    Args:
        coordinates: NumPy array of Cartesian (x, y, z) coordinates.
                     Can be a single (3,) array or an array of shape (..., 3).

    Returns:
        A tuple containing two NumPy arrays:
        - grid_latitude:  Latitude in degrees.  Shape matches input coordinates, without the last dimension.
        - grid_longitude: Longitude in degrees. Shape matches input coordinates, without the last dimension.
        Handles edge cases and singularities (poles) gracefully.
    """
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    z = coordinates[..., 2]

    theta = - np.arccos(z) + np.pi/2  # theta (colatitude)
    phi = np.arctan2(y, x)    # phi (longitude)

    grid_latitude = theta
    grid_longitude = phi

    return grid_latitude, grid_longitude


def get_max_edge_distance(mesh):
    faces = mesh.faces
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    edge_distances = np.linalg.norm(mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
    
    return edge_distances.max()
