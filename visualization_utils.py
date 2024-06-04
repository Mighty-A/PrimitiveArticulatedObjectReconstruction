import os
import torch
import numpy as np
import pickle

import trimesh
from pyquaternion import Quaternion


def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)

def fexp_torch(x, p):
    return torch.sign(x)*(torch.abs(x)**p)

def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z

def sq_surface_torch(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp_torch(torch.cos(eta), e1) * fexp_torch(torch.cos(omega), e2)
    y = a2 * fexp_torch(torch.cos(eta), e1) * fexp_torch(torch.sin(omega), e2)
    z = a3 * fexp_torch(torch.sin(eta), e1)
    return x, y, z


def points_on_sq_surface(a1, a2, a3, e1, e2, R, t, n_samples=100):
    """Computes a SQ given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    eta = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, n_samples, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, eta, omega)

    # Get an array of size 3x10000 that contains the points of the SQ
    points = np.stack([x, y, z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t

    x_tr = points_transformed[0].reshape(n_samples, n_samples)
    y_tr = points_transformed[1].reshape(n_samples, n_samples)
    z_tr = points_transformed[2].reshape(n_samples, n_samples)

    return x_tr, y_tr, z_tr, points_transformed


def points_on_sq_surface_torch(a1, a2, a3, e1, e2, R, t, n_samples=100):
    """Computes a SQ given a set of parameters and saves it into a tensor array"""
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    eta = torch.linspace(-torch.pi / 2, torch.pi / 2, n_samples).cuda()
    omega = torch.linspace(-torch.pi, torch.pi, n_samples).cuda()
    eta, omega = torch.meshgrid(eta, omega, indexing="ij")

    x, y, z = sq_surface_torch(a1, a2, a3, e1, e2, eta, omega)

    # Get a tensor of size 3x10000 that contains the points of the SQ
    points = torch.stack([x, y, z]).reshape(3, -1)
    points_transformed = R.T @ points + t
    
    points = points_transformed.transpose(1, 0)
    w = torch.ones((points.shape[0], 1)).cuda()
    points = torch.cat((points, w), dim=1)
    faces = []
    for i in range(n_samples - 1):
        for j in range(n_samples - 1):
            p1 = i * n_samples + j
            p2 = p1 + n_samples
            p3 = p2 + 1
            p4 = p1 + 1
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
    faces = torch.IntTensor(faces).cuda()


    return points, faces


def points_on_cuboid(a1, a2, a3, e1, e2, R, t, n_samples=100):
    """Computes a cube given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    X = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1]
    ], dtype=np.float32)
    X[X == 1.0] = a1
    X[X == 0.0] = -a1

    Y = np.array([
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ], dtype=np.float32)
    Y[Y == 1.0] = a2
    Y[Y == 0.0] = -a2

    Z = np.array([
        [1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1]
    ], dtype=np.float32)
    Z[Z == 1.0] = a3
    Z[Z == 0.0] = -a3

    points = np.stack([X, Y, Z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t
    print ("R:", R)
    print ("t:", t)

    assert points.shape == (3, 18)

    x_tr = points_transformed[0].reshape(2, 9)
    y_tr = points_transformed[1].reshape(2, 9)
    z_tr = points_transformed[2].reshape(2, 9)
    return x_tr, y_tr, z_tr, points_transformed


def _from_primitive_parms_to_mesh(primitive_params):
    if not isinstance(primitive_params, dict):
        raise Exception(
            "Expected dict and got {} as an input"
            .format(type(primitive_params))
        )
    # Extract the parameters of the primitives
    a1, a2, a3 = primitive_params["size"]
    e1, e2 = primitive_params["shape"]
    t = np.array(primitive_params["location"]).reshape(3, 1)
    R = Quaternion(primitive_params["rotation"]).rotation_matrix.reshape(3, 3)

    # Sample points on the surface of its mesh
    _, _, _, V = points_on_sq_surface(a1, a2, a3, e1, e2, R, t)
    assert V.shape[0] == 3

    color = np.array(primitive_params["color"])
    color = (color*255).astype(np.uint8)

    # Build a mesh object using the vertices loaded before and get its
    # convex hull
    m = trimesh.Trimesh(vertices=V.T).convex_hull
#     m = trimesh.Trimesh(vertices=V.T)
    # Apply color
    for i in range(len(m.faces)):
        m.visual.face_colors[i] = color

    return m


def _from_primitive_parms_to_mesh_v2(verts, color):
    '''
    Args:
        verts: sampled_points x 3
    '''
    color = np.array(color)
    color = (color*255).astype(np.uint8)

    # Build a mesh object using the vertices loaded before and get its
    # convex hull
    m = trimesh.Trimesh(vertices=verts).convex_hull
#     m = trimesh.Trimesh(vertices=V.T)
    # Apply color
    for i in range(len(m.faces)):
        m.visual.face_colors[i] = color

    return m


def _from_primitive_parms_to_mesh_v3(verts, color):
    '''
    Args:
        verts: sampled_points x 3
    '''
    color = np.array(color)
    color = (color*255).astype(np.uint8)

    # Build a mesh object using the vertices loaded before
    m = trimesh.Trimesh(vertices=verts)
#     m = trimesh.Trimesh(vertices=V.T)
    # Apply color
    for i in range(len(m.faces)):
        m.visual.face_colors[i] = color

    return m


def _from_primitive_parms_to_mesh_v4(verts, faces, color):
    color = np.array(color)
    color = (color * 255).astype(np.uint8)

    verts = verts.squeeze(0)
    faces = faces.squeeze(0)
    verts = verts.cpu().detach().data.numpy()
    faces = faces.cpu().detach().data.numpy()

    m = trimesh.Trimesh(vertices=verts, faces=faces)
    for i in range(len(m.faces)):
        m.visual.face_colors[i] = color

    return m


def save_primitive_as_ply(primitive_params, filepath):
    m = _from_primitive_parms_to_mesh(primitive_params)
    # Make sure that the filepath endswith .obj
    if not filepath.endswith(".ply"):
        raise Exception(
            "The filepath should have an .ply suffix, instead we received {}"
            .format(filepath)
        )
    m.export(filepath, file_type="ply")


def save_prediction_as_ply(primitive_files, filepath):
    if not isinstance(primitive_files, list):
        raise Exception(
            "Expected list and got {} as an input"
            .format(type(primitive_files))
        )
    m = None
    for p in primitive_files:
        # Parse the primitive parameters
        prim_params = pickle.load(open(p, "rb"))
        _m = _from_primitive_parms_to_mesh(prim_params)
        m = trimesh.util.concatenate(_m, m)

    m.export(filepath, file_type="ply")


def save_prediction_as_ply_v2(vertices, colors, filepath):
    bone_num = vertices.shape[0]

    m = None
    for bone_i in range(bone_num):
        _m = _from_primitive_parms_to_mesh_v2(vertices[bone_i], (colors[bone_i % len(colors)]) + (1.0,))
        m = trimesh.util.concatenate(_m, m)

    m.export(filepath, file_type="ply")


def save_prediction_as_ply_v3(vertices, colors, filepath):
    bone_num = vertices.shape[0]

    for bone_i in range(bone_num):
        m = _from_primitive_parms_to_mesh_v2(vertices[bone_i], (colors[bone_i % len(colors)]) + (1.0,))
        m.export(os.path.join(filepath, str(bone_i)+'.ply'), file_type="ply")


def save_prediction_as_ply_v4(vertices, colors, filepath):
    bone_num = vertices.shape[0]

    for bone_i in range(bone_num):
        m = _from_primitive_parms_to_mesh_v3(vertices[bone_i], (colors[bone_i % len(colors)]) + (1.0,))
        m.export(os.path.join(filepath, str(bone_i)+'.ply'), file_type="ply")


def save_prediction_as_ply_v5(vert_list, face_list, colors, filepath):
    bone_num = len(vert_list)

    m = None
    for bone_i in range(bone_num):
        _m = _from_primitive_parms_to_mesh_v4(vert_list[bone_i], face_list[bone_i], (colors[bone_i % len(colors)]) + (1.0,))
        m = trimesh.util.concatenate(_m, m)

    m.export(os.path.join(filepath, 'whole.ply'), file_type="ply")


def save_prediction_as_ply_v6(vert_list, face_list, colors, filepath):
    bone_num = len(vert_list)

    for bone_i in range(bone_num):
        m = _from_primitive_parms_to_mesh_v4(vert_list[bone_i], face_list[bone_i], (colors[bone_i % len(colors)]) + (1.0,))
        m.export(os.path.join(filepath, str(bone_i) + '.ply'), file_type="ply")


