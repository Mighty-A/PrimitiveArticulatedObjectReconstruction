import os
import nvdiffrast.torch as dr
import trimesh
import torch
import numpy as np
import pytorch3d.transforms

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import nvdiffrast.torch as dr
import trimesh
import torch
import numpy as np


class Nvdiffrast(object):
    def __init__(self, height=640, width=640, focal_length=640):
        self.focal_length = focal_length / max(height, width) * 2
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

        self.mv = torch.eye(4).cuda()
        self.campos = torch.linalg.inv(self.mv)[:3, 3]

        self.znear = 1 - 0.75
        self.zfar = 1 + 0.75

        self.image_height = height
        self.image_width = width

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0),
            torch.transpose(matrix, 1, 2),
        )

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(
            attr.contiguous(),
            rast,
            attr_idx,
            rast_db=rast_db,
            diff_attrs=None if rast_db is None else "all",
        )

    def prepare_input_vector(self, x):
        x = (
            torch.tensor(x, dtype=torch.float32, device="cuda")
            if not torch.is_tensor(x)
            else x
        )
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_seg_map(self, v_pos_clip, faces, colors, render_reso):
        # print ('prepare raster.')
        glctx = dr.RasterizeGLContext()
        # print ('raster prepared.')
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                seg_map, _ = self.interpolate(colors[None, ...], rast, faces)

        return seg_map

    def __call__(self, verts, faces, colors):
        verts = torch.matmul(self.rot, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))

        the_focal_length = self.focal_length
        render_reso = (self.image_height, self.image_width)

        proj = self.intrinsics(
            fx=the_focal_length,
            fy=the_focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
        )
        mvp = proj @ self.mv
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        seg_map = self.render_seg_map(
            v_pos_clip, faces, colors, render_reso
        )  # torch.Size([1, ori_h, ori_w, 3])
        

        return seg_map


class NvdiffrastPartIdx(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

        self.mv = torch.eye(4).cuda()
        self.campos = torch.linalg.inv(self.mv)[:3, 3]

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_prim_index(self, v_pos_clip, faces, primitives, uvs, render_reso):
        glctx = dr.RasterizeGLContext()
        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(primitives[None, ...].float(), rast, faces.int())
                prim_uvs, _ = self.interpolate(uvs[None, ...].float(), rast, faces.int())

        prim_idx = prim_idx.squeeze(0)
        prim_uvs = prim_uvs.squeeze(0)

        return prim_idx, prim_uvs

    def __call__(self, mesh, height, width, focal_length):
        verts = mesh.vertices
        verts = torch.matmul(self.rot, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))
        faces = mesh.faces

        if focal_length == 0:
            the_focal_length = self.focal_length * max(height, width) / 2
        else:
            the_focal_length = focal_length
        render_reso = (height, width)

        proj = self.intrinsics(fx=the_focal_length, fy=the_focal_length, cx=width / 2, cy=height / 2)
        mvp = proj @ self.mv
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        seg_map, uv_map = self.render_prim_index(v_pos_clip, faces, mesh.primitives, mesh.uvs, render_reso)

        return seg_map, uv_map


class NvdiffrastColorANDIdx(object):
    def __init__(self, FOV=39.6, height=512, width=512, focal_length=None):
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        rot = torch.Tensor(rot).cuda()
        self.rot = rot[:3, :3]

        self.mv = torch.eye(4).cuda()
        self.mv[:3, -1] = torch.Tensor([0, 0, -5]).cuda()
        self.campos = torch.linalg.inv(self.mv)[:3, 3]

    def xfm_points(self, points, matrix):
        return torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(matrix, 1, 2))

    def interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                              diff_attrs=None if rast_db is None else 'all')

    def prepare_input_vector(self, x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def intrinsics(self, fx, fy, cx, cy, znear=0.1, zfar=1000.0, device=None):
        P = torch.zeros((4, 4)).cuda()
        width = cx * 2
        height = cy * 2

        P[0][0] = 2.0 * fx / width
        P[1][1] = -2.0 * fy / height
        P[0][2] = 1.0 - 2.0 * cx / width
        P[1][2] = 2.0 * cy / height - 1.0
        P[3][2] = -1.0

        n = znear
        f = zfar
        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2 * f * n) / (n - f)

        return P

    def render_idx_and_color(self, v_pos_clip, faces, primitives, colors, render_reso):
        glctx = dr.RasterizeGLContext()

        with dr.DepthPeeler(glctx, v_pos_clip, faces, render_reso) as peeler:
            for layer_idx in range(1):
                rast, _ = peeler.rasterize_next_layer()
                B, H, W, _ = rast.shape

                prim_idx, _ = self.interpolate(primitives[None, ...].float(), rast, faces.int())
                seg_map, _ = self.interpolate(colors[None, ...], rast, faces.int())

        prim_idx = prim_idx.squeeze(0)
        seg_map = seg_map.squeeze(0)

        return prim_idx, seg_map

    def __call__(self, mesh, rotmat, height=1000, width=800, focal_length=0):
        verts = mesh.vertices
        verts = torch.matmul(rotmat, torch.permute(verts, (1, 0)))
        verts = torch.permute(verts, (1, 0))
        faces = mesh.faces

        verts_mean_x = torch.mean(verts[:, 0])
        verts_mean_y = torch.mean(verts[:, 1])
        verts_mean_z = torch.mean(verts[:, 2])

        # verts = torch.sub(verts, torch.Tensor([verts_mean_x, verts_mean_y, verts_mean_z])[None, :].repeat(verts.size(0), 1).cuda())

        if focal_length == 0:
            the_focal_length = self.focal_length * max(height, width) / 2
        else:
            the_focal_length = focal_length
        render_reso = (height, width)

        proj = self.intrinsics(fx=the_focal_length, fy=the_focal_length, cx=width / 2, cy=height / 2)
        mvp = proj @ self.mv
        v_pos_clip = self.xfm_points(verts[None, ...], mvp[None, ...])

        idx_map, color_map = self.render_idx_and_color(v_pos_clip, faces, mesh.primitives, mesh.colors, render_reso)

        return idx_map, color_map


def render(
    vertices,
    vertex_colors,
    faces,
    resolution=(512, 512)
):

    ctx = dr.RasterizeCudaContext()

    vertices = vertices.unsqueeze(0)

    rast_out, _ = dr.rasterize(
        ctx, vertices.contiguous(), faces.contiguous(), resolution
    )

    image, _ = dr.interpolate(vertex_colors.contiguous(), rast_out, faces)

    image = dr.antialias(image, rast_out, vertices, faces)
    return image


if __name__ == '__main__':
    from visualization_utils import points_on_sq_surface_torch

    vert, faces = points_on_sq_surface_torch(0.5, 0.5, 0.5, 1, 1, torch.eye(3).cuda(), torch.Tensor([0, 0, 0]).reshape(3, 1).cuda(), n_samples=10)
    vert.requires_grad_()
    vert.retain_grad()
    col = torch.tensor([[[1, 0, 0]]], dtype=torch.float32).repeat(1, vert.shape[0], 1).cuda()
    col.requires_grad_()
    col.retain_grad()

    image = render(vertices=vert, vertex_colors=col, faces=faces)
    loss = image.sum()
    loss.backward()
    print(vert.grad.norm())
    image = torch.pow(image, 1/10)
    from matplotlib import pyplot as plt
    plt.imshow(image[0, :, :, 0].detach().cpu().numpy())
    plt.savefig("/home/liweiting/yhy/term_project/temp/test.pdf")
