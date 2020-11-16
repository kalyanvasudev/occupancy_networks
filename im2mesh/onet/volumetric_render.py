import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [-(i - W * 0.5) / focal, -(j - H * 0.5) / focal, torch.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, has_rgb=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray. Same as mask/occupancy [Todo: check this!]
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # torch.set_printoptions(profile="full")
    print("blah", torch.sum(raw > 0.2))
    # input("Press Enter to continue...")

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    if has_rgb:
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    noise = 0.0
    # if raw_noise_std > 0.:
    #    noise = torch.randn(raw[...,3].shape) * raw_noise_std

    if has_rgb:
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples] # Ex: Nert
    else:
        alpha = raw2alpha(raw + noise, dists)  # [N_rays, N_samples] # Ex: Onet

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = (
        alpha
        * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1
        )[:, :-1]
    )

    ret = {}

    depth_map = torch.sum(weights * z_vals, -1)
    ret["depth_map"] = depth_map
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )
    ret["disp_map"] = disp_map

    acc_map = torch.sum(weights, -1)
    # print(acc_map)
    ret["acc_map"] = acc_map
    ret["sharp_acc_map"] = (torch.sum(raw > 0.2*100.0, -1) > 0.0).to(torch.float32).to(device)

    if has_rgb:
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        ret["rgb_map"] = rgb_map

    ret["weights"] = weights
    return ret


def render_rays(
    ray_batch,
    z_latent,
    c_latent,
    decoder,
    N_samples,
    network_query_fn,
    retraw=False,
    white_bkgd=False,
    raw_noise_std=0.0,
    has_rgb=False,
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_query_fn: function that querries model values for the points
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.

      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    z_vals = near * (1.0 - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, z_latent, c_latent, decoder).cpu()
    ret = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, has_rgb)

    if retraw:
        ret["raw"] = raw

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def network_query_fn_onet(inputs, z, c, decoder, decoder_kwargs={}, netchunk=200):

    inputs_lower = inputs > -0.5
    inputs_higher = inputs < 0.5
    inputs_flags = inputs_higher * inputs_lower
    inputs_indicator = torch.prod(inputs_flags, -1).to(device)

    z_inp = torch.cat(inputs.shape[0] * [z])
    
    #return                                  decoder(inputs.to(device), z_inp.to(device), c.to(device))
    #return inputs_indicator *               decoder(inputs.to(device), z_inp.to(device), c.to(device))
    #return  torch.sigmoid(decoder(inputs.to(device), z_inp.to(device), c.to(device)))
    return 100.0*inputs_indicator * torch.sigmoid(decoder(inputs.to(device), z_inp.to(device), c.to(device)))
    # return torch.cat([decoder(inputs[i:i+netchunk],z[i:i+netchunk],c[i:i+netchunk])
    #            for i in range(0, inputs.shape[0], netchunk)], 0)


def batchify_rays(rays_flat, chunk, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    z_latent = kwargs["z_latent"]
    c_latent = kwargs["c_latent"]
    for i in range(0, rays_flat.shape[0], chunk):
        kwargs["z_latent"] = z_latent  # [i:i+chunk]
        kwargs["c_latent"] = c_latent  # [i:i+chunk]
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024, rays=None, c2w=None, near=0.0, far=2.0, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # Render and  reshape to match img dimentions
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def render_path(render_poses, hwf, chunk, render_kwargs, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    occs = []
    depths = []
    sharp_occs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        all_ret = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        depth_img = all_ret["depth_map"]
        sharp_occ_img = all_ret["sharp_acc_map"]
        occ_img = all_ret["acc_map"]
        #print("fjldsjflsdjflksjd", torch.max(occ_img), torch.min(occ_img))
        #occ_img[occ_img>0.05] = 1
        #occ_img.to(dtype=torch.float32)
        
        occs.append(occ_img.cpu().detach().numpy())
        sharp_occs.append(sharp_occ_img.cpu().detach().numpy())
        depths.append(depth_img.cpu().detach().numpy())

        if i == 0:
            print(depth_img.shape, occ_img.shape)
            np.set_printoptions(threshold=sys.maxsize)
            # print(depths[-1])
        if savedir is not None:
            dep = depths[-1]
            filename = os.path.join(savedir, "{:03d}_depth.png".format(i))
            imageio.imwrite(filename, dep)
            # print(dep)
            occ = occs[-1]
            filename = os.path.join(savedir, "{:03d}_occ.png".format(i))
            imageio.imwrite(filename, occ)

            sharp_occ = sharp_occs[-1]
            filename = os.path.join(savedir, "{:03d}_sharp_occ.png".format(i))
            imageio.imwrite(filename, sharp_occ)

    depths = np.stack(depths, 0)
    occs = np.stack(occs, 0)
    return depths, occs


import os
import torch
import copy
from random import Random


import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)

import numpy as np
from pytorch3d.renderer.mesh import TexturesAtlas, Textures

from pytorch3d.renderer.mesh.shader import (
    TexturedSoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    BlendParams,
)


def render_masks(mesh, rots, trans, device):
    fx_screen = 300.0 
    image_width = 256
    # follow formula here - 
    # https://github.com/facebookresearch/pytorch3d/blob/c25fd836942c42101c7519c5124ebbb74d450bf8/docs/notes/cameras.md
    cameras = PerspectiveCameras(device=device, R=rots, T=trans,focal_length=fx_screen * 2.0 / image_width)
    #cameras = OpenGLPerspectiveCameras(device=device, R=rots, T=trans)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0, #np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=1,
        bin_size=0,
    )
    renderer_mask = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    img_mask = renderer_mask(mesh.extend(rots.shape[0]))
    img_mask = img_mask[:, ..., 3]
    return img_mask.to(device)
