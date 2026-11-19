import base64
import warnings

import numpy as np
import torch
from skimage.measure._marching_cubes_lewiner_luts import CASESCLASSIC

from .fill_holes import fill_holes_torch


def surface_cells(keys, sdf, resolution):
    # A cell is eligible only if all eight sparse samples exist and their signs differ.
    # Keep the classic marching cubes corner order when assembling the edge table.
    r = resolution
    offsets = keys.new_tensor([0, 1, r + 1, r, r * r, r * r + 1, r * r + r + 1, r * r + r])
    bits = 2 ** torch.arange(8, device=keys.device)
    cells, cases = [], []
    for origins in keys.split(131072):
        neighbors = origins[:, None] + offsets
        index = torch.searchsorted(keys, neighbors).clamp_max_(len(keys) - 1)
        case = ((sdf[index] < 0).long() * bits).sum(1)
        active = (case != 0) & (case != 255) & (keys[index] == neighbors).all(1)
        cells.append(index[active])
        cases.append(case[active])
    return torch.cat(cells), torch.cat(cases)


def get_mesh(sparse_sdf, keys, cubes, case_ids, voxel_size):
    device = keys.device
    if not len(cubes):
        raise ValueError("No surface crosses the requested marching cubes threshold.")
    shape, encoded = CASESCLASSIC
    table = np.frombuffer(base64.b64decode(encoded), dtype=np.int8).reshape(shape).copy()
    table = torch.from_numpy(table).to(device)
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
        device=device,
    )
    rows = table[case_ids]
    owners, slots = (rows >= 0).nonzero(as_tuple=True)
    ends = edges[rows[owners, slots].long()]
    ends = cubes[owners[:, None], ends]
    # Pack each undirected sample pair into one integer to deduplicate shared edges.
    n = len(keys)
    pair_keys = ends.amin(1) * n + ends.amax(1)
    unique, faces = torch.unique(pair_keys, return_inverse=True)
    a, b = unique // n, unique % n

    def positions(indices):
        k = keys[indices]
        return torch.stack((k // voxel_size**2, k // voxel_size % voxel_size, k % voxel_size), 1).float()

    t = sparse_sdf[a] / (sparse_sdf[a] - sparse_sdf[b])
    vertices = torch.lerp(positions(a), positions(b), t[:, None])
    return vertices / (voxel_size - 1) - 0.5, faces.reshape(-1, 3)


def sparse_mc_infer(latent_sdf, latent_index, voxel_size, patch_size, mc_threshold):
    if len(latent_index) == 0:
        raise ValueError("The sparse SDF is empty.")
    origins = latent_index.long() * patch_size
    i = torch.arange(patch_size**3, device=latent_sdf.device)
    offsets = (i // patch_size**2, i // patch_size % patch_size, i % patch_size)
    strides = (voxel_size**2, voxel_size, 1)
    base = sum(origins[:, axis] * stride for axis, stride in enumerate(strides))
    delta = sum(offset * stride for offset, stride in zip(offsets, strides))
    keys = base[:, None] + delta
    valid = torch.ones_like(keys, dtype=torch.bool)
    for axis, offset in enumerate(offsets):
        coord = origins[:, axis, None] + offset
        valid &= (coord >= 0) & (coord < voxel_size - 1)
    keys, order = keys[valid].sort()
    if not len(keys):
        raise ValueError("No SDF samples fall inside the voxel grid.")
    sdf = latent_sdf.reshape_as(valid)[valid][order] - mc_threshold
    del valid, coord, order
    cubes, case_ids = surface_cells(keys, sdf, voxel_size)
    vertices, faces = get_mesh(sdf, keys, cubes, case_ids, voxel_size)
    del cubes, case_ids, sdf, keys
    try:
        faces = fill_holes_torch(vertices, faces)
    except RuntimeError as error:
        warnings.warn(f"Hole filling failed: {error}", RuntimeWarning, stacklevel=2)
    return vertices, faces
