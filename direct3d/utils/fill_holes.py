import fillhole_cpp
import torch


def _group_edges(faces, num_vertices):
    # Keep directed edges for the boundary winding, group their undirected keys.
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    keys = edges.amin(1) * num_vertices + edges.amax(1)
    unique, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    first = torch.full_like(unique, len(edges))
    first.scatter_reduce_(0, inverse, torch.arange(len(edges), device=faces.device), reduce="amin")
    return edges, unique, counts, first


def fill_holes_torch(vertices, faces):
    num_vertices = len(vertices)
    edges, unique, counts, first = _group_edges(faces, num_vertices)
    face_ids = first[counts == 1] % len(faces)
    ids, number = torch.unique(face_ids, return_counts=True)
    keep = torch.ones(len(faces), device=faces.device, dtype=torch.bool)
    keep[ids[number == 2]] = False
    # Edges were concatenated by side, so the owning face is the index modulo face count.
    keep[first[counts > 2] % len(faces)] = False
    if not keep.all():
        faces = faces[keep]
        if not len(faces):
            return faces
        edges, unique, counts, first = _group_edges(faces, num_vertices)

    boundary = first[counts == 1]
    if not len(boundary):
        return faces
    _, inverse, counts = torch.unique(boundary % len(faces), return_inverse=True, return_counts=True)
    boundary_edges = edges[boundary[counts[inverse] < 3]][:, [1, 0]]
    if not len(boundary_edges):
        return faces
    unique_edges = torch.stack((unique // num_vertices, unique % num_vertices), dim=1)
    filled = fillhole_cpp.fill_hole_cuda1(
        vertices.cpu(), faces.int().cpu(), boundary_edges.int().cpu(), unique_edges.int().cpu().contiguous()
    )
    return torch.cat((faces, filled.to(device=faces.device, dtype=faces.dtype))) if filled.numel() else faces
