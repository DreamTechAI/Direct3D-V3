from pathlib import Path

import numpy as np
import trimesh


def load_coarse(source):
    if isinstance(source, trimesh.Trimesh):
        mesh = source
    else:
        path = Path(source)
        if path.suffix.lower() not in (".obj", ".glb", ".ply"):
            raise ValueError("Choose a coarse mesh in OBJ, GLB or PLY format.")
        mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not len(mesh.vertices):
        raise ValueError("The coarse mesh must contain triangles.")
    if not np.isfinite(mesh.vertices).all() or mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError("The coarse mesh contains invalid vertices or face indices.")
    return mesh


def export_mesh(vertices, faces, target_faces):
    vertices = vertices.cpu().numpy()
    faces = faces.cpu().numpy()
    if target_faces is not None and len(faces) > target_faces:
        import pyfqmr

        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(vertices, faces)
        simplifier.simplify_mesh(
            target_count=target_faces, aggressiveness=5, preserve_border=True, verbose=False
        )
        vertices, faces, _ = simplifier.getMesh()
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
