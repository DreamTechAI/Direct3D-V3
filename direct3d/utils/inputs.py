"""Generate coarse mesh conditioning with VecShape."""

import trimesh

DEFAULT_VECSHAPE_MODEL_ID = "wushuang98/VecShape"


def load_vecshape(model, device):
    try:
        from vecshape import VecShapePipeline
    except ModuleNotFoundError as error:
        if error.name != "vecshape":
            raise
        raise SystemExit(
            "Generating a coarse mesh requires VecShape. Run "
            "'pip install git+https://github.com/DreamTechAI/VecShape.git'."
        ) from error

    return VecShapePipeline.from_pretrained(model).to(device)


def generate_coarse_mesh(vecshape, image, seed):
    mesh = vecshape(image=image, seed=seed)
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
