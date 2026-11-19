import torch
from safetensors import safe_open

from .sparse_dit import SparseDiT


def load_denoiser(path, config, dtype, *, allow_wrapped=False):
    with torch.device("meta"):
        model = SparseDiT(**config)
    keys = model.state_dict().keys()
    with safe_open(path, framework="pt", device="cpu") as archive:
        names = set(archive.keys())
        prefix = ""
        if names != keys:
            prefixes = ("", "model.", "model.model.") if allow_wrapped else ()
            for candidate in prefixes:
                if all(candidate + key in names for key in keys):
                    prefix = candidate
                    break
            else:
                raise ValueError(f"Checkpoint {path} does not match the configured SparseDiT parameters.")
        tensors = {key: archive.get_tensor(prefix + key).to(dtype=dtype) for key in keys}
    model.load_state_dict(tensors, assign=True, strict=True)
    return model.eval()
