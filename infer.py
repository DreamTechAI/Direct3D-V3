import argparse
from pathlib import Path

import torch
from PIL import Image

from direct3d import Direct3DV3Pipeline
from direct3d.pipeline import DEFAULT_MAX_TOKENS, DEFAULT_MODEL_ID, DEFAULT_TARGET_FACES, RESOLUTIONS
from direct3d.utils.inputs import DEFAULT_VECSHAPE_MODEL_ID, generate_coarse_mesh, load_vecshape

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_qwen_editor(device):
    from diffusers import QwenImage21Pipeline

    editor = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", torch_dtype=torch.bfloat16)
    editor.enable_model_cpu_offload(device=device)
    return editor


def generate_clay_render(editor, image, seed, device):
    prompt_path = Path(__file__).resolve().parent / "assets" / "prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    generator = torch.Generator(device=device).manual_seed(seed)
    return editor(image=image, prompt=prompt, width=1280, height=1280, generator=generator).images[0]


def input_pairs(image, coarse, output):
    if image.is_dir():
        if coarse is not None and not coarse.is_dir():
            raise ValueError(
                "For an image directory, supply a coarse mesh directory with matching .obj stems."
            )
        images = sorted(p for p in image.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No images found in {image}")
        pairs = [
            (p, coarse / f"{p.stem}.obj" if coarse is not None else None, output / f"{p.stem}.glb")
            for p in images
        ]
    else:
        pairs = [(image, coarse, output)]
    outputs = set()
    for source, mesh, dest in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        if mesh is not None and not mesh.is_file():
            raise FileNotFoundError(mesh)
        if dest.suffix.lower() not in {".obj", ".glb", ".ply"}:
            raise ValueError("Choose an OBJ, GLB or PLY output filename.")
        resolved = dest.resolve()
        if resolved in outputs or resolved == source.resolve() or (mesh is not None and resolved == mesh.resolve()):
            raise ValueError(f"Conflicting output path: {dest}")
        outputs.add(resolved)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate a 3D mesh from a reference image.")
    parser.add_argument(
        "--model", help=f"Hugging Face repository ID or local model directory (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--vecshape-model", default=DEFAULT_VECSHAPE_MODEL_ID,
        help="VecShape Hub repository ID or local model directory for coarse mesh generation (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Run a single stage from a local safetensors checkpoint"
    )
    parser.add_argument("--image", "--infer_path", type=Path, required=True, help="Input image or directory")
    parser.add_argument(
        "--coarse-mesh", type=Path,
        help="Use an aligned OBJ, GLB or PLY (or a matching OBJ directory) with the supplied image, skipping VecShape and Qwen",
    )
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="Use the original image for Direct3D-V3. By default, when generating a coarse mesh, "
             "Qwen-Image-2.1-Turbo produces a clay render under the Qwen Research License.",
    )
    parser.add_argument("--output", type=Path, help="Output mesh or batch output directory")
    parser.add_argument("--resolution", "--voxel_size", type=int, choices=RESOLUTIONS, default=1024)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="One step count for all Direct3D-V3 stages or one per active resolution in ascending order",
    )
    parser.add_argument(
        "--cfg",
        "--guidance-scale",
        nargs="+",
        type=float,
        help="One CFG scale for all Direct3D-V3 stages or one per active resolution in ascending order",
    )
    for stage in RESOLUTIONS:
        parser.add_argument(f"--steps-{stage}", type=int, help=f"Override the step count for {stage} only")
        parser.add_argument(f"--cfg-{stage}", type=float, help=f"Override the CFG scale for {stage} only")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum sparse tokens per stage (default: %(default)s). Lower this if you run out of GPU memory. "
             "The mesh may lose fine surface detail.",
    )
    parser.add_argument(
        "--mc-threshold", "--mc_threshold", type=float, help="Override the threshold for every stage"
    )
    clean = parser.add_mutually_exclusive_group()
    clean.add_argument(
        "--clean",
        action="store_const",
        const=DEFAULT_TARGET_FACES,
        dest="target_faces",
        help="Use the default final target of 2 million triangles",
    )
    clean.add_argument("--target-faces", type=int, help="Custom final triangle budget")
    clean.add_argument(
        "--no-simplify",
        action="store_const",
        const=None,
        dest="target_faces",
        help="Keep all final triangles",
    )
    parser.set_defaults(target_faces=DEFAULT_TARGET_FACES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    stages = (args.resolution,) if args.checkpoint else tuple(r for r in RESOLUTIONS if r <= args.resolution)

    def stage_values(option):
        shared = getattr(args, option)
        values = {}
        if shared:
            if len(shared) == 1:
                values = dict.fromkeys(stages, shared[0])
            elif len(shared) == len(stages):
                values = dict(zip(stages, shared))
            else:
                parser.error(f"--{option} expects one value or one per active stage {stages}.")
        for stage in stages:
            override = getattr(args, f"{option}_{stage}")
            if override is not None:
                values[stage] = override
        return values

    steps = stage_values("steps")
    guidance = stage_values("cfg")
    output = args.output or Path("output" if args.image.is_dir() else "output.glb")
    try:
        pairs = input_pairs(args.image, args.coarse_mesh, output)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    vecshape = load_vecshape(args.vecshape_model, args.device) if args.coarse_mesh is None else None
    editor = load_qwen_editor(args.device) if vecshape is not None and not args.no_enhance else None
    if args.checkpoint is None:
        pipe = Direct3DV3Pipeline.from_pretrained(args.model or DEFAULT_MODEL_ID)
    else:
        pipe = Direct3DV3Pipeline.from_single_file(args.checkpoint, resolution=args.resolution, config_repo=args.model)
    pipe.to(args.device)
    for image, coarse, dest in pairs:
        reference_image = image
        if vecshape is not None:
            print(f"Generating coarse mesh with VecShape: {image}", flush=True)
            coarse = generate_coarse_mesh(vecshape, image, args.seed)
        if editor is not None:
            print(f"Generating clay render with Qwen-Image-2.1-Turbo: {image}", flush=True)
            with Image.open(image) as original:
                reference_image = generate_clay_render(editor, original, args.seed, args.device)
        result = pipe(
            reference_image,
            coarse,
            resolution=args.resolution,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=args.seed,
            max_tokens=args.max_tokens,
            mc_threshold=args.mc_threshold,
            target_faces=args.target_faces,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.export(dest)
        print(f"{dest} ({len(result.faces):,} triangles)", flush=True)


if __name__ == "__main__":
    main()
