"""Launch the Direct3D-V3 demo from the repository root with python app.py."""

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event

import gradio as gr
from PIL import Image

from direct3d import Direct3DV3Pipeline
from direct3d.pipeline import DEFAULT_MAX_TOKENS, DEFAULT_MODEL_ID, DEFAULT_TARGET_FACES, RESOLUTIONS

ASSETS = Path(__file__).resolve().parent / "assets"
TEASER = ASSETS / "demo" / "paper-teaser.webp"
STYLES = ASSETS / "demo" / "app.css"
VIEWER = ASSETS / "demo" / "mesh-viewer.js"
GRADIO_MAJOR = int(gr.__version__.split(".")[0])


def build_demo(pipeline):
    defaults = pipeline.sampling_defaults
    examples = []
    thumbnails = []
    meshes = sorted((ASSETS / "coarse_meshes").glob("*.obj"), key=lambda path: int(path.stem))
    image_dir = ASSETS / "input_imgs"
    for normal_map in (False, True):
        for mesh in meshes:
            if normal_map:
                reference = image_dir / f"{mesh.stem}_normal.png"
            else:
                reference, = image_dir.glob(f"{mesh.stem}.*")
            with Image.open(reference) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((256, 192), Image.Resampling.LANCZOS)
            thumbnails.append(thumb)
            examples.append((reference, mesh))

    def load_example(event: gr.SelectData):
        reference, mesh = examples[event.index]
        return str(reference), str(mesh), None, None, ""

    def stage_bars(totals, completed):
        rows = []
        for stage, total in totals.items():
            done = completed.get(stage, 0)
            rows.append(
                f'<label><b>{stage}³</b><progress max="{total}" value="{done}"></progress>'
                f'<span>{done} / {total}</span></label>'
            )
        return '<div class="stage-bars" aria-label="Diffusion sampling progress">' + "".join(rows) + "</div>"

    def generate(
        image, coarse, resolution, seed, simplify, target_faces, max_tokens,
        steps_512, cfg_512, steps_1024, cfg_1024, steps_1536, cfg_1536,
        progress=gr.Progress(),
    ):
        if image is None or coarse is None:
            raise gr.Error("Select an example or upload both a reference image and an aligned coarse mesh.")
        stage_steps = {}
        stage_cfg = {}
        for stage, count, cfg in (
            (512, steps_512, cfg_512),
            (1024, steps_1024, cfg_1024),
            (1536, steps_1536, cfg_1536),
        ):
            if stage > resolution:
                break
            stage_steps[stage] = defaults[stage][0] if count is None else int(count)
            if cfg is not None:
                stage_cfg[stage] = cfg

        completed = {}
        yield None, None, stage_bars(stage_steps, completed)
        progress(0, desc="Removing image background")
        cutout = pipeline.remove_background(image)
        yield cutout, gr.skip(), gr.skip()
        events = Queue()
        cancelled = Event()

        def report(stage, done, total):
            if cancelled.is_set():
                raise CancelledError()
            events.put((stage, done, total))

        with ThreadPoolExecutor(max_workers=1) as executor:
            task = executor.submit(
                pipeline, cutout, coarse, background_removed=True, resolution=resolution,
                num_inference_steps=stage_steps, guidance_scale=stage_cfg, seed=int(seed),
                target_faces=int(target_faces) if simplify else None, max_tokens=max_tokens,
                on_sampling_step=report,
            )
            task.add_done_callback(lambda _: events.put(None))
            try:
                while (event := events.get()) is not None:
                    stage, done, total = event
                    completed[stage] = done
                    if done == 0:
                        progress(None, desc=f"Preparing {stage}³ geometry")
                    elif done == total:
                        progress(None, desc=f"Extracting {stage}³ mesh")
                    else:
                        progress((done, total), desc=f"Sampling {stage}³")
                    yield gr.skip(), gr.skip(), stage_bars(stage_steps, completed)
                mesh = task.result()
            finally:
                cancelled.set()
        progress(None, desc="Exporting GLB")
        with TemporaryDirectory(prefix="direct3d-") as directory:
            output = Path(directory) / "generated.glb"
            mesh.export(output)
            cached = download.move_resource_to_block_cache(output)
        yield cutout, cached, stage_bars(stage_steps, completed)

    # Gradio 6 moved custom styles from Blocks to launch.
    block_style = {"css_paths": [STYLES]} if GRADIO_MAJOR < 6 else {}
    gallery_tools = {"show_download_button": False} if GRADIO_MAJOR < 6 else {"buttons": []}
    ui_event = {"api_name": False} if GRADIO_MAJOR < 6 else {"api_visibility": "private"}
    with gr.Blocks(title="Direct3D-V3", delete_cache=(3600, 3600), **block_style) as demo:
        gr.HTML(f"""
          <header class="paper-banner">
            <img class="paper-art" src="gradio_api/file={TEASER.as_posix()}"
                 alt="A scene assembled from meshes generated by Direct3D-V3">
            <div class="paper-info">
              <p class="paper-venue">ACM Transactions on Graphics · SIGGRAPH Asia 2026</p>
              <h1>Direct3D-V3</h1>
              <p class="paper-title">Latent-Free Diffusion Transformers in the Native Voxel Domain</p>
              <p class="paper-authors">Shuang Wu, Jiachen Qian, Youtian Lin, Feihu Zhang,
                 Philip Torr, Yao Yao, and Xun Cao</p>
              <nav class="paper-links" aria-label="Project links">
                <a class="paper-link" href="https://dl.acm.org/doi/10.1145/3842524"
                   target="_blank" rel="noopener noreferrer">Paper ↗</a>
                <a class="paper-link" href="https://github.com/DreamTechAI/Direct3D-V3"
                   target="_blank" rel="noopener noreferrer">GitHub ↗</a>
              </nav>
            </div>
          </header>
        """, elem_id="paper-header")

        with gr.Column(elem_id="examples-strip"):
            gr.HTML("""
              <div class="examples-heading">
                <h2>Examples</h2>
                <p role="note">For a more detailed mesh, we recommend using GPT Image 2.5 or
                  <a href="https://github.com/EnVision-Research/Lotus-2" target="_blank" rel="noopener noreferrer">Lotus-2</a>
                  to convert your input image into a surface normal map or a white clay render that preserves fine geometric details.</p>
              </div>
            """)
            gallery = gr.Gallery(
                thumbnails, label="Paired examples", show_label=False, columns=6, rows=2,
                height="auto", object_fit="contain", allow_preview=False, interactive=False,
                format="webp", elem_id="example-gallery", **gallery_tools,
            )

        with gr.Row(equal_height=False, elem_id="generation-workspace"):
            with gr.Column(scale=5, min_width=300, variant="panel", elem_classes="work-panel"):
                gr.Markdown("## Inputs")
                with gr.Row():
                    image = gr.Image(
                        type="pil", label="Reference image", image_mode="RGBA", sources=["upload"],
                        height=192, min_width=125,
                    )
                    foreground = gr.Image(
                        type="pil", label="Foreground", image_mode="RGBA", format="png",
                        interactive=False, height=192, min_width=125,
                    )
                coarse = gr.File(
                    file_types=[".obj", ".glb", ".ply"], type="filepath", label="Aligned coarse mesh",
                    height=116,
                )
                resolution = gr.Radio(list(RESOLUTIONS), value=1024, label="Output resolution")
                with gr.Accordion("Sampling and mesh options", open=False):
                    stage_inputs = []
                    for stage in RESOLUTIONS:
                        steps, cfg = defaults[stage]
                        with gr.Row():
                            stage_inputs.append(gr.Number(
                                label=f"{stage} steps", precision=0, minimum=1, value=steps,
                            ))
                            stage_inputs.append(gr.Number(label=f"{stage} CFG", minimum=0, value=cfg))
                    with gr.Row():
                        seed = gr.Number(value=2026, precision=0, label="Seed")
                        max_tokens = gr.Number(
                            value=DEFAULT_MAX_TOKENS, minimum=1, precision=0, label="Maximum tokens per stage",
                            info="If you run out of GPU memory, try a lower value. The mesh may lose fine surface detail.",
                        )
                    simplify = gr.Checkbox(value=True, label="Simplify mesh")
                    target_faces = gr.Slider(
                        100_000, 5_000_000, step=1, value=DEFAULT_TARGET_FACES, label="Target triangles",
                    )
                    simplify.change(
                        lambda enabled: gr.Slider(interactive=enabled), inputs=simplify, outputs=target_faces,
                        queue=False, show_progress="hidden", **ui_event,
                    )
                submit = gr.Button("Generate", variant="primary", elem_id="generate-mesh")
                stage_progress = gr.HTML("", elem_id="sampling-progress")

            with gr.Column(scale=7, min_width=300, variant="panel", elem_classes="work-panel"):
                gr.Markdown("## Mesh viewer")
                gr.HTML("<direct3d-mesh-viewer></direct3d-mesh-viewer>", elem_id="mesh-viewer")
                download = gr.File(label="Download GLB", interactive=False, height=80)

        demo.load(fn=None, js=VIEWER.read_text(encoding="utf-8"), **ui_event)
        download.change(
            fn=None, inputs=download, js="""(file) => {
                const viewer = document.querySelector("direct3d-mesh-viewer");
                if (viewer) viewer.setAttribute("src", file?.url || "");
            }""", **ui_event,
        )
        results = [foreground, download, stage_progress]
        job = submit.click(
            generate, [image, coarse, resolution, seed, simplify, target_faces, max_tokens, *stage_inputs],
            results, concurrency_limit=1, api_name="generate", show_progress_on=[download],
        )
        gallery.select(
            load_example, outputs=[image, coarse, *results], cancels=[job],
            queue=False, show_progress="hidden", **ui_event,
        )
        gr.on(
            triggers=[image.upload, image.clear, coarse.upload, coarse.clear],
            fn=lambda: (None, None, ""), outputs=results, cancels=[job],
            queue=False, show_progress="hidden", **ui_event,
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the Direct3D-V3 mesh generation demo.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help="Hugging Face repository ID or local model directory (default: %(default)s)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    pipeline = Direct3DV3Pipeline.from_pretrained(args.model).to(args.device)
    launch_style = {"css_paths": [STYLES]} if GRADIO_MAJOR >= 6 else {}
    build_demo(pipeline).queue().launch(
        server_name=args.host, server_port=args.port, allowed_paths=[str(TEASER)], **launch_style,
    )


if __name__ == "__main__":
    main()
