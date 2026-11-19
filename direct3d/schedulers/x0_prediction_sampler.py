import torch
from tqdm.auto import tqdm


@torch.inference_mode()
def sample_x0(
    model,
    latents_index,
    cond: torch.Tensor,
    generator,
    *,
    patch_size,
    steps,
    guidance_scale,
    cfg_interval,
    resolution: int | None = None,
    clamp_range: tuple[float, float] = (-1.0, 1.0),
    on_sampling_step=None,
):
    clamp_min, clamp_max = clamp_range
    if not clamp_min <= clamp_max:
        raise ValueError("clamp_range requires a lower bound less than or equal to its upper bound.")
    device = latents_index.device
    # The entry script passes a float32 image into sample() inside CUDA autocast.
    latents = torch.randn(
        (len(latents_index), patch_size**3 * model.out_channels),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    layout = [slice(0, len(latents_index))]
    uncond = torch.zeros_like(cond) if guidance_scale > 1 else None
    timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
    desc = "Diffusion sampling"
    if resolution is not None:
        desc += f" resolution={resolution}"
    for i in tqdm(range(steps), desc=f"{desc} shape={tuple(latents.shape)}"):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        timestep = t.reshape(1).to(latents.dtype)
        x_cond = model(latents, latents_index, layout, timestep, cond)
        if uncond is not None:
            x_uncond = model(latents, latents_index, layout, timestep, uncond)
            low, high = cfg_interval
            interval_mask = (t < high) & ((low == 0) | (t > low))
            cfg_scale_interval = torch.where(interval_mask, guidance_scale, 1.0)
            output = x_uncond + cfg_scale_interval * (x_cond - x_uncond)
        else:
            output = x_cond
        output = output.clamp(clamp_min, clamp_max)
        noise = torch.randn(output.shape, device=device, dtype=output.dtype, generator=generator)
        latents = output * t_next + (1 - t_next) * noise
        if on_sampling_step is not None:
            on_sampling_step(resolution, i + 1, steps)
    return latents
