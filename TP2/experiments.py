from __future__ import annotations

import os
from PIL import Image
from pipeline_utils import DEFAULT_MODEL_ID, load_text2img, to_img2img, get_device, make_generator


def save(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def run_text2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    prompt = "ultra-realistic product photo of a leather backpack on a white background, studio lighting, soft shadow, very sharp, 4k"
    negative = "text, watermark, logo, low quality, blurry, deformed"

    plan = [
        ("run01_baseline", "EulerA", 30, 7.5),
        ("run02_steps15",  "EulerA", 15, 7.5),
        ("run03_steps50",  "EulerA", 50, 7.5),
        ("run04_guid4",    "EulerA", 30, 4.0),
        ("run05_guid12",   "EulerA", 30, 12.0),
        ("run06_ddim",     "DDIM",   30, 7.5),
    ]

    pipe = load_text2img(model_id, "EulerA")
    device = get_device()

    for name, scheduler_name, steps, guidance in plan:
        print(f"\n[T2I] {name} | scheduler={scheduler_name} seed={seed} steps={steps} guidance={guidance}")
        from pipeline_utils import set_scheduler
        set_scheduler(pipe, scheduler_name)
        g = make_generator(seed, device)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=g,
        )

        img = out.images[0]
        save(img, f"outputs/t2i_{name}.png")
        print(f"[T2I] saved outputs/t2i_{name}.png")


def run_img2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    scheduler_name = "EulerA"
    steps = 30
    guidance = 7.5

    init_path = "inputs/product.jpg"
    prompt = "ultra-realistic product photo of a leather backpack on a clean white background, studio lighting, sharp details, professional photography"
    negative = "text, watermark, logo, low quality, blurry, deformed"

    strengths = [
        ("run07_strength035", 0.35),
        ("run08_strength060", 0.60),
        ("run09_strength085", 0.85),
    ]

    pipe_t2i = load_text2img(model_id, scheduler_name)
    pipe_i2i = to_img2img(pipe_t2i)
    device = get_device()

    init_image = Image.open(init_path).convert("RGB")

    for name, strength in strengths:
        print(f"\n[I2I] {name} | strength={strength} scheduler={scheduler_name} seed={seed} steps={steps} guidance={guidance}")
        g = make_generator(seed, device)

        out = pipe_i2i(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
        )
        img = out.images[0]
        save(img, f"outputs/i2i_{name}.png")
        print(f"[I2I] saved outputs/i2i_{name}.png")


def main() -> None:
    run_text2img_experiments()
    run_img2img_experiments()


if __name__ == "__main__":
    main()
