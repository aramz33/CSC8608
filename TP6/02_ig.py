import os
import sys
import time

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tp6")

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from captum.attr import IntegratedGradients, NoiseTunnel
from transformers import AutoImageProcessor, AutoModelForImageClassification

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "TP6/data/pneumo_1.jpeg"
    print(f"Analyse fine au pixel sur : {image_path}")

    image = Image.open(image_path).convert("RGB")
    model_name = "Aunsiels/resnet-pneumonia-detection"

    processor = AutoImageProcessor.from_pretrained(model_name)
    hf_model = AutoModelForImageClassification.from_pretrained(model_name)
    wrapped_model = ModelWrapper(hf_model)

    device = get_device()
    wrapped_model.to(device)
    wrapped_model.eval()
    print(f"device: {device}")

    inputs = processor(images=image, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(device)
    input_tensor.requires_grad_(True)

    _ = wrapped_model(input_tensor)
    sync_device(device)

    sync_device(device)
    start_infer = time.perf_counter()
    logits = wrapped_model(input_tensor)
    sync_device(device)
    end_infer = time.perf_counter()

    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = hf_model.config.id2label[predicted_class_idx]
    print(f"Temps d'inférence : {end_infer - start_infer:.4f} secondes")
    print(f"Classe prédite : {predicted_label}")

    ig = IntegratedGradients(wrapped_model)
    baseline = torch.zeros_like(input_tensor)

    sync_device(device)
    start_ig = time.perf_counter()
    attributions_ig = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=predicted_class_idx,
        n_steps=50,
        internal_batch_size=2,
    )
    sync_device(device)
    end_ig = time.perf_counter()

    noise_tunnel = NoiseTunnel(ig)

    sync_device(device)
    start_sg = time.perf_counter()
    attributions_sg = noise_tunnel.attribute(
        input_tensor,
        nt_samples=100,
        nt_type="smoothgrad",
        target=predicted_class_idx,
        stdevs=0.1,
        internal_batch_size=2,
    )
    sync_device(device)
    end_sg = time.perf_counter()

    print(f"Temps IG pur : {end_ig - start_ig:.4f}s")
    print(f"Temps SmoothGrad (IG x 100) : {end_sg - start_sg:.4f}s")

    attr_ig_signed = attributions_ig.squeeze().detach().cpu().numpy()
    attr_sg_signed = attributions_sg.squeeze().detach().cpu().numpy()
    attr_ig_vis = np.sum(np.abs(attr_ig_signed), axis=0)
    attr_sg_vis = np.sum(np.abs(attr_sg_signed), axis=0)

    threshold_ig = np.percentile(attr_ig_vis, 70)
    attr_ig_vis[attr_ig_vis < threshold_ig] = 0
    threshold_sg = np.percentile(attr_sg_vis, 70)
    attr_sg_vis[attr_sg_vis < threshold_sg] = 0

    vmax_ig = np.max(attr_ig_vis) if np.max(attr_ig_vis) > 0 else 1.0
    vmax_sg = np.max(attr_sg_vis) if np.max(attr_sg_vis) > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    resized_image = image.resize(input_tensor.shape[2:][::-1])
    axes[0].imshow(resized_image)
    axes[0].set_title(f"Image Originale\nPred: {predicted_label}")
    axes[0].axis("off")

    axes[1].imshow(resized_image, alpha=0.6)
    axes[1].imshow(attr_ig_vis, cmap="hot", alpha=0.6, vmin=0, vmax=vmax_ig)
    axes[1].set_title("Integrated Gradients (Seuillé)")
    axes[1].axis("off")

    axes[2].imshow(resized_image, alpha=0.6)
    axes[2].imshow(attr_sg_vis, cmap="hot", alpha=0.6, vmin=0, vmax=vmax_sg)
    axes[2].set_title("SmoothGrad (Seuillé)")
    axes[2].axis("off")

    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"TP6/outputs/ig/ig_smooth_{stem}.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Visualisation sauvegardée dans {output_filename}")

    signed_stats = {
        "ig_min": float(attr_ig_signed.min()),
        "ig_max": float(attr_ig_signed.max()),
        "sg_min": float(attr_sg_signed.min()),
        "sg_max": float(attr_sg_signed.max()),
    }
    print(f"Signed attribution stats: {signed_stats}")


if __name__ == "__main__":
    main()
