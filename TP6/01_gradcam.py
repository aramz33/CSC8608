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
from captum.attr import LayerAttribution, LayerGradCam, visualization as viz
from transformers import AutoImageProcessor, AutoModelForImageClassification

matplotlib.use("Agg")


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
    image_path = sys.argv[1] if len(sys.argv) > 1 else "TP6/data/normal_1.jpeg"
    print(f"Analyse de l'image : {image_path}")

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

    target_layer = wrapped_model.model.resnet.encoder.stages[-1].layers[-1]

    sync_device(device)
    start_xai = time.perf_counter()
    layer_gradcam = LayerGradCam(wrapped_model, target_layer)
    attributions_gradcam = layer_gradcam.attribute(input_tensor, target=predicted_class_idx)
    sync_device(device)
    end_xai = time.perf_counter()

    print(f"Temps d'explicabilité (Grad-CAM) : {end_xai - start_xai:.4f} secondes")

    upsampled_attr = LayerAttribution.interpolate(attributions_gradcam, input_tensor.shape[2:])
    original_img_np = np.array(image.resize(input_tensor.shape[2:][::-1]))
    attr_gradcam_np = upsampled_attr.squeeze().detach().cpu().numpy()
    attr_gradcam_np = np.expand_dims(attr_gradcam_np, axis=2)

    fig, _ = viz.visualize_image_attr(
        attr_gradcam_np,
        original_img_np,
        method="blended_heat_map",
        sign="positive",
        show_colorbar=True,
        title=f"Grad-CAM - Pred: {predicted_label}",
        use_pyplot=False,
    )

    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"TP6/outputs/gradcam/gradcam_{stem}.png"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Visualisation sauvegardée dans {output_filename}")


if __name__ == "__main__":
    main()
