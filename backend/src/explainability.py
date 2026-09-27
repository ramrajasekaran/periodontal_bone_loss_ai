import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# NOTE: pytorch_grad_cam is intentionally NOT imported here.
# Its dependency chain (sklearn -> scipy) loads a Cython DLL that is
# blocked by Windows Application Control on this machine.
# We implement GradCAM manually using pure PyTorch hooks + OpenCV.

import sys
# Insert this file's own directory so imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PeriodontalDataset, get_transforms
from models import get_model


# ---------------------------------------------------------------------------
# Manual GradCAM — no external dependency beyond PyTorch + NumPy + OpenCV
# ---------------------------------------------------------------------------

def _get_target_layer(model, model_name):
    """Return the target convolutional layer for GradCAM."""
    if model_name == 'cnn':
        return model.features[18]
    elif model_name == 'resnet50':
        return model.backbone.layer4[-1]
    elif model_name == 'efficientnet':
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _apply_colormap_on_image(img_float, mask):
    """
    Overlay a [0,1] grayscale GradCAM mask on a [0,1] RGB float image.
    Returns a uint8 RGB visualization.
    """
    mask_u8 = np.ascontiguousarray(np.clip(255 * mask, 0, 255), dtype=np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    cam = heatmap + img_float
    cam = cam / (cam.max() + 1e-8)
    return np.uint8(255 * cam)


def _compute_gradcam(model, target_layer, input_tensor, target_class=None):
    """
    Compute a GradCAM heatmap via PyTorch forward/backward hooks.

    Returns
    -------
    grayscale_cam : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    gradients = []
    activations = []

    def _fwd_hook(module, inp, out):
        activations.append(out.detach())

    def _bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        model.eval()
        output = model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    grads = gradients[0]        # (1, C, H, W)
    acts = activations[0]       # (1, C, H, W)

    # Global-average-pool the gradients then weight the activation maps
    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze(0)   # (H, W)
    cam = torch.relu(cam).cpu().numpy()

    # Resize to 224×224 and normalize to [0, 1]
    cam = cv2.resize(cam, (224, 224))
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


class _ModelWrapper(torch.nn.Module):
    """Wraps a multimodal model so GradCAM only needs a single-tensor forward."""
    def __init__(self, model, tabular_features):
        super().__init__()
        self.model = model
        self.tabular_features = tabular_features

    def forward(self, x):
        return self.model(x, self.tabular_features)


# ---------------------------------------------------------------------------
# Public API — same signature as before
# ---------------------------------------------------------------------------

def generate_gradcam(model_name, img_path, original_img_path, target_class=None, model_dir=None):
    """
    Generate a GradCAM visualization for a periodontal X-ray.

    Parameters
    ----------
    model_name        : str  — 'cnn', 'resnet50', or 'efficientnet'
    img_path          : str  — path to the (possibly temp) image file
    original_img_path : str  — original filename (used for metadata lookup)
    target_class      : int or None — GradCAM target class (None = argmax)
    model_dir         : str or None — directory containing best_*.pth weights

    Returns
    -------
    rgb_img       : np.ndarray  float32 (224,224,3) in [0,1]  preprocessed image
    grayscale_cam : np.ndarray  float32 (224,224)   in [0,1]  heatmap
    visualization : np.ndarray  uint8   (224,224,3)           overlay
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name).to(device)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    model_weight_path = os.path.join(model_dir, f'best_{model_name}.pth')
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval()

    # ------------------------------------------------------------------
    # Load tabular features from metadata CSV (if available)
    # ------------------------------------------------------------------
    import pandas as pd
    meta_csv_path = os.path.join(model_dir, '..', 'data', 'meta_data.csv')
    tabular_features = torch.zeros((1, 6), dtype=torch.float32)
    filename = os.path.basename(original_img_path)
    if os.path.exists(meta_csv_path):
        df = pd.read_csv(meta_csv_path)
        patient_row = df[df['File name'] == filename]
        if not patient_row.empty:
            row = patient_row.iloc[0]
            age             = float(row['Age']) / 100.0
            gender          = float(row['Gender']) / 2.0
            missing_teeth   = float(row['Number of missing teeth']) / 32.0
            implant         = float(row['Implant'])
            residual_root   = float(row['Residual root'])
            func_tooth_log  = float(row['Functional tooth logarithm']) / 32.0
            tabular_features = torch.tensor(
                [[age, gender, missing_teeth, implant, residual_root, func_tooth_log]],
                dtype=torch.float32
            )
    tabular_features = tabular_features.to(device)

    # ------------------------------------------------------------------
    # Preprocess image
    # ------------------------------------------------------------------
    _, val_transform = get_transforms()
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(np.ascontiguousarray(image, dtype=np.uint8))
    image_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(image_rgb)
    transformed = val_transform(pil_img)
    input_tensor = torch.as_tensor(transformed).unsqueeze(0).to(device)

    # Float [0,1] RGB for overlay
    rgb_img = np.float32(cv2.resize(image_rgb, (224, 224))) / 255.0

    # ------------------------------------------------------------------
    # GradCAM
    # ------------------------------------------------------------------
    wrapped_model = _ModelWrapper(model, tabular_features)
    target_layer  = _get_target_layer(model, model_name)

    grayscale_cam = _compute_gradcam(
        model=wrapped_model,
        target_layer=target_layer,
        input_tensor=input_tensor,
        target_class=target_class,
    )

    visualization = _apply_colormap_on_image(rgb_img, grayscale_cam)

    return rgb_img, grayscale_cam, visualization


# ---------------------------------------------------------------------------
# CLI entry-point (unchanged behaviour)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--img_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--out_dir', type=str, default='../results/figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rgb_img, heatmap, viz = generate_gradcam(args.model, args.img_path, args.img_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original (Preprocessed)")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(viz)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    out_path = os.path.join(args.out_dir, f"gradcam_{os.path.basename(args.img_path)}")
    plt.savefig(out_path)
    print(f"Saved Grad-CAM visualization to {out_path}")
