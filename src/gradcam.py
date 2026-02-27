"""
Grad-CAM Visualization for ECA-MGNet and Baseline Models

Generates publication-quality heatmap overlays comparing the proposed
ECA-MGNet model with a baseline (MobileNetV2) using Gradient-weighted
Class Activation Mapping (Grad-CAM).

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization," ICCV 2017.

Usage:
    python src/gradcam.py \
        --model_path checkpoints/ecamgnet_dataset.pth \
        --data_dir data/dataset_name \
        --model_name ecamgnet \
        --num_classes 10 \
        --output_dir figures/gradcam

    python src/gradcam.py \
        --model_path checkpoints/ecamgnet_dataset.pth \
        --baseline_path checkpoints/mobilenetv2_dataset.pth \
        --data_dir data/dataset_name \
        --num_classes 10 \
        --output_dir figures/gradcam
"""
import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Enable relative imports from the src package
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models import get_model, ECAMGNet

# ============================================================
# IEEE-quality figure settings (consistent with generate_figures.py)
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ImageNet normalization constants (used during training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# ============================================================
# Grad-CAM Implementation Using PyTorch Hooks
# ============================================================

class GradCAM:
    """Grad-CAM implementation using PyTorch forward/backward hooks.

    Computes class-discriminative localization maps by using the gradients
    of the target class flowing into the final convolutional layer.

    Attributes:
        model: The neural network model (in eval mode).
        target_layer: The nn.Module whose activations and gradients are captured.
        activations: Stored forward activations from the target layer.
        gradients: Stored gradients from the target layer.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: A PyTorch model (will be set to eval mode).
            target_layer: The nn.Module to hook into for Grad-CAM
                          (should be the last conv/attention layer before GAP).
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: store the output activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: store the gradients of the output."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: A preprocessed image tensor of shape (1, C, H, W).
            target_class: The class index to compute Grad-CAM for.
                          If None, uses the predicted class.

        Returns:
            cam: A numpy array of shape (H, W) with values in [0, 1],
                 where H and W are the spatial dimensions of the input image.
            predicted_class: The class index predicted by the model.
            confidence: The softmax probability for the predicted class.
        """
        # Forward pass
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        # Use predicted class if no target specified
        if target_class is None:
            target_class = predicted_class

        # Backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Compute Grad-CAM weights: global average pooling of gradients
        # gradients shape: (1, C, H_feat, W_feat)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # Apply ReLU (we only care about features with positive influence)
        cam = F.relu(cam)

        # Upsample to input spatial dimensions
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class, confidence

    def remove_hooks(self):
        """Remove the registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ============================================================
# Target Layer Identification
# ============================================================

def get_target_layer(model, model_name):
    """Identify the correct target layer for Grad-CAM based on model type.

    For each architecture, this returns the last layer that produces
    spatial feature maps (before global average pooling).

    Args:
        model: The instantiated PyTorch model.
        model_name: String identifier (e.g., 'ecamgnet', 'mobilenetv2').

    Returns:
        target_layer: The nn.Module to use as the Grad-CAM target.
    """
    if model_name == 'ecamgnet':
        # ECAMGNet: final_attention is the last module before classifier
        # (classifier starts with AdaptiveAvgPool2d)
        return model.final_attention
    elif model_name == 'mobilenetv2':
        # MobileNetV2: the last block in features (conv + bn + relu)
        return model.features[-1]
    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0: last block in features
        return model.features[-1]
    elif model_name == 'shufflenetv2':
        # ShuffleNetV2: conv5 is the last conv layer before FC
        return model.conv5
    elif model_name == 'resnet18':
        # ResNet-18: layer4 is the last residual block
        return model.layer4
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Cannot determine Grad-CAM target layer. "
            f"Supported: ecamgnet, mobilenetv2, efficientnet_b0, shufflenetv2, resnet18"
        )


# ============================================================
# Image Loading and Preprocessing
# ============================================================

def get_eval_transform(img_size=224):
    """Get the evaluation transform (must match training preprocessing)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(),
                             std=IMAGENET_STD.tolist()),
    ])


def load_image(image_path, img_size=224):
    """Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file.
        img_size: Target spatial size.

    Returns:
        input_tensor: Preprocessed tensor of shape (1, 3, img_size, img_size).
        original_image: The original PIL image resized to (img_size, img_size).
    """
    image = Image.open(image_path).convert('RGB')
    original_image = image.resize((img_size, img_size), Image.BILINEAR)

    transform = get_eval_transform(img_size)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return input_tensor, original_image


def denormalize_tensor(tensor):
    """Convert a normalized tensor back to a displayable numpy image.

    Args:
        tensor: A (C, H, W) or (1, C, H, W) tensor normalized with ImageNet stats.

    Returns:
        image: A numpy array of shape (H, W, 3) with values in [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return img


# ============================================================
# Heatmap Overlay Generation
# ============================================================

def create_heatmap_overlay(original_image, cam, alpha=0.5, colormap='jet'):
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: A numpy array of shape (H, W, 3) in [0, 1] or a PIL Image.
        cam: The Grad-CAM heatmap of shape (H, W) in [0, 1].
        alpha: Blending factor for the overlay (0 = only image, 1 = only heatmap).
        colormap: Matplotlib colormap name. 'jet' is traditional for Grad-CAM;
                  'viridis' is colorblind-friendly.

    Returns:
        overlay: A numpy array of shape (H, W, 3) in [0, 1].
    """
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image).astype(np.float32) / 255.0

    # Apply colormap to the CAM
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # Drop alpha channel, shape (H, W, 3)

    # Blend
    overlay = (1.0 - alpha) * original_image + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)

    return overlay


# ============================================================
# Sample Image Collection
# ============================================================

def collect_sample_images(data_dir, num_per_class=1):
    """Collect sample images from each class in the dataset directory.

    Expects directory structure:
        data_dir/
            class_a/
                img1.jpg
                ...
            class_b/
                img1.jpg
                ...

    Args:
        data_dir: Path to the dataset root directory.
        num_per_class: Number of sample images to collect per class.

    Returns:
        samples: List of (image_path, class_name, class_idx) tuples.
        class_names: Sorted list of class names.
    """
    data_dir = Path(data_dir)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    samples = []
    for idx, cls_dir in enumerate(class_dirs):
        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in supported_extensions
        ])
        selected = images[:num_per_class]
        for img_path in selected:
            samples.append((str(img_path), cls_dir.name, idx))

    return samples, class_names


# ============================================================
# Model Loading
# ============================================================

def load_model_from_checkpoint(model_name, num_classes, checkpoint_path,
                                width_mult=1.0, device='cpu'):
    """Instantiate a model and load trained weights from a checkpoint.

    Args:
        model_name: Model architecture name (e.g., 'ecamgnet', 'mobilenetv2').
        num_classes: Number of output classes.
        checkpoint_path: Path to the .pth checkpoint file.
        width_mult: Width multiplier for ECA-MGNet.
        device: Device to load the model onto.

    Returns:
        model: The model with loaded weights, in eval mode, on the specified device.
    """
    model = get_model(model_name, num_classes, pretrained=False,
                      width_mult=width_mult)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle checkpoints that wrap the state_dict in a dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# ============================================================
# Visualization: Single Image Comparison Grid
# ============================================================

def visualize_single_image(original_image, cam_proposed, cam_baseline,
                            class_name, pred_proposed, conf_proposed,
                            pred_baseline, conf_baseline,
                            class_names=None, colormap='jet'):
    """Create a single-row comparison: Original | ECA-MGNet | Baseline.

    Args:
        original_image: PIL Image or numpy array (H, W, 3).
        cam_proposed: Grad-CAM heatmap from ECA-MGNet, shape (H, W).
        cam_baseline: Grad-CAM heatmap from baseline model, shape (H, W).
        class_name: Ground truth class name.
        pred_proposed: Predicted class index from ECA-MGNet.
        conf_proposed: Confidence from ECA-MGNet.
        pred_baseline: Predicted class index from baseline.
        conf_baseline: Confidence from baseline.
        class_names: List of class name strings (for converting indices to names).
        colormap: Colormap for heatmap rendering.

    Returns:
        fig: The matplotlib Figure object.
    """
    if isinstance(original_image, Image.Image):
        orig_np = np.array(original_image).astype(np.float32) / 255.0
    else:
        orig_np = original_image

    overlay_proposed = create_heatmap_overlay(orig_np, cam_proposed,
                                              alpha=0.5, colormap=colormap)
    overlay_baseline = create_heatmap_overlay(orig_np, cam_baseline,
                                              alpha=0.5, colormap=colormap)

    def _class_label(idx):
        if class_names and 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

    # Original image
    axes[0].imshow(orig_np)
    axes[0].set_title(f'Original\nGT: {class_name}', fontsize=10)
    axes[0].axis('off')

    # ECA-MGNet heatmap
    im1 = axes[1].imshow(overlay_proposed)
    pred_label = _class_label(pred_proposed)
    axes[1].set_title(
        f'ECA-MGNet (Proposed)\nPred: {pred_label} ({conf_proposed:.1%})',
        fontsize=10
    )
    axes[1].axis('off')

    # Baseline heatmap
    axes[2].imshow(overlay_baseline)
    pred_label = _class_label(pred_baseline)
    axes[2].set_title(
        f'MobileNetV2 (Baseline)\nPred: {pred_label} ({conf_baseline:.1%})',
        fontsize=10
    )
    axes[2].axis('off')

    plt.tight_layout()
    return fig


# ============================================================
# Visualization: Full Grid Figure
# ============================================================

def generate_gradcam_grid(samples, class_names, model_proposed, model_baseline,
                           model_name_proposed, model_name_baseline,
                           output_path, device='cpu', img_size=224,
                           colormap='jet'):
    """Generate a publication-quality grid figure with Grad-CAM comparisons.

    Each row corresponds to one sample image. Columns are:
        [Original Image] | [ECA-MGNet Heatmap] | [Baseline Heatmap]

    Args:
        samples: List of (image_path, class_name, class_idx) tuples.
        class_names: List of class name strings.
        model_proposed: The proposed ECA-MGNet model.
        model_baseline: The baseline model (e.g., MobileNetV2).
        model_name_proposed: Name string for proposed model.
        model_name_baseline: Name string for baseline model.
        output_path: Path (without extension) to save the output figure.
        device: Torch device.
        img_size: Input image size.
        colormap: Colormap name for heatmaps.
    """
    n_samples = len(samples)
    if n_samples == 0:
        print("No sample images found. Skipping Grad-CAM grid generation.")
        return

    # Set up Grad-CAM for both models
    target_proposed = get_target_layer(model_proposed, model_name_proposed)
    target_baseline = get_target_layer(model_baseline, model_name_baseline)

    gradcam_proposed = GradCAM(model_proposed, target_proposed)
    gradcam_baseline = GradCAM(model_baseline, target_baseline)

    # Determine grid layout
    n_cols = 3  # Original | Proposed | Baseline
    fig_width = 3.2 * n_cols
    fig_height = 3.0 * n_samples + 0.6  # Extra space for suptitle

    fig, axes = plt.subplots(n_samples, n_cols,
                              figsize=(fig_width, fig_height),
                              squeeze=False)

    for row_idx, (img_path, class_name, class_idx) in enumerate(samples):
        print(f"  Processing [{row_idx + 1}/{n_samples}]: {class_name} - "
              f"{Path(img_path).name}")

        input_tensor, original_image = load_image(img_path, img_size)
        input_tensor = input_tensor.to(device)
        orig_np = np.array(original_image).astype(np.float32) / 255.0

        # Generate Grad-CAMs
        cam_p, pred_p, conf_p = gradcam_proposed.generate(input_tensor)
        cam_b, pred_b, conf_b = gradcam_baseline.generate(input_tensor)

        overlay_p = create_heatmap_overlay(orig_np, cam_p, alpha=0.5,
                                           colormap=colormap)
        overlay_b = create_heatmap_overlay(orig_np, cam_b, alpha=0.5,
                                           colormap=colormap)

        def _label(idx):
            if class_names and 0 <= idx < len(class_names):
                return class_names[idx]
            return str(idx)

        # Column 0: Original
        axes[row_idx, 0].imshow(orig_np)
        axes[row_idx, 0].set_ylabel(class_name, fontsize=10, fontweight='bold',
                                     rotation=90, labelpad=10)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 0].set_title('Original', fontsize=11, fontweight='bold')

        # Column 1: ECA-MGNet
        axes[row_idx, 1].imshow(overlay_p)
        pred_text = f'{_label(pred_p)} ({conf_p:.0%})'
        axes[row_idx, 1].set_xlabel(pred_text, fontsize=8)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 1].set_title('ECA-MGNet (Proposed)',
                                        fontsize=11, fontweight='bold')

        # Column 2: Baseline
        axes[row_idx, 2].imshow(overlay_b)
        pred_text = f'{_label(pred_b)} ({conf_b:.0%})'
        axes[row_idx, 2].set_xlabel(pred_text, fontsize=8)
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 2].set_title('MobileNetV2 (Baseline)',
                                        fontsize=11, fontweight='bold')

    # Add a colorbar on the right side
    cmap = plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, aspect=40)
    cbar.set_label('Grad-CAM Activation Intensity', fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle('Grad-CAM Visualization: ECA-MGNet vs. MobileNetV2',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()

    # Save in multiple formats for publication
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(f"{output_path}.png", format='png', dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    fig.savefig(f"{output_path}.pdf", format='pdf',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"  Saved: {output_path}.png")
    print(f"  Saved: {output_path}.pdf")

    # Clean up hooks
    gradcam_proposed.remove_hooks()
    gradcam_baseline.remove_hooks()


# ============================================================
# Visualization: Individual Heatmaps per Image
# ============================================================

def generate_individual_heatmaps(samples, class_names,
                                  model_proposed, model_baseline,
                                  model_name_proposed, model_name_baseline,
                                  output_dir, device='cpu', img_size=224,
                                  colormap='jet'):
    """Save individual Grad-CAM comparison figures for each sample image.

    Args:
        samples: List of (image_path, class_name, class_idx) tuples.
        class_names: List of class name strings.
        model_proposed: The proposed ECA-MGNet model.
        model_baseline: The baseline model.
        model_name_proposed: Name string for proposed model.
        model_name_baseline: Name string for baseline model.
        output_dir: Directory to save individual figures.
        device: Torch device.
        img_size: Input image size.
        colormap: Colormap name for heatmaps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_proposed = get_target_layer(model_proposed, model_name_proposed)
    target_baseline = get_target_layer(model_baseline, model_name_baseline)

    gradcam_proposed = GradCAM(model_proposed, target_proposed)
    gradcam_baseline = GradCAM(model_baseline, target_baseline)

    for idx, (img_path, class_name, class_idx) in enumerate(samples):
        print(f"  Individual [{idx + 1}/{len(samples)}]: {class_name} - "
              f"{Path(img_path).name}")

        input_tensor, original_image = load_image(img_path, img_size)
        input_tensor = input_tensor.to(device)

        cam_p, pred_p, conf_p = gradcam_proposed.generate(input_tensor)
        cam_b, pred_b, conf_b = gradcam_baseline.generate(input_tensor)

        fig = visualize_single_image(
            original_image, cam_p, cam_b,
            class_name, pred_p, conf_p, pred_b, conf_b,
            class_names=class_names, colormap=colormap,
        )

        fname = f"gradcam_{class_name}_{idx:03d}"
        fig.savefig(output_dir / f"{fname}.png", format='png', dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
        fig.savefig(output_dir / f"{fname}.pdf", format='pdf',
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    gradcam_proposed.remove_hooks()
    gradcam_baseline.remove_hooks()

    print(f"  Individual heatmaps saved to: {output_dir}")


# ============================================================
# Visualization: Raw Heatmap (without overlay) for each model
# ============================================================

def generate_raw_heatmaps(samples, model, model_name, output_dir,
                           device='cpu', img_size=224, colormap='jet'):
    """Save standalone Grad-CAM heatmaps (no overlay) for a single model.

    Useful for supplementary material or detailed analysis.

    Args:
        samples: List of (image_path, class_name, class_idx) tuples.
        model: The PyTorch model.
        model_name: Name string for the model.
        output_dir: Directory to save heatmap images.
        device: Torch device.
        img_size: Input image size.
        colormap: Colormap name.
    """
    output_dir = Path(output_dir) / f"raw_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    for idx, (img_path, class_name, class_idx) in enumerate(samples):
        input_tensor, original_image = load_image(img_path, img_size)
        input_tensor = input_tensor.to(device)

        cam, pred, conf = gradcam.generate(input_tensor)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        # Original
        orig_np = np.array(original_image).astype(np.float32) / 255.0
        axes[0].imshow(orig_np)
        axes[0].set_title('Original', fontsize=10)
        axes[0].axis('off')

        # Raw heatmap
        im = axes[1].imshow(cam, cmap=colormap, vmin=0, vmax=1)
        axes[1].set_title(f'{model_name} Grad-CAM', fontsize=10)
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        fname = f"raw_{model_name}_{class_name}_{idx:03d}"
        fig.savefig(output_dir / f"{fname}.png", dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    gradcam.remove_hooks()
    print(f"  Raw heatmaps for {model_name} saved to: {output_dir}")


# ============================================================
# Main Entry Point
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Grad-CAM Visualization for ECA-MGNet and Baseline Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with ECA-MGNet checkpoint
  python src/gradcam.py \\
      --model_path checkpoints/ecamgnet_beans.pth \\
      --data_dir data/beans \\
      --num_classes 3

  # With explicit baseline checkpoint
  python src/gradcam.py \\
      --model_path checkpoints/ecamgnet_beans.pth \\
      --baseline_path checkpoints/mobilenetv2_beans.pth \\
      --data_dir data/beans \\
      --num_classes 3

  # Custom settings
  python src/gradcam.py \\
      --model_path checkpoints/ecamgnet_cifar10.pth \\
      --data_dir data/cifar10 \\
      --num_classes 10 \\
      --width_mult 1.0 \\
      --num_per_class 2 \\
      --colormap viridis \\
      --output_dir figures/gradcam_cifar10
        """,
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained ECA-MGNet .pth checkpoint.')
    parser.add_argument('--baseline_path', type=str, default=None,
                        help='Path to the trained baseline .pth checkpoint. '
                             'If not provided, the baseline will use ImageNet '
                             'pretrained weights (or random if unavailable).')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory (class-per-folder '
                             'structure).')
    parser.add_argument('--model_name', type=str, default='ecamgnet',
                        choices=['ecamgnet'],
                        help='Proposed model architecture name (default: ecamgnet).')
    parser.add_argument('--baseline_name', type=str, default='mobilenetv2',
                        choices=['mobilenetv2', 'efficientnet_b0',
                                 'shufflenetv2', 'resnet18'],
                        help='Baseline model architecture name '
                             '(default: mobilenetv2).')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of output classes.')
    parser.add_argument('--width_mult', type=float, default=1.0,
                        help='Width multiplier for ECA-MGNet (default: 1.0).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output figures. Defaults to '
                             '<repo_root>/figures/gradcam/')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224).')
    parser.add_argument('--num_per_class', type=int, default=1,
                        help='Number of sample images per class (default: 1).')
    parser.add_argument('--colormap', type=str, default='jet',
                        choices=['jet', 'viridis', 'inferno', 'magma',
                                 'plasma', 'hot', 'coolwarm'],
                        help='Colormap for heatmap rendering (default: jet). '
                             'Use viridis for colorblind-friendly output.')
    parser.add_argument('--no_individual', action='store_true',
                        help='Skip generating individual per-image figures.')
    parser.add_argument('--no_raw', action='store_true',
                        help='Skip generating raw (non-overlay) heatmaps.')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device (default: auto-detect cuda/cpu).')

    return parser.parse_args()


def main():
    """Main function: load models, collect samples, generate Grad-CAM figures."""
    args = parse_args()

    # Resolve output directory
    repo_root = Path(__file__).resolve().parent.parent
    if args.output_dir is None:
        output_dir = repo_root / 'figures' / 'gradcam'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 65)
    print("Grad-CAM Visualization for ECA-MGNet")
    print("=" * 65)
    print(f"  Proposed model : {args.model_name}")
    print(f"  Baseline model : {args.baseline_name}")
    print(f"  Checkpoint     : {args.model_path}")
    print(f"  Baseline ckpt  : {args.baseline_path or '(pretrained / random)'}")
    print(f"  Dataset        : {args.data_dir}")
    print(f"  Num classes    : {args.num_classes}")
    print(f"  Width mult     : {args.width_mult}")
    print(f"  Image size     : {args.img_size}")
    print(f"  Colormap       : {args.colormap}")
    print(f"  Device         : {device}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load proposed model (ECA-MGNet)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading proposed model...")
    model_proposed = load_model_from_checkpoint(
        model_name=args.model_name,
        num_classes=args.num_classes,
        checkpoint_path=args.model_path,
        width_mult=args.width_mult,
        device=device,
    )
    print(f"  Loaded {args.model_name} from {args.model_path}")

    # ------------------------------------------------------------------
    # 2. Load baseline model
    # ------------------------------------------------------------------
    print("\n[2/4] Loading baseline model...")
    if args.baseline_path:
        model_baseline = load_model_from_checkpoint(
            model_name=args.baseline_name,
            num_classes=args.num_classes,
            checkpoint_path=args.baseline_path,
            device=device,
        )
        print(f"  Loaded {args.baseline_name} from {args.baseline_path}")
    else:
        # Use pretrained weights; replace classifier head for num_classes
        model_baseline = get_model(args.baseline_name, args.num_classes,
                                   pretrained=True)
        model_baseline.to(device)
        model_baseline.eval()
        print(f"  Using pretrained {args.baseline_name} (classifier head "
              f"randomly initialized for {args.num_classes} classes)")

    # ------------------------------------------------------------------
    # 3. Collect sample images
    # ------------------------------------------------------------------
    print("\n[3/4] Collecting sample images...")
    samples, class_names = collect_sample_images(args.data_dir,
                                                  num_per_class=args.num_per_class)
    print(f"  Found {len(samples)} samples across {len(class_names)} classes: "
          f"{class_names}")

    if len(samples) == 0:
        print("\nERROR: No images found in the dataset directory.")
        print(f"  Expected structure: {args.data_dir}/<class_name>/<image_files>")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Generate Grad-CAM visualizations
    # ------------------------------------------------------------------
    print("\n[4/4] Generating Grad-CAM visualizations...")

    # (a) Grid figure (main publication figure)
    print("\n  --- Grid Figure ---")
    grid_path = output_dir / "gradcam_comparison"
    generate_gradcam_grid(
        samples=samples,
        class_names=class_names,
        model_proposed=model_proposed,
        model_baseline=model_baseline,
        model_name_proposed=args.model_name,
        model_name_baseline=args.baseline_name,
        output_path=grid_path,
        device=device,
        img_size=args.img_size,
        colormap=args.colormap,
    )

    # (b) Individual per-image comparison figures
    if not args.no_individual:
        print("\n  --- Individual Figures ---")
        generate_individual_heatmaps(
            samples=samples,
            class_names=class_names,
            model_proposed=model_proposed,
            model_baseline=model_baseline,
            model_name_proposed=args.model_name,
            model_name_baseline=args.baseline_name,
            output_dir=output_dir / "individual",
            device=device,
            img_size=args.img_size,
            colormap=args.colormap,
        )

    # (c) Raw heatmaps (no overlay)
    if not args.no_raw:
        print("\n  --- Raw Heatmaps ---")
        generate_raw_heatmaps(
            samples=samples,
            model=model_proposed,
            model_name=args.model_name,
            output_dir=output_dir,
            device=device,
            img_size=args.img_size,
            colormap=args.colormap,
        )
        generate_raw_heatmaps(
            samples=samples,
            model=model_baseline,
            model_name=args.baseline_name,
            output_dir=output_dir,
            device=device,
            img_size=args.img_size,
            colormap=args.colormap,
        )

    print("\n" + "=" * 65)
    print("Grad-CAM visualization complete!")
    print(f"Output saved to: {output_dir}")
    print("=" * 65)


if __name__ == '__main__':
    main()
