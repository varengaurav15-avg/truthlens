import base64
import io
from typing import Optional

import numpy as np
import torch


def grad_cam_saliency(
    model, tensor: torch.Tensor, target_layer=None
) -> np.ndarray:
    """
    Compute a Grad-CAM saliency map for the 'fake' class (index 1).

    Returns a 2-D numpy array in [0, 1] at the resolution of `target_layer`'s
    feature map (upsampled to input resolution by the caller if needed).
    Falls back to input-gradient saliency when no target_layer is provided.
    """
    model.eval()
    # Detach from any existing graph, then re-enable grad
    tensor = tensor.detach().requires_grad_(True)

    acts: Optional[torch.Tensor] = None
    grads: Optional[torch.Tensor] = None
    hooks = []

    if target_layer is not None:
        def fw_hook(m, i, o):
            nonlocal acts
            acts = o  # keep as tensor so backward can flow through

        def bw_hook(m, i, o):
            nonlocal grads
            grads = o[0].detach()

        hooks.append(target_layer.register_forward_hook(fw_hook))
        hooks.append(target_layer.register_full_backward_hook(bw_hook))

    out = model(tensor)
    fake_logit = out[0, min(1, out.shape[1] - 1)]
    fake_logit.backward()

    for h in hooks:
        h.remove()

    if acts is not None and grads is not None:
        acts_np = acts.detach()[0].cpu().numpy()   # C x H x W
        grads_np = grads[0].cpu().numpy()          # C x H x W
        weights = grads_np.mean(axis=(1, 2))       # C
        cam = (weights[:, None, None] * acts_np).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        return cam / (cam_max + 1e-9) if cam_max > 0 else cam
    else:
        # Fallback: integrated-gradient approximation via input gradient
        grad = tensor.grad
        if grad is not None:
            saliency = grad[0].cpu().numpy().abs().sum(axis=0)
        else:
            saliency = np.zeros(tensor.shape[2:])
        sal_max = saliency.max()
        return saliency / (sal_max + 1e-9) if sal_max > 0 else saliency


def saliency_to_heatmap_b64(
    saliency: np.ndarray,
    orig_size: tuple = (224, 224),
    colormap: str = "hot",
) -> str:
    """
    Resize the saliency map to `orig_size`, apply a colormap, and return
    a base64-encoded PNG string suitable for embedding in JSON API responses.
    """
    from PIL import Image
    import cv2

    # Resize to original image dimensions
    h, w = orig_size
    sal_resized = np.array(
        Image.fromarray((saliency * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
    )

    # Apply colormap (OpenCV uses BGR)
    cmap_id = cv2.COLORMAP_HOT if colormap == "hot" else cv2.COLORMAP_JET
    heatmap_bgr = cv2.applyColorMap(sal_resized, cmap_id)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    buf = io.BytesIO()
    Image.fromarray(heatmap_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_efficientnet_target_layer(model):
    """Return the last convolutional block of EfficientNet-B0 for Grad-CAM."""
    try:
        # timm EfficientNet
        return model.blocks[-1]
    except (AttributeError, IndexError):
        try:
            # torchvision EfficientNet
            return model.features[-1]
        except (AttributeError, IndexError):
            return None
