"""
First-pass signal screen: Tata Li's CNN waterfall classifier.

Copied from https://github.com/tlofbj/waterfall-classifier-project-v3.5cnn-lite
(MIT licence, see models/LICENSE-waterfall-cnn.txt). The network and the image
preprocessing are unchanged; only the decision threshold is ours.

The model looks at the SatNOGS waterfall PNG (not the audio) and returns a
probability that it contains a signal. Its original cut-off of 0.5 calls almost
every FUNcube-1 waterfall "signal": in our test of 40 passes even empty ones
scored 0.59. The scores still rank passes well, though, so we re-set the cut-off
to CNN_THRESHOLD. Above it, the pass goes on to the audio check in
validate_signals.py; below it, the pass is rejected straight away.

PyTorch is optional. Without it, predict() returns None and the audio check
runs on its own.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "waterfall_cnn_2025-05-08.pth"

# Recalibrated on 40 hand-labelled FUNcube-1 passes (docs/signal_validation.md):
# every real signal scored >= 0.62 and the typical empty waterfall scores 0.59.
# 0.60 keeps all real signals and removes about half of the empty passes; the
# audio check makes the final call on whatever gets through.
CNN_THRESHOLD = 0.60

# SatNOGS waterfall layout: the plot area inside the axes, in pixels.
CROP_BOX = (67, 11, 686, 1550)

_model = None


def _load():
    """Build the network and load the weights once. Returns None if torch is missing."""
    global _model
    if _model is not None:
        return _model
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4, 4),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 1), nn.Sigmoid(),
    )
    # The original wraps the layers in a module attribute called "sequential".
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    net.load_state_dict({k.removeprefix("sequential."): v for k, v in state.items()})
    net.eval()
    _model = net
    return _model


def _image_tensor(png_path):
    """Grayscale, crop to the plot area, resize to 256x256, scale to [-1, 1]."""
    from PIL import Image
    from torchvision import transforms

    image = Image.open(png_path).convert("L").crop(CROP_BOX)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(image).unsqueeze(0)


def predict(png_path):
    """Probability (0-1) that this SatNOGS waterfall has a signal, or None if unavailable."""
    if png_path is None or not Path(png_path).exists():
        return None
    model = _load()
    if model is None:
        return None
    import torch

    with torch.no_grad():
        return float(model(_image_tensor(png_path)).item())
