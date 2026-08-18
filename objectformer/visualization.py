import os

import numpy as np
from PIL import Image


DEFAULT_PALETTE = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    255: (0, 0, 0),
}


def evenly_spaced_indices(length, count):
    length, count = int(length), int(count)
    if length <= 0 or count <= 0:
        return []
    return np.linspace(0, length - 1, min(length, count), dtype=int).tolist()


def colorize_labels(label, palette=None):
    palette = palette or DEFAULT_PALETTE
    label = np.asarray(label)
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        rgb[label == int(class_id)] = color
    return rgb


def save_label_panels(labels, path, gap=8):
    """Save ordered colorized label panels without resizing their masks."""
    panels = [colorize_labels(label) for label in labels]
    if not panels:
        raise ValueError("At least one label panel is required")
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels) + int(gap) * (len(panels) - 1)
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    x = 0
    for panel in panels:
        canvas[:panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + int(gap)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)
