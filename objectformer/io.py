import json
import os
from copy import deepcopy

import numpy as np
import yaml
from PIL import Image


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(os.path.expandvars(f.read()))


def apply_overrides(cfg, overrides):
    """Apply validated dotted-key YAML overrides for reproducible ablations."""
    resolved = deepcopy(cfg)
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"Override must use key=value syntax: {expression}")
        dotted_key, raw_value = expression.split("=", 1)
        keys = [key for key in dotted_key.strip().split(".") if key]
        if not keys:
            raise ValueError(f"Override has an empty key: {expression}")
        cursor = resolved
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                raise KeyError(f"Unknown override path: {dotted_key}")
            cursor = cursor[key]
        if keys[-1] not in cursor:
            raise KeyError(f"Unknown override key: {dotted_key}")
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return resolved


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_tif(arr, path):
    ensure_dir(os.path.dirname(path))
    Image.fromarray(np.asarray(arr)).save(path)


def save_color(mask, path, palette=None, ignore_index=255):
    palette = palette or {
        0: (255, 255, 255),
        1: (0, 0, 255),
        2: (0, 255, 255),
        3: (0, 255, 0),
        4: (255, 255, 0),
        5: (255, 0, 0),
        ignore_index: (0, 0, 0),
    }
    mask = np.asarray(mask)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in palette.items():
        rgb[mask == int(cid)] = color
    ensure_dir(os.path.dirname(path))
    Image.fromarray(rgb).save(path)
