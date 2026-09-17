# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dinov3.hub.backbones import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)

try:
    from dinov3.hub.classifiers import dinov3_vit7b16_lc
except Exception as exc:  # pragma: no cover - optional dependency path
    def dinov3_vit7b16_lc(*args, **kwargs):
        raise ImportError("dinov3_vit7b16_lc is unavailable in this environment") from exc

try:
    from dinov3.hub.detectors import dinov3_vit7b16_de
except Exception as exc:  # pragma: no cover - optional dependency path
    def dinov3_vit7b16_de(*args, **kwargs):
        raise ImportError("dinov3_vit7b16_de is unavailable in this environment") from exc

try:
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
except Exception as exc:  # pragma: no cover - optional dependency path
    def dinov3_vitl16_dinotxt_tet1280d20h24l(*args, **kwargs):
        raise ImportError("dinov3_vitl16_dinotxt_tet1280d20h24l is unavailable in this environment") from exc

try:
    from dinov3.hub.segmentors import dinov3_vit7b16_ms
except Exception as exc:  # pragma: no cover - optional dependency path
    def dinov3_vit7b16_ms(*args, **kwargs):
        raise ImportError("dinov3_vit7b16_ms is unavailable in this environment") from exc

try:
    from dinov3.hub.depthers import dinov3_vit7b16_dd
except Exception as exc:  # pragma: no cover - optional dependency path
    def dinov3_vit7b16_dd(*args, **kwargs):
        raise ImportError("dinov3_vit7b16_dd is unavailable in this environment") from exc

dependencies = ["torch", "numpy"]
