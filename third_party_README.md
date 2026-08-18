# External model dependencies

GLOWS keeps only the stable adapters in `objectformer/encoders/dinov3.py` and
`objectformer/proposals/sam2.py`. The upstream SAM2 and DINOv3 source trees
and their checkpoints are intentionally not copied into this repository.

Install or clone them separately, then point to them from the YAML config:

```yaml
encoder:
  repo: /path/to/dinov3
  weight_path: /path/to/dinov3_vitl16_checkpoint.pth
proposal:
  repo: /path/to/sam2
  checkpoint: /path/to/sam2.1_hiera_base_plus.pt
```

This keeps GLOWS small and avoids redistributing third-party weights.
