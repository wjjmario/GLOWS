# 第三方代码和权重

- `third_party/dinov3`：从本机正在使用的 DINOv3 源码复制，保留 `LICENSE.md`、`MODEL_CARD.md`。
  官方来源：https://github.com/facebookresearch/dinov3 。
- `third_party/sam2`：从本机正在使用的 SAM2 源码复制，保留 `LICENSE`、`LICENSE_cctorch`。
  官方来源：https://github.com/facebookresearch/sam2 。
- 为避免无关体积，不包含第三方 notebooks、demo、数据集、Git 历史和已有 build 产物。
  模型 Python 包和加载入口保留。运行时请使用仓库内的源码。
- 两个预训练权重复制到 `weights/`，不提交 Git。`weights/SHA256SUMS` 记录校验值。
  公布权重前应按上游许可核对再分发条件；本仓库不改变第三方许可。
- GLOWS 自有代码的开源许可证由作者在正式公开前决定。
