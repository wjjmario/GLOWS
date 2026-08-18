# GLOWS

GLOWS is the compact reference implementation of our full weakly supervised
instance-aware semantic segmentation method.

The public version keeps only the reproducible training and image-only test
path:

1. one point per annotated class;
2. SAM2 proposals matched to the points;
3. DINOv3 intermediate features and local/global class prototypes;
4. persistent pseudo-instance updates with a teacher/student EMA;
5. an instance query decoder trained from the evolving pseudo instances;
6. image-only semantic prediction at test time.

All ablation scripts, LPOSS experiments, visualization utilities, stitching
code, logs, checkpoints and full datasets are intentionally excluded.

## Dependencies

GLOWS contains the adapters used by the method. Install the upstream model
repositories separately:

- SAM2: `/path/to/sam2` and a SAM2 checkpoint;
- DINOv3: `/path/to/dinov3` and a DINOv3 ViT-L/16 checkpoint.

Set those paths in the dataset config. Model weights are not included.

```bash
conda env create -f environment.yml
conda activate glows
```

## Dataset configuration

The example configs are:

```text
configs/vaihingen.yaml
configs/potsdam.yaml
configs/uavid.yaml
```

Each dataset should have `train/image`, `train/mask`, and an evaluation split
with the same layout. Update `dataset.root`, `encoder.repo`,
`encoder.weight_path`, `proposal.repo`, and `proposal.checkpoint` before use.

## Train

```bash
python train.py \
  --config configs/vaihingen.yaml \
  --device cuda:0 \
  --output-dir outputs/vaihingen
```

Training writes checkpoints and `train_log.csv` under the output directory.

## Test

```bash
python test.py \
  --config configs/vaihingen.yaml \
  --checkpoint outputs/vaihingen/checkpoints/best.pt \
  --split test \
  --device cuda:0
```

The test command writes `image_only_metrics.csv` and prediction/ground-truth
panels when visualization is enabled in the config. It never uses point
annotations or SAM proposals during evaluation.

## Included samples

Two image/mask pairs are included under `samples/` for each supported dataset.
They are only for checking the data loader and command wiring; they are not a
benchmark split and are not intended for training a meaningful model.

To run the sample data loader/test path:

```bash
export GLOWS_ROOT=$PWD
python test.py --config configs/sample_vaihingen.yaml \
  --checkpoint /path/to/checkpoint.pt --split test --device cuda:0
```

The analogous files are `configs/sample_potsdam.yaml` and
`configs/sample_uavid.yaml`.
