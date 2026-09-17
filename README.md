# GLOWS

稀疏点监督遥感语义分割的 Full 实现。训练只优化查询解码器，冻结 DINOv3 和 SAM2。

训练流程：点匹配 SAM2 候选 → 区域特征构建类别原型 → 生成实例及语义伪标签 →
查询解码器联合监督 → EMA 教师、持久实例库和全局原型按训练进度更新。
测试只使用图像、DINOv3 和训练后的解码器，不调用 SAM2，不需要点或标签。

## 文件结构

```text
train.py                 训练、每轮评价、续训
test.py                  仅图像预测，可选计算指标
configs/full.yaml        Full 共享配置
configs/<dataset>.yaml   数据集差异
objectformer/            数据、点匹配、伪标签、模型和损失
third_party/dinov3/      DINOv3 源码及其许可证
third_party/sam2/        SAM2 源码及其许可证
weights/                 本地预训练权重，不纳入 Git
samples/                 原仓库的少量样本，仅用于检查运行
```

只保留主训练/测试路径，不包含 LPOSS、全监督对比、消融脚本、论文绘图和大图拼接。
核心的置信度筛选、冲突处理、数值检查仍保留，它们影响方法行为或正常运行。

## 安装

建议 Python 3.10+、支持 BF16 的 NVIDIA GPU。先安装与 CUDA 对应的 PyTorch，随后：

```bash
cd /root/autodl-tmp/wsdino_demo/GLOWS
python -m pip install -r requirements.txt
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e third_party/sam2
```

DINOv3 从仓库内源码加载，无需联网下载。SAM2 的 CUDA 扩展为可选项；
上面的安装禁用扩展编译，不改变自动掩码生成器的 min_mask_region_area 配置。
服务器副本含以下两个真实文件，不是指向外部目录的软链接：

```text
weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
weights/sam2.1_hiera_base_plus.pt
```

公开 Git 仓库时权重默认被忽略。第三方代码及权重遵循各自许可证；
请在公开前确认再分发条件，不要把整个仓库默认声明为同一个许可证。
见 `THIRD_PARTY.md`。

## 数据

预先准备切片和单通道类别 ID 标签（不是 RGB 彩色标签）：

```text
data/<dataset>/train/image/name.png
data/<dataset>/train/mask/name.png
data/<dataset>/val/image/name.png
data/<dataset>/val/mask/name.png
```

Vaihingen/Potsdam 使用 `test` 评价目录。标签与影像同名，可使用 PNG/TIF/JPG 影像。
所有相对路径均以仓库根目录为基准；修改对应 YAML 的 dataset.root 即可接入其他位置的数据。
样本必须先按原始区域划分训练/评价集，再切片，避免相邻切片泄漏。

| 配置 | 原始 mask 编码 | 输出类别 |
| --- | --- | --- |
| vaihingen / potsdam | 0–4 地物，5 背景 | 0–4，背景不参与点与指标 |
| uavid | 0建筑 1道路 2树 3低植被 4运动车 5静止车 6人 7杂类 | 4、5合为4车辆；6忽略；7变为5杂类 |
| floodnet | 0背景，1–9有效类别 | 1–9，背景忽略 |

UAVid 必须输入原始八类 ID；不要对已经映射的六类标签再次使用该 remap。
1/3-point 指每个训练切片中每个有效类别的点数，不是每个实例的点数。
训练从 mask 模拟点，点以外的稠密真值不作为训练损失目标。

## 训练

```bash
python train.py --config configs/vaihingen.yaml --points 1 --output-dir outputs/vaihingen_1point
python train.py --config configs/uavid.yaml --points 3 --output-dir outputs/uavid_3point
python train.py --config configs/floodnet.yaml --points 1 --batch-size 16 --output-dir outputs/floodnet_1point
```

首次训练自动构建 SAM2 和旋转融合 DINO 缓存。缓存位于 `cache/<dataset>/`。
**改变 SAM2 参数、DINO 权重/输入尺寸或数据内容后，需更换 cache_dir 或移走旧缓存。**
多个不同实验必须使用不同 output-dir，避免实例库和日志相互覆盖。

```bash
python train.py --config configs/vaihingen.yaml --output-dir outputs/vaihingen_1point \
  --resume outputs/vaihingen_1point/checkpoints/last.pt --epochs 40
```

续训保留原输出目录中的 pseudo_bank，并保持点数及配置一致；epochs 表示目标总轮数。
默认只保留 `best.pt` 与 `last.pt`。保存的 checkpoint 包含模型、EMA、优化器与全局原型，
实例库单独存储。不要只复制 last.pt 就声称完整恢复闭环状态。

输出包括 `resolved_config.json`、`train_log.csv`、`closed_loop_log.csv`、
`image_only_metrics.csv`。指标含各类 IoU/F1、mIoU/mF1/OA，数值是 0–1。
每轮在配置指定的评价集上评价并按 mIoU 选 best；Vaihingen/Potsdam 延续历史 test 选轮协议。

## 测试

```bash
python test.py --config configs/vaihingen.yaml \
  --checkpoint outputs/vaihingen_1point/checkpoints/best.pt --output-dir outputs/prediction
```

保存每张图的单通道类别 ID PNG，不保存 panel、不拼接、不把 GT 覆盖到预测上。
有标签时同时写指标，无标签时仅保存预测。checkpoint 使用 EMA 参数（若存在）。
只能加载信任来源的 checkpoint。

仅推理任意图像目录：

```bash
python test.py --config configs/uavid.yaml --checkpoint /path/to/best.pt \
  --images /path/to/images --output-dir outputs/prediction
```

## 最小运行检查

```bash
python train.py --config configs/sample_uavid.yaml --max-images 1 --eval-max-images 1 \
  --epochs 3 --batch-size 1 --output-dir outputs/smoke
python test.py --config configs/sample_uavid.yaml --max-images 1 \
  --checkpoint outputs/smoke/checkpoints/best.pt --output-dir outputs/smoke/prediction
```

三轮用于覆盖初始伪标签、全局原型启用和教师候选更新；不是有效精度实验。
样本与正式数据切勿混用。正式默认 BS 为 Vaihingen/Potsdam 8、UAVid/FloodNet 16，
显存不足时直接降低 --batch-size，不要认为 24GB GPU 必然能承受所有配置的默认 BS。

## 保留的实验配置差异

四个数据集均启用全局原型、持久实例库、一致性筛选和同分辨率四层解码，
DINOv3 block 索引为 `[4,11,17,23]`（从 0 计数）。
沿用实际配置：Vaihingen/Potsdam 使用全局原型替换局部原型，UAVid/FloodNet 使用各 0.5 融合。
UAVid 编码尺寸512，其余1024；FloodNet 使用当前 SAM2 0.80/0.70/0.85、region area100、无额外裁剪层配置。
这些差异明确保存在数据集 YAML 中，整理仓库不会倒改历史实验。

## 本次验证 2026-09-17

- Python 编译、train/test 命令行与四个数据集配置加载通过。
- Vaihingen、Potsdam、UAVid 的现有样本读取和 1-point 数量检查通过。
- FloodNet 真实训练目录读取通过（5771 张），背景映射为 ignore。
- UAVid 样本 1-point / BS1 三轮训练通过。
- UAVid 样本 3-point / BS2 三轮训练和恢复至第4轮通过。
- 加载生成的最佳权重，有标签评价和无标签预测通过，实际输出 PNG。
- 验证是运行检查，不是精度复现；未重新运行完整数据集实验，也未验证所有默认大 batch 的显存需求。

服务器检查日志：`/root/autodl-tmp/glows_verify_20260917.log`。
整理前备份：`/root/autodl-tmp/glows_before_cleanup_20260917.tar.gz`。
上述绝对路径只用于本次交接记录，不参与训练/测试运行。
