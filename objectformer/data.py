import os

import numpy as np
from PIL import Image


def split_requires_labels(split):
    return str(split).lower() not in {"test", "infer"}


class SegDataset:
    def __init__(self, cfg, split):
        ds = cfg["dataset"]
        self.root = ds["root"]
        self.split = split
        self.subsets = list(ds.get("subsets", [None]))
        self.dataset_image_dir = ds.get("image_dir", "image")
        self.dataset_label_dir = ds.get("label_dir", "mask")
        self.image_dir = os.path.join(self.root, split, self.dataset_image_dir)
        self.label_dir = os.path.join(self.root, split, self.dataset_label_dir)
        self.label_cfg = cfg.get("label", {"format": "gray"})
        self.task_cfg = cfg["task"]
        self.crop_size = int(ds.get("crop_size", 0) or 0)
        self.crop_stride = int(ds.get("crop_stride", self.crop_size) or self.crop_size)
        self.exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        self.items = self._list_items()

    def _list_items(self):
        items = []
        for subset in self.subsets:
            image_dir = self.image_dir if subset is None else os.path.join(self.root, self.split, subset, self.dataset_image_dir)
            label_dir = self.label_dir if subset is None else os.path.join(self.root, self.split, subset, self.dataset_label_dir)
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(image_dir)
            if not os.path.isdir(label_dir):
                label_dir = None
            names = sorted(n for n in os.listdir(image_dir) if os.path.splitext(n)[1].lower() in self.exts)
            for name in names:
                stem = os.path.splitext(name)[0]
                label_path = None
                if label_dir is not None:
                    for ext in self.exts:
                        candidate = os.path.join(label_dir, stem + ext)
                        if os.path.exists(candidate):
                            label_path = candidate
                            break
                if label_path is None and split_requires_labels(self.split):
                    continue
                image_path = os.path.join(image_dir, name)
                prefix = f"{subset}_" if subset else ""
                if self.crop_size > 0:
                    with Image.open(image_path) as im:
                        width, height = im.size
                    stride = max(1, self.crop_stride)
                    xs = list(range(0, max(1, width - self.crop_size + 1), stride))
                    ys = list(range(0, max(1, height - self.crop_size + 1), stride))
                    if xs[-1] != max(0, width - self.crop_size): xs.append(max(0, width - self.crop_size))
                    if ys[-1] != max(0, height - self.crop_size): ys.append(max(0, height - self.crop_size))
                    for y in ys:
                        for x in xs:
                            items.append({"stem": f"{prefix}{stem}_{x}_{y}", "image": image_path, "label": label_path,
                                          "crop_box": (x, y, min(width, x + self.crop_size), min(height, y + self.crop_size))})
                else:
                    items.append({"stem": f"{prefix}{stem}", "image": image_path, "label": label_path})
        return items

    def __len__(self):
        return len(self.items)

    def read_image(self, path):
        return Image.open(path).convert("RGB")

    def read_item_image(self, item):
        image = self.read_image(item["image"])
        return image.crop(item["crop_box"]) if item.get("crop_box") else image

    def read_label(self, path):
        if path is None:
            return None
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        task_mode = str(self.task_cfg.get("mode", "multiclass")).lower()
        if task_mode == "binary":
            bg = int(self.task_cfg.get("background_id", 0))
            targets = sorted(int(k) for k in self.task_cfg.get("target_classes", {}).keys())
            if len(targets) != 1:
                raise ValueError("Binary mode requires exactly one target class.")
            out = np.full(arr.shape, bg, dtype=np.uint16)
            out[arr.astype(np.uint16) != bg] = int(targets[0])
            return out
        # Dataset-specific remapping for experiments such as UAVid where
        # moving/static cars are merged and human is excluded from scoring.
        # Unmapped raw labels become ignore (255), so the original dataset and
        # the main method configuration remain unchanged.
        remap = self.task_cfg.get("label_remap")
        if remap:
            mapped = np.full(arr.shape, 255, dtype=np.uint16)
            for raw_id, target_id in remap.items():
                mapped[arr.astype(np.uint16) == int(raw_id)] = int(target_id)
            arr = mapped
        # LoveDA uses 0 for ignore/no-data and semantic IDs 1..7.  Convert
        # ignore pixels to the training ignore index while retaining the
        # official class IDs used for prototypes and reporting.
        ignore_id = self.task_cfg.get("ignore_id")
        if ignore_id is not None:
            arr = arr.astype(np.uint16)
            arr[arr == int(ignore_id)] = 255
            return arr
        return arr.astype(np.uint16)

    def read_item_label(self, item):
        arr = self.read_label(item["label"])
        if arr is None:
            return None
        if item.get("crop_box"):
            x0, y0, x1, y1 = item["crop_box"]
            arr = arr[y0:y1, x0:x1]
        return arr

    def __getitem__(self, idx):
        item = self.items[idx]
        return self.read_item_image(item), self.read_item_label(item), item


def class_ids_from_cfg(cfg):
    task = cfg["task"]
    targets = sorted(int(k) for k in task["target_classes"].keys())
    if str(task.get("mode", "multiclass")).lower() == "binary":
        bg = int(task.get("background_id", 0))
        return [bg] + [cid for cid in targets if cid != bg]
    return targets


def class_names_from_cfg(cfg):
    return {int(k): str(v) for k, v in cfg["task"]["target_classes"].items()}
