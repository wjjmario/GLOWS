import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DINOv3Encoder:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        repo = cfg["repo"]
        if repo not in sys.path:
            sys.path.append(repo)
        torch_home = cfg.get("torch_home")
        if torch_home:
            os.environ.setdefault("TORCH_HOME", os.path.expanduser(str(torch_home)))
        weight_path = cfg.get("weight_path", "")
        if weight_path and not os.path.exists(weight_path):
            raise FileNotFoundError(weight_path)
        print(f"loading DINOv3 {cfg['model_name']}: {weight_path}")
        self.model = torch.hub.load(repo, cfg["model_name"], source="local", weights=weight_path)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.image_size = int(cfg.get("image_size", 1024))
        self.patch_size = int(cfg.get("patch_size", 16))
        self.model_name = str(cfg.get("model_name", "")).lower()
        self.encoder_size = str(cfg.get("encoder_size", self._infer_encoder_size()))
        self.intermediate_layers = [
            int(index) for index in cfg.get("intermediate_layers", self._default_layers(self.encoder_size))
        ]
        if not self.intermediate_layers:
            raise ValueError("encoder.intermediate_layers must be non-empty")
        if self.intermediate_layers != sorted(set(self.intermediate_layers)):
            raise ValueError(
                "encoder.intermediate_layers must contain unique, increasing block indices"
            )
        self.memory_layer_slots = [
            int(index) for index in cfg.get(
                "memory_layer_slots", range(len(self.intermediate_layers)),
            )
        ]
        if len(self.memory_layer_slots) != 4:
            raise ValueError("encoder.memory_layer_slots must define exactly four decoder memories")
        if min(self.memory_layer_slots) < 0 or max(self.memory_layer_slots) >= len(self.intermediate_layers):
            raise ValueError(
                "encoder.memory_layer_slots contains an index outside intermediate_layers"
            )

    def _infer_encoder_size(self):
        if "vits" in self.model_name or "small" in self.model_name:
            return "small"
        if "vitb" in self.model_name or "base" in self.model_name:
            return "base"
        return "large"

    @staticmethod
    def _default_layers(size):
        return {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
        }.get(size, [4, 11, 17, 23])

    @property
    def embed_dim(self):
        return int(getattr(self.model, "embed_dim"))

    @property
    def patch_hw(self):
        return self.image_size // self.patch_size, self.image_size // self.patch_size

    def preprocess(self, image_pil):
        image = image_pil.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype)[:, None, None]
        x = (x - mean) / std
        return x[None].to(self.device)

    def preprocess_batch(self, images):
        """Preprocess equally sized PIL images into one encoder batch."""
        if not images:
            raise ValueError("images must be non-empty")
        return torch.cat([self.preprocess(image) for image in images], dim=0)

    def _tokens_to_feature_map(self, feat):
        b, n, c = feat.shape
        hf, wf = self.patch_hw
        if n == hf * wf + 1:
            feat = feat[:, 1:, :]
            n -= 1
        if n != hf * wf:
            side = int(n ** 0.5)
            if side * side != n:
                raise RuntimeError(f"Cannot reshape DINO tokens: N={n}, Hf={hf}, Wf={wf}")
            hf = wf = side
        fmap = feat.transpose(1, 2).reshape(b, c, hf, wf).contiguous().float()
        return F.normalize(fmap, dim=1)

    def _strip_cls(self, tokens):
        hf, wf = self.patch_hw
        out = []
        for x in tokens:
            if isinstance(x, (tuple, list)):
                x = x[0]
            if x.shape[1] == hf * wf + 1:
                x = x[:, 1:, :]
            out.append(x.float().detach().clone())
        return out

    def extract(self, image_pil):
        return self.extract_batch([image_pil])

    def extract_batch(self, images):
        batch = self.preprocess_batch(images)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model.forward_features(batch)
            else:
                out = self.model.forward_features(batch)
        if isinstance(out, dict):
            feat = None
            for key in ["x_norm_patchtokens", "patch_tokens", "x_prenorm"]:
                if key in out:
                    feat = out[key]
                    break
            if feat is None:
                raise RuntimeError(f"No patch token field in DINOv3 output keys: {list(out.keys())}")
        else:
            feat = out
        return self._tokens_to_feature_map(feat).detach().clone()

    def extract_intermediate_tokens(self, image_pil):
        return self.extract_intermediate_tokens_batch([image_pil])

    def extract_intermediate_tokens_batch(self, images):
        batch = self.preprocess_batch(images)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    toks = self.model.get_intermediate_layers(batch, n=self.intermediate_layers)
            else:
                toks = self.model.get_intermediate_layers(batch, n=self.intermediate_layers)
        toks = self._strip_cls(list(toks))
        if len(toks) != len(self.intermediate_layers):
            raise RuntimeError(
                "DINOv3 returned "
                f"{len(toks)} intermediate layers for {len(self.intermediate_layers)} requested indices"
            )
        # The decoder always receives four memories.  Reusing a slot makes a
        # controlled last-layer-only ablation possible without changing the
        # decoder capacity, level embeddings, or number of cross-attention
        # operations.
        return [toks[index] for index in self.memory_layer_slots]
