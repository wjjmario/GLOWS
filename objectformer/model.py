import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = float(temperature)

    def forward(self, x):
        b, _, h, w = x.shape
        y, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=torch.float32),
            torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij"
        )
        y = y / max(h - 1, 1) * (2 * math.pi)
        xx = xx / max(w - 1, 1) * (2 * math.pi)
        dim = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim = self.temperature ** (2 * torch.div(dim, 2, rounding_mode="floor") / self.num_pos_feats)
        px, py = xx[..., None] / dim, y[..., None] / dim
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=-1).flatten(-2)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((py, px), dim=-1).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)


class PixelDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, mask_dim=256, memory_mode="pyramid"):
        super().__init__()
        self.memory_mode = str(memory_mode).lower()
        if self.memory_mode not in {"pyramid", "same_resolution"}:
            raise ValueError(f"Unsupported memory_mode: {memory_mode}")
        self.projections = nn.ModuleList([nn.Conv2d(in_dim, hidden_dim, 1) for _ in range(4)])
        self.refine = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False), nn.GroupNorm(32, hidden_dim), nn.ReLU())
            for _ in range(4)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim), nn.ReLU(),
        )
        self.mask_features = nn.Conv2d(hidden_dim, mask_dim, 1)

    def forward(self, tokens, patch_h, patch_w):
        features = []
        for projection, refine, token in zip(self.projections, self.refine, tokens):
            b, n, c = token.shape
            if n != patch_h * patch_w:
                raise ValueError(f"Token count {n} != patch grid {patch_h}x{patch_w}")
            fmap = token.transpose(1, 2).reshape(b, c, patch_h, patch_w)
            features.append(refine(projection(fmap)))
        fused = self.fuse(torch.cat(features, dim=1))
        if self.memory_mode == "same_resolution":
            # DINOv3 intermediate layers have equal spatial resolution but
            # different semantic depth. Keep them as separate memories rather
            # than manufacturing a Swin-style spatial pyramid.
            memories = features
        else:
            # Historical Mask2Former-style pyramid; kept as the default for
            # exact backward compatibility.
            memories = [fused, F.avg_pool2d(fused, 2), F.avg_pool2d(fused, 4)]
        return self.mask_features(fused), memories


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, query, memory, query_pos, memory_pos, attention_mask=None):
        q = self.norm1(query + query_pos)
        cross, _ = self.cross(q, memory + memory_pos, memory, attn_mask=attention_mask, need_weights=False)
        query = query + cross
        q = self.norm2(query + query_pos)
        self_out, _ = self.self_attn(q, q, query, need_weights=False)
        query = query + self_out
        query = query + self.ffn(self.norm3(query))
        return query


class ObjectFormer(nn.Module):
    """Image-only Mask2Former-style decoder over frozen DINOv3 features."""

    def __init__(
        self,
        num_classes,
        in_dim,
        hidden_dim=256,
        num_queries=100,
        num_layers=6,
        num_heads=8,
        use_prototype_head=True,
        memory_mode="pyramid",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_heads = int(num_heads)
        self.memory_mode = str(memory_mode).lower()
        self.pixel_decoder = PixelDecoder(in_dim, hidden_dim, hidden_dim, self.memory_mode)
        self.position = PositionEmbeddingSine(hidden_dim // 2)
        self.query_content = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)
        self.level_embedding = nn.Embedding(4 if self.memory_mode == "same_resolution" else 3, hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.prototype_head = nn.Linear(hidden_dim, in_dim) if bool(use_prototype_head) else None

    def _prediction(self, query, mask_features):
        normalized = self.decoder_norm(query)
        logits = self.class_head(normalized)
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_head(normalized), mask_features)
        prediction = {
            "pred_logits": logits,
            "pred_masks": masks,
            "query_embeddings": normalized,
        }
        if self.prototype_head is not None:
            prediction["pred_prototypes"] = F.normalize(self.prototype_head(normalized), dim=-1)
        return prediction

    def _attention_mask(self, masks, memory_hw):
        resized = F.interpolate(masks, size=memory_hw, mode="bilinear", align_corners=False)
        blocked = resized.sigmoid().flatten(2) < 0.5
        # MultiheadAttention cannot accept a row with every key masked.
        all_blocked = blocked.all(-1, keepdim=True)
        blocked = blocked & ~all_blocked
        return blocked[:, None].expand(-1, self.num_heads, -1, -1).flatten(0, 1).detach()

    def forward(self, tokens, patch_h, patch_w):
        mask_features, memories = self.pixel_decoder(tokens, patch_h, patch_w)
        b = mask_features.shape[0]
        query = self.query_content.weight.unsqueeze(0).expand(b, -1, -1)
        query_pos = self.query_position.weight.unsqueeze(0).expand(b, -1, -1)
        outputs = [self._prediction(query, mask_features)]
        for i, layer in enumerate(self.layers):
            level = i % len(memories)
            fmap = memories[level]
            memory = fmap.flatten(2).transpose(1, 2)
            pos = (self.position(fmap) + self.level_embedding.weight[level][None, :, None, None]).flatten(2).transpose(1, 2)
            attention_mask = self._attention_mask(outputs[-1]["pred_masks"], fmap.shape[-2:])
            query = layer(query, memory, query_pos, pos, attention_mask)
            outputs.append(self._prediction(query, mask_features))
        final = outputs[-1]
        final["aux_outputs"] = [
            {key: x[key] for key in ("pred_logits", "pred_masks", "pred_prototypes") if key in x}
            for x in outputs[:-1]
        ]
        return final


def semantic_logits(outputs):
    class_prob = outputs["pred_logits"].softmax(-1)[..., :-1]
    mask_prob = outputs["pred_masks"].sigmoid()
    probability = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob).clamp_min(1e-8)
    return probability.log()


def build_model(cfg, num_classes, encoder_dim):
    mcfg = cfg.get("model", {})
    return ObjectFormer(
        num_classes,
        encoder_dim,
        hidden_dim=int(mcfg.get("hidden_dim", 256)),
        num_queries=int(mcfg.get("num_queries", 100)),
        num_layers=int(mcfg.get("decoder_layers", 6)),
        num_heads=int(mcfg.get("num_heads", 8)),
        use_prototype_head=bool(mcfg.get("use_prototype_head", True)),
        memory_mode=str(mcfg.get("memory_mode", "pyramid")),
    )
