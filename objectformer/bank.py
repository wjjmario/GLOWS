import json
import os
import tempfile

from .objects import PseudoInstance


class PseudoInstanceBank:
    """Atomic, versioned per-image pseudo-label persistence."""

    SCHEMA_VERSION = 1

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def path(self, split, stem):
        directory = os.path.join(self.root, split)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{stem}.json")

    def load(self, split, stem):
        path = self.path(split, stem)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported pseudo-bank schema: {path}")
        payload["instances"] = [PseudoInstance(**item) for item in payload.get("instances", [])]
        return payload

    def save(self, split, stem, instances, epoch, point_signature):
        path = self.path(split, stem)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "epoch": int(epoch),
            "point_signature": str(point_signature),
            "instances": [x.to_dict() for x in instances],
        }
        fd, tmp = tempfile.mkstemp(prefix=".pseudo-", suffix=".json", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path
