"""Load the public state-dict format, independent of training objects/paths."""
import math
import torch
from .pairwise import PairwiseScorerMLPSymmetry


def load_model(path, config, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != "ppi-word-model-1":
        raise ValueError("A public ppi-word-model-1 checkpoint is required")
    for key in ("temperature", "dropout"):
        value = float(config[key])
        if not math.isfinite(value) or value != float(saved["model_config"][key]):
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    if config["temperature"] <= 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid temperature/dropout")
    model = PairwiseScorerMLPSymmetry(config["temperature"], config["dropout"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()
