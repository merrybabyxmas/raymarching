from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.evaluator import Phase65Evaluator
from phase65_min3d.scene_module import SceneModule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset = Phase65Dataset(
        root=cfg["data"]["root"],
        image_size=cfg["data"].get("image_size", 256),
        mask_size=cfg["data"].get("mask_size", 64),
        num_frames=cfg["data"].get("num_frames", 16),
    )
    loader = DataLoader(dataset, batch_size=cfg["train"].get("batch_size", 2), shuffle=False, collate_fn=phase65_collate_fn)

    model = SceneModule(
        slot_dim=cfg["model"].get("slot_dim", 256),
        feat_dim=cfg["model"].get("feat_dim", 64),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
        Hs=cfg["model"].get("Hs", 64),
        Ws=cfg["model"].get("Ws", 64),
        Hf=cfg["model"].get("Hf", 32),
        Wf=cfg["model"].get("Wf", 32),
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("scene_module", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    evaluator = Phase65Evaluator()
    all_metrics = []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            visible = batch["visible_masks"].to(device)
            amodal = batch["amodal_masks"].to(device)
            prev_state = None
            B, T = frames.shape[:2]
            for t in range(T):
                scene_state = model(
                    entity_names=batch["entity_names"][0],
                    text_prompt=batch["text_prompts"][0],
                    prev_state=prev_state,
                    prev_frame=None if t == 0 else frames[:, t - 1],
                    t_index=t,
                )
                metrics = evaluator.evaluate_scene(scene_state, visible[:, t], amodal[:, t], prev_state=prev_state)
                all_metrics.append(metrics)
                prev_state = scene_state.detach()
    mean_metrics = {k: float(sum(m[k] for m in all_metrics) / max(len(all_metrics), 1)) for k in all_metrics[0].keys()}
    print(json.dumps(mean_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
