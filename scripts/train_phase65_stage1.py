from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from phase65_min3d.data import Phase65Dataset, phase65_collate_fn
from phase65_min3d.scene_module import SceneModule
from phase65_min3d.trainer_stage1 import Stage1Batch, Stage1Trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset = Phase65Dataset(
        root=cfg["data"]["root"],
        image_size=cfg["data"].get("image_size", 256),
        mask_size=cfg["data"].get("mask_size", 64),
        num_frames=cfg["data"].get("num_frames", 16),
    )
    loader = DataLoader(dataset, batch_size=cfg["train"].get("batch_size", 2), shuffle=True, collate_fn=phase65_collate_fn)

    model = SceneModule(
        slot_dim=cfg["model"].get("slot_dim", 256),
        feat_dim=cfg["model"].get("feat_dim", 64),
        hidden_dim=cfg["model"].get("hidden_dim", 128),
        Hs=cfg["model"].get("Hs", 64),
        Ws=cfg["model"].get("Ws", 64),
        Hf=cfg["model"].get("Hf", 32),
        Wf=cfg["model"].get("Wf", 32),
    )
    trainer = Stage1Trainer(
        scene_module=model,
        device=device,
        lr=cfg["train"].get("lr_scene_stage1", 3e-4),
        lambda_vis=cfg["loss"].get("lambda_vis", 1.0),
        lambda_amo=cfg["loss"].get("lambda_amo", 1.0),
        lambda_temp=cfg["loss"].get("lambda_temp", 0.25),
        lambda_depth=cfg["loss"].get("lambda_depth", 0.1),
    )

    out_dir = Path(cfg.get("output_dir", "outputs/phase65/stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    history = []
    for epoch in range(cfg["train"].get("stage1_epochs", 50)):
        prev_state = None
        epoch_metrics = []
        for batch in loader:
            frames = batch["frames"].to(device)  # (B, T, 3, H, W)
            visible = batch["visible_masks"].to(device)
            amodal = batch["amodal_masks"].to(device)
            B, T = frames.shape[:2]
            for t in range(T):
                step_batch = Stage1Batch(
                    entity_names=batch["entity_names"][0],
                    text_prompt=batch["text_prompts"][0],
                    prev_frame=None if t == 0 else frames[:, t - 1],
                    gt_visible=visible[:, t],
                    gt_amodal=amodal[:, t],
                    gt_front_idx=None,
                    t_index=t,
                )
                metrics, prev_state = trainer.step(step_batch, prev_state=prev_state)
                epoch_metrics.append(metrics)
        mean_metrics = {k: float(sum(m[k] for m in epoch_metrics) / max(len(epoch_metrics), 1)) for k in epoch_metrics[0].keys()}
        mean_metrics["epoch"] = epoch + 1
        history.append(mean_metrics)
        score = mean_metrics.get("visible_iou_min", 0.0) + 0.5 * mean_metrics.get("amodal_iou_min", 0.0)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": mean_metrics}, out_dir / "best_scene_module.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[epoch {epoch+1}] {mean_metrics}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
