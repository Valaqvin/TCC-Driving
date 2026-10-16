# -*- coding: utf-8 -*-
"""
distill_align.py - Distill LLM knowledge into UtilityEstimator.

Uses contrastive learning to align the UtilityEstimator with LLM judgments.
Good offers (from LLM) should get high utility scores, random offers should get low scores.

Usage:
    python distill_align.py
    python distill_align.py --data Result/distill_data/labeled_scenes.json
    python distill_align.py --epochs=50

Output:
    Result/distilled_model.pth
    Result/distill_eval.json
"""

import os, sys, json, math, datetime, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from opencood.models.comm_modules.rubinstein_bargaining import UtilityEstimator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KinematicEncoder(nn.Module):
    """Encodes kinematic state (10-dim) into 64-dim embedding for UtilityEstimator."""
    def __init__(self, input_dim=10, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.LayerNorm(output_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


def build_training_data(scenes):
    states, good_offers = [], []
    for s in scenes:
        off = s.get("llm_offers")
        if not off or not isinstance(off, dict):
            continue
        kin = [
            s["pos_a"][0], s["pos_a"][1], s["pos_b"][0], s["pos_b"][1],
            s["speed_a"], s["speed_b"],
            s["heading_a"], s["heading_b"],
            s["ttc"],
            1.0 if s["scenario"] == "head_on" else 0.0,
        ]
        for key in ["vehicle_a", "vehicle_b"]:
            if key in off and isinstance(off[key], dict):
                offer = [off[key].get(k, 0.5) for k in ["speed_factor","lateral_shift","time_headway","yield_willingness"]]
                if all(0 <= v <= 1 for v in offer):
                    states.append(kin)
                    good_offers.append(offer)
    return torch.tensor(states, dtype=torch.float32), torch.tensor(good_offers, dtype=torch.float32)


def train_distill(scenes_path, epochs=50, lr=1e-4):
    scenes = json.load(open(scenes_path))
    states, good_offers = build_training_data(scenes)
    n = states.shape[0]
    print(f"[Distill] {n} training pairs from {len(scenes)} scenes")

    encoder = KinematicEncoder().to(DEVICE)
    estimator = UtilityEstimator(state_dim=64, offer_dim=4, hidden_dim=128).to(DEVICE)
    params = list(encoder.parameters()) + list(estimator.parameters())
    optimizer = optim.Adam(params, lr=lr)

    for ep in range(epochs):
        idx = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, 64):
            batch_idx = idx[i:i+64]
            s = states[batch_idx].to(DEVICE)
            g = good_offers[batch_idx].to(DEVICE)
            b = torch.rand_like(g)

            s_enc = encoder(s)
            u_good = estimator(s_enc, g)
            u_bad = estimator(s_enc, b)

            loss = torch.clamp(u_bad - u_good + 0.3, min=0).mean()
            loss += 0.1 * torch.abs(u_good - 1.0).mean()

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        avg_loss = total_loss / n
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"  Ep {ep}: loss={avg_loss:.6f}, u_good={u_good.mean().item():.4f}, u_bad={u_bad.mean().item():.4f}")

    return encoder, estimator


def save_model(encoder, estimator, output_dir):
    torch.save({"encoder": encoder.state_dict(), "estimator": estimator.state_dict()},
               str(output_dir / "distilled_model.pth"))
    print(f"[Save] -> {output_dir / 'distilled_model.pth'}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(Path(PROJECT_ROOT) / "Result" / "distill_data" / "labeled_scenes.json"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(PROJECT_ROOT) / "Result" / f"distill_{ts}"
    output.mkdir(parents=True, exist_ok=True)

    print(f"[Distill] LLM Knowledge Distillation")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print()

    encoder, estimator = train_distill(args.data, args.epochs, args.lr)
    save_model(encoder, estimator, output)
    print()
    print(f"Done. Model -> {output}")


if __name__ == "__main__":
    main()
