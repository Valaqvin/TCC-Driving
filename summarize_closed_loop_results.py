#!/usr/bin/env python3
"""Summarize CARLA leaderboard JSON files into CSV and Markdown tables."""

import argparse
import csv
import json
from pathlib import Path


METRIC_ALIASES = {
    "Avg. driving score": "driving_score",
    "Avg. route completion": "route_completion",
    "Avg. infraction penalty": "infraction_penalty",
    "Collisions with pedestrians": "collisions_pedestrian",
    "Collisions with vehicles": "collisions_vehicle",
    "Collisions with layout": "collisions_layout",
    "Red lights infractions": "red_light",
    "Stop sign infractions": "stop_infraction",
    "Off-road infractions": "outside_route_lanes",
    "Route deviations": "route_dev",
    "Route timeouts": "route_timeout",
    "Agent blocked": "vehicle_blocked",
}

SUMMARY_FIELDS = [
    "variant",
    "source",
    "entry_status",
    "eligible",
    "progress",
    "records",
    "driving_score",
    "route_completion",
    "infraction_penalty",
    "collisions_vehicle",
    "collisions_layout",
    "red_light",
    "route_timeout",
    "vehicle_blocked",
    "bargain_steps",
    "bargain_mean_delta",
    "bargain_max_delta",
]


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def variant_from_path(path):
    parts = [p.lower() for p in path.parts]
    for name in ("bargaining", "baseline"):
        if any(part.startswith(name) for part in parts):
            return name
    return path.parent.name


def values_to_metrics(data):
    metrics = {}
    labels = data.get("labels") or []
    values = data.get("values") or []
    for label, value in zip(labels, values):
        key = METRIC_ALIASES.get(label, label.lower().replace(" ", "_"))
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = value

    global_record = (data.get("_checkpoint") or {}).get("global_record") or {}
    scores = global_record.get("scores") or {}
    infractions = global_record.get("infractions") or {}
    score_map = {
        "score_composed": "driving_score",
        "score_route": "route_completion",
        "score_penalty": "infraction_penalty",
    }
    for src, dst in score_map.items():
        if dst not in metrics and src in scores:
            metrics[dst] = scores[src]
    for key, value in infractions.items():
        metrics.setdefault(key, value)
    return metrics


def summarize_bargain_log(result_dir):
    log_path = result_dir / "bargain_log.npy"
    if not log_path.exists():
        return {}
    try:
        import numpy as np
        entries = np.load(log_path, allow_pickle=True).tolist()
    except Exception:
        return {}
    if not entries:
        return {"bargain_steps": 0}
    deltas = [float(e.get("max_waypoint_delta", 0.0)) for e in entries]
    return {
        "bargain_steps": len(entries),
        "bargain_mean_delta": sum(deltas) / len(deltas),
        "bargain_max_delta": max(deltas),
    }


def summarize_result(path):
    data = load_json(path)
    checkpoint = data.get("_checkpoint") or {}
    metrics = values_to_metrics(data)
    result_dir = path.parent
    row = {
        "variant": variant_from_path(path),
        "source": str(path),
        "entry_status": data.get("entry_status", ""),
        "eligible": data.get("eligible", ""),
        "progress": "/".join(str(x) for x in checkpoint.get("progress", [])),
        "records": len(checkpoint.get("records") or []),
    }
    row.update(metrics)
    row.update(summarize_bargain_log(result_dir))
    return row


def find_result_files(root):
    files = []
    for path in root.rglob("results.json"):
        files.append(path)
    return sorted(files)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value):
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows, path):
    columns = [
        "variant",
        "entry_status",
        "progress",
        "records",
        "driving_score",
        "route_completion",
        "infraction_penalty",
        "collisions_vehicle",
        "vehicle_blocked",
        "bargain_steps",
        "bargain_mean_delta",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize closed-loop eval results")
    parser.add_argument("root", nargs="?", default="Result/closed_loop_suite")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    root = Path(args.root)
    result_files = find_result_files(root)
    if not result_files:
        raise FileNotFoundError(f"No results.json files found under {root}")

    rows = [summarize_result(path) for path in result_files]
    out_csv = Path(args.out_csv) if args.out_csv else root / "closed_loop_summary.csv"
    out_md = Path(args.out_md) if args.out_md else root / "closed_loop_summary.md"
    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    for row in rows:
        print(
            f"{row['variant']}: status={row.get('entry_status') or '-'} "
            f"progress={row.get('progress') or '-'} records={row.get('records', 0)} "
            f"score={fmt(row.get('driving_score'))} "
            f"bargain_steps={fmt(row.get('bargain_steps'))}")


if __name__ == "__main__":
    main()
