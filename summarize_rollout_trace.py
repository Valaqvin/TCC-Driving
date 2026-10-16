#!/usr/bin/env python3
"""Summarize per-frame ego_vehicle JSON traces from CARLA runs."""

import argparse
import json
import math
from pathlib import Path


def load_frame(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def distance(a, b):
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def summarize_ego(ego_dir):
    frames = sorted(ego_dir.glob("*.json"))
    if not frames:
        return {"ego": ego_dir.name, "frames": 0}
    data = [load_frame(path) for path in frames]
    poses = [
        (d.get("lidar_pose_x"), d.get("lidar_pose_y"))
        for d in data
        if d.get("lidar_pose_x") is not None and d.get("lidar_pose_y") is not None
    ]
    speeds = [float(d.get("speed", 0.0)) for d in data]
    throttles = [float(d.get("throttle", 0.0)) for d in data]
    brakes = [float(d.get("brake", 0.0)) for d in data]
    steer = [float(d.get("steer", 0.0)) for d in data]
    return {
        "ego": ego_dir.name,
        "frames": len(frames),
        "first_frame": frames[0].stem,
        "last_frame": frames[-1].stem,
        "displacement": distance(poses[0], poses[-1]) if len(poses) >= 2 else 0.0,
        "last_x": poses[-1][0] if poses else None,
        "last_y": poses[-1][1] if poses else None,
        "mean_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "max_speed": max(speeds) if speeds else 0.0,
        "nonzero_throttle": sum(1 for v in throttles if abs(v) > 1e-4),
        "nonzero_brake": sum(1 for v in brakes if abs(v) > 1e-4),
        "nonzero_steer": sum(1 for v in steer if abs(v) > 1e-4),
        "last_speed": speeds[-1] if speeds else 0.0,
        "last_throttle": throttles[-1] if throttles else 0.0,
        "last_brake": brakes[-1] if brakes else 0.0,
    }


def find_route_dirs(root):
    route_dirs = []
    for path in root.rglob("ego_vehicle_0"):
        route_dirs.append(path.parent)
    return sorted(set(route_dirs))


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize CARLA per-frame traces")
    parser.add_argument("root", help="Run root or route directory")
    args = parser.parse_args()
    root = Path(args.root)

    route_dirs = [root] if (root / "ego_vehicle_0").exists() else find_route_dirs(root)
    if not route_dirs:
        raise FileNotFoundError(f"No ego_vehicle_* traces found under {root}")

    for route in route_dirs:
        print(f"route: {route}")
        ego_dirs = sorted(route.glob("ego_vehicle_*"))
        rows = [summarize_ego(ego_dir) for ego_dir in ego_dirs]
        columns = [
            "ego", "frames", "first_frame", "last_frame", "displacement",
            "mean_speed", "max_speed", "nonzero_throttle", "nonzero_brake",
            "last_speed", "last_throttle", "last_brake",
        ]
        print("| " + " | ".join(columns) + " |")
        print("|" + "|".join(["---"] * len(columns)) + "|")
        for row in rows:
            print("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")


if __name__ == "__main__":
    main()
