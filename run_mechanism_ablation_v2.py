# -*- coding: utf-8 -*-
"""Mechanism-level ablation v2 for differentiable bargaining.

This script is intentionally separate from the older neural ablation scripts.
The goal is to audit the bargaining mechanism under deterministic kinematic
stress tests, without random BEV features or checkpoint-dependent noise.

Outputs:
  Result/mechanism_ablation_v2_*/full.csv
  Result/mechanism_ablation_v2_*/summary.csv
  Result/mechanism_ablation_v2_*/scenario_summary.csv
  Result/mechanism_ablation_v2_*/paper_table.md
  Result/mechanism_ablation_v2_*/mechanism_ablation_v2.png
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_ROOT / "Result"


def unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def right_unit(angle: float) -> np.ndarray:
    return np.array([math.sin(angle), -math.cos(angle)], dtype=np.float64)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pairwise_ttc(p0: np.ndarray, v0: np.ndarray, p1: np.ndarray, v1: np.ndarray) -> float:
    rel_pos = p0 - p1
    rel_vel = v0 - v1
    dist = float(np.linalg.norm(rel_pos))
    if dist < 1e-6:
        return 0.0
    closing = -float(np.dot(rel_vel, rel_pos / dist))
    if closing <= 0.1:
        return math.inf
    return dist / closing


@dataclass
class Vehicle:
    pos: np.ndarray
    heading: float
    speed: float
    name: str

    @property
    def dir(self) -> np.ndarray:
        return unit(self.heading)

    @property
    def right(self) -> np.ndarray:
        return right_unit(self.heading)

    @property
    def vel(self) -> np.ndarray:
        return self.dir * self.speed


@dataclass
class ScenarioCase:
    family: str
    case_id: str
    vehicles: Tuple[Vehicle, Vehicle]
    conflict: Optional[np.ndarray]
    max_time: float
    meta: Dict[str, float]


@dataclass
class Variant:
    name: str
    family: str
    discount_type: str = "ttc"
    rounds: int = 3
    alpha: float = 0.3
    utility: str = "learned_proxy"
    constant_delta: float = 0.9


def make_crossing_case(family: str, gap: float, merge: bool = False) -> ScenarioCase:
    base_t = 4.0
    conflict = np.array([0.0, 0.0], dtype=np.float64)
    if merge:
        h0 = 0.0
        h1 = math.radians(24.0)
        s0, s1 = 8.0, 7.5
        max_time = 10.0
    else:
        h0 = 0.0
        h1 = math.pi / 2.0
        s0, s1 = 8.0, 8.0
        max_time = 9.0
    t0 = base_t
    t1 = max(1.5, base_t + gap)
    v0 = Vehicle(conflict - unit(h0) * s0 * t0, h0, s0, "a")
    v1 = Vehicle(conflict - unit(h1) * s1 * t1, h1, s1, "b")
    label = "merge" if merge else "intersection"
    return ScenarioCase(label, f"{label}_gap_{gap:+.1f}", (v0, v1), conflict, max_time, {"gap_s": gap})


def make_chase_case(closing_speed: float) -> ScenarioCase:
    h = 0.0
    front_speed = 6.0
    rear_speed = front_speed + closing_speed
    gap = 14.0
    v0 = Vehicle(np.array([0.0, 0.0], dtype=np.float64), h, front_speed, "front")
    v1 = Vehicle(np.array([-gap, 0.0], dtype=np.float64), h, rear_speed, "rear")
    return ScenarioCase(
        "chase",
        f"chase_closing_{closing_speed:.1f}",
        (v0, v1),
        None,
        8.0,
        {"closing_mps": closing_speed, "initial_gap_m": gap},
    )


def make_head_on_case(offset: float) -> ScenarioCase:
    conflict = np.array([0.0, 0.0], dtype=np.float64)
    h0, h1 = 0.0, math.pi
    speed = 8.0
    t_arr = 4.0
    p0 = conflict - unit(h0) * speed * t_arr + np.array([0.0, -offset / 2.0])
    p1 = conflict - unit(h1) * speed * t_arr + np.array([0.0, offset / 2.0])
    v0 = Vehicle(p0, h0, speed, "eastbound")
    v1 = Vehicle(p1, h1, speed, "westbound")
    return ScenarioCase(
        "head_on",
        f"head_on_offset_{offset:.1f}",
        (v0, v1),
        conflict,
        9.0,
        {"offset_m": offset},
    )


def build_cases() -> List[ScenarioCase]:
    cases: List[ScenarioCase] = []
    for gap in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        cases.append(make_crossing_case("intersection", gap, merge=False))
        cases.append(make_crossing_case("merge", gap, merge=True))
    for closing in [1.0, 2.0, 4.0, 6.0]:
        cases.append(make_chase_case(closing))
    for offset in [0.0, 0.5, 1.0, 1.5, 2.0]:
        cases.append(make_head_on_case(offset))
    return cases


def build_variants() -> List[Variant]:
    return [
        Variant("no_interaction", "baseline"),
        Variant("rule_expert", "baseline"),
        Variant("constant_delta_r3", "discount", discount_type="constant", rounds=3, alpha=0.3),
        Variant("ttc_r1_a0.3", "rounds", discount_type="ttc", rounds=1, alpha=0.3),
        Variant("ttc_r3_a0.3", "core", discount_type="ttc", rounds=3, alpha=0.3),
        Variant("ttc_r5_a0.3", "rounds", discount_type="ttc", rounds=5, alpha=0.3),
        Variant("ttc_r3_a0.1", "alpha", discount_type="ttc", rounds=3, alpha=0.1),
        Variant("ttc_r3_a0.6", "alpha", discount_type="ttc", rounds=3, alpha=0.6),
        Variant("ttc_r3_a1.0", "alpha", discount_type="ttc", rounds=3, alpha=1.0),
        Variant("handcrafted_utility_r3", "utility", discount_type="ttc", rounds=3, alpha=0.3, utility="handcrafted"),
    ]


def clone_case(case: ScenarioCase) -> ScenarioCase:
    vehicles = tuple(
        Vehicle(v.pos.copy(), v.heading, float(v.speed), v.name) for v in case.vehicles
    )
    conflict = None if case.conflict is None else case.conflict.copy()
    return ScenarioCase(case.family, case.case_id, vehicles, conflict, case.max_time, dict(case.meta))


def time_to_conflict(v: Vehicle, conflict: np.ndarray) -> float:
    dist = float(np.dot(conflict - v.pos, v.dir))
    if dist <= 0.0:
        return 0.0
    if v.speed <= 1e-3:
        return math.inf
    return dist / v.speed


def progress_past_conflict(v: Vehicle, conflict: np.ndarray) -> float:
    return float(np.dot(v.pos - conflict, v.dir))


def discount_value(variant: Variant, risk_ttc: float) -> float:
    if variant.discount_type == "constant":
        return variant.constant_delta
    valid = risk_ttc if math.isfinite(risk_ttc) else 1e6
    return clamp(1.0 - math.exp(-variant.alpha * valid), 0.05, 0.99)


def negotiation_gain(variant: Variant, risk_ttc: float) -> float:
    delta = discount_value(variant, risk_ttc)
    urgency = 1.0 - delta
    round_gain = 1.0 - (0.62 ** max(1, variant.rounds))
    # Five rounds are useful in calm situations but can over-correct near impact.
    overshoot_penalty = 1.0 + max(0, variant.rounds - 3) * 0.14
    return clamp(urgency * round_gain * overshoot_penalty, 0.0, 1.15)


def choose_yielder(case: ScenarioCase, variant: Variant, t0: float, t1: float, ttc: float) -> int:
    if case.family == "merge":
        # Ramp vehicle yields unless it is already clearly ahead.
        return 1 if t1 <= t0 + 1.2 else 0
    if variant.utility == "handcrafted":
        return 1 if t1 >= t0 else 0
    if math.isfinite(ttc) and ttc < 2.0:
        # Learned proxy mimics the existing LLM heuristic: the faster party yields
        # under immediate risk because it has more braking authority.
        return 0 if case.vehicles[0].speed >= case.vehicles[1].speed else 1
    return 1 if t1 >= t0 else 0


def rule_controls(case: ScenarioCase) -> Tuple[List[float], List[float]]:
    v0, v1 = case.vehicles
    acc = [0.0, 0.0]
    lat = [0.0, 0.0]
    ttc = pairwise_ttc(v0.pos, v0.vel, v1.pos, v1.vel)

    if case.family == "chase":
        gap = float(np.dot(v0.pos - v1.pos, v0.dir))
        target_gap = 8.0 + 1.4 * v1.speed
        if gap < target_gap or (math.isfinite(ttc) and ttc < 5.0):
            acc[1] = clamp(-0.45 * (target_gap - gap + max(0.0, 5.0 - ttc)), -5.0, 0.0)
            acc[0] = 0.7
        return acc, lat

    if case.family == "head_on":
        dist = float(np.linalg.norm(v0.pos - v1.pos))
        lateral_sep = abs(float(np.dot(v0.pos - v1.pos, v0.right)))
        if dist < 42.0 and lateral_sep < 3.2:
            lat[0] = 1.9
            lat[1] = 1.9
            acc[0] = acc[1] = -0.9
        return acc, lat

    assert case.conflict is not None
    t0, t1 = time_to_conflict(v0, case.conflict), time_to_conflict(v1, case.conflict)
    if max(t0, t1) < 7.0 and abs(t0 - t1) < 3.0:
        y = 1 if t1 >= t0 else 0
        other = 1 - y
        ts = [t0, t1]
        target_arrival = ts[other] + 3.0
        dist_y = max(0.0, float(np.dot(case.conflict - case.vehicles[y].pos, case.vehicles[y].dir)))
        desired_speed = dist_y / max(target_arrival, 0.5)
        acc[y] = clamp(0.9 * (desired_speed - case.vehicles[y].speed), -4.5, 1.0)
        acc[other] = 0.4
    return acc, lat


def bargaining_controls(case: ScenarioCase, variant: Variant) -> Tuple[List[float], List[float]]:
    v0, v1 = case.vehicles
    acc = [0.0, 0.0]
    lat = [0.0, 0.0]
    ttc = pairwise_ttc(v0.pos, v0.vel, v1.pos, v1.vel)

    if case.family == "chase":
        gap = float(np.dot(v0.pos - v1.pos, v0.dir))
        target_gap = 7.0 + 1.3 * v1.speed
        risk_ttc = ttc if math.isfinite(ttc) else 10.0
        gain = negotiation_gain(variant, risk_ttc)
        if gap < target_gap or risk_ttc < 5.5:
            deficit = target_gap - gap + max(0.0, 5.5 - risk_ttc) * 1.4
            acc[1] = clamp(-gain * 1.25 * deficit, -5.0, 0.0)
            acc[0] = clamp(gain * 1.1, 0.0, 1.4)
        return acc, lat

    if case.family == "head_on":
        dist = float(np.linalg.norm(v0.pos - v1.pos))
        lateral_sep = abs(float(np.dot(v0.pos - v1.pos, v0.right)))
        risk_ttc = ttc if math.isfinite(ttc) else 10.0
        gain = negotiation_gain(variant, risk_ttc)
        if dist < 45.0 and lateral_sep < 3.0:
            lat_cmd = clamp(2.5 * gain + 0.35, 0.0, 2.4)
            lat[0] = lat_cmd
            lat[1] = lat_cmd
            acc[0] = acc[1] = clamp(-1.5 * gain, -3.0, 0.0)
        return acc, lat

    assert case.conflict is not None
    t0, t1 = time_to_conflict(v0, case.conflict), time_to_conflict(v1, case.conflict)
    if min(t0, t1) > 7.5 or max(t0, t1) == 0.0:
        return acc, lat
    arrival_gap = abs(t0 - t1)
    if arrival_gap >= 3.4:
        return acc, lat

    risk_ttc = min(ttc if math.isfinite(ttc) else 10.0, max(0.05, min(t0, t1)))
    gain = negotiation_gain(variant, risk_ttc)
    yielder = choose_yielder(case, variant, t0, t1, risk_ttc)
    other = 1 - yielder
    times = [t0, t1]
    target_gap = 2.7 if variant.utility == "handcrafted" else 2.4
    target_arrival = times[other] + target_gap
    dist_y = max(0.0, float(np.dot(case.conflict - case.vehicles[yielder].pos, case.vehicles[yielder].dir)))
    desired_speed = dist_y / max(target_arrival, 0.5)
    acc[yielder] = clamp(gain * 1.45 * (desired_speed - case.vehicles[yielder].speed), -4.8, 1.0)
    acc[other] = clamp(gain * 0.55, 0.0, 1.2)
    return acc, lat


def controls_for(case: ScenarioCase, variant: Variant) -> Tuple[List[float], List[float]]:
    if variant.name == "no_interaction":
        return [0.0, 0.0], [0.0, 0.0]
    if variant.name == "rule_expert":
        return rule_controls(case)
    return bargaining_controls(case, variant)


def case_finished(case: ScenarioCase, t: float) -> bool:
    if t >= case.max_time:
        return True
    v0, v1 = case.vehicles
    if case.family == "chase":
        return t >= case.max_time
    assert case.conflict is not None
    return progress_past_conflict(v0, case.conflict) > 9.0 and progress_past_conflict(v1, case.conflict) > 9.0


def run_case(base_case: ScenarioCase, variant: Variant, dt_step: float = 0.05) -> Dict[str, object]:
    case = clone_case(base_case)
    t = 0.0
    collision = False
    near_miss = False
    min_dist = math.inf
    min_ttc = math.inf
    intervention = 0.0
    max_decel = 0.0
    speed_integral = 0.0
    steps = 0

    while not case_finished(case, t):
        v0, v1 = case.vehicles
        dist = float(np.linalg.norm(v0.pos - v1.pos))
        ttc = pairwise_ttc(v0.pos, v0.vel, v1.pos, v1.vel)
        min_dist = min(min_dist, dist)
        if math.isfinite(ttc):
            min_ttc = min(min_ttc, ttc)
        if dist < 2.2:
            collision = True
            break
        if dist < 4.0 or (math.isfinite(ttc) and ttc < 1.0):
            near_miss = True

        acc, lat = controls_for(case, variant)
        for i, veh in enumerate(case.vehicles):
            a = clamp(acc[i], -5.5, 2.0)
            l = clamp(lat[i], -2.7, 2.7)
            max_decel = max(max_decel, max(0.0, -a))
            intervention += abs(a) * dt_step + 0.35 * abs(l) * dt_step
            veh.speed = clamp(veh.speed + a * dt_step, 0.0, 16.0)
            veh.pos = veh.pos + veh.dir * veh.speed * dt_step + veh.right * l * dt_step
        speed_integral += sum(v.speed for v in case.vehicles) * dt_step / 2.0
        t += dt_step
        steps += 1

    # Final distance sample after the last integration step.
    v0, v1 = case.vehicles
    final_dist = float(np.linalg.norm(v0.pos - v1.pos))
    min_dist = min(min_dist, final_dist)
    if final_dist < 2.2:
        collision = True
    if final_dist < 4.0:
        near_miss = True

    ideal_time = {
        "intersection": 5.2,
        "merge": 5.4,
        "chase": case.max_time,
        "head_on": 5.2,
    }[case.family]
    delay = max(0.0, t - ideal_time)
    avg_speed = speed_integral / max(t, dt_step)
    finite_min_ttc = min_ttc if math.isfinite(min_ttc) else 99.0

    return {
        "variant": variant.name,
        "variant_family": variant.family,
        "scenario": case.family,
        "case_id": case.case_id,
        "collision": int(collision),
        "near_miss": int(near_miss and not collision),
        "unsafe": int(collision or near_miss),
        "min_distance_m": round(min_dist, 3),
        "min_ttc_s": round(finite_min_ttc, 3),
        "time_s": round(t, 3),
        "delay_s": round(delay, 3),
        "intervention": round(intervention, 3),
        "max_decel_mps2": round(max_decel, 3),
        "avg_speed_mps": round(avg_speed, 3),
        "steps": steps,
        "discount_type": variant.discount_type,
        "rounds": variant.rounds,
        "alpha": variant.alpha,
        "utility": variant.utility,
        **{f"meta_{k}": v for k, v in case.meta.items()},
    }


def summarize(rows: List[Dict[str, object]], group_keys: Iterable[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    keys = list(group_keys)
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    out: List[Dict[str, object]] = []
    for key, vals in sorted(groups.items()):
        n = len(vals)
        item = {keys[i]: key[i] for i in range(len(keys))}
        item.update(
            {
                "n": n,
                "collision_pct": round(100.0 * sum(int(v["collision"]) for v in vals) / n, 1),
                "near_miss_pct": round(100.0 * sum(int(v["near_miss"]) for v in vals) / n, 1),
                "unsafe_pct": round(100.0 * sum(int(v["unsafe"]) for v in vals) / n, 1),
                "min_distance_m": round(float(np.mean([float(v["min_distance_m"]) for v in vals])), 3),
                "min_ttc_s": round(float(np.mean([float(v["min_ttc_s"]) for v in vals])), 3),
                "delay_s": round(float(np.mean([float(v["delay_s"]) for v in vals])), 3),
                "intervention": round(float(np.mean([float(v["intervention"]) for v in vals])), 3),
                "max_decel_mps2": round(float(np.mean([float(v["max_decel_mps2"]) for v in vals])), 3),
            }
        )
        out.append(item)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: List[Dict[str, object]]) -> None:
    order = [
        "no_interaction",
        "rule_expert",
        "constant_delta_r3",
        "ttc_r1_a0.3",
        "ttc_r3_a0.3",
        "ttc_r5_a0.3",
        "ttc_r3_a0.1",
        "ttc_r3_a0.6",
        "ttc_r3_a1.0",
        "handcrafted_utility_r3",
    ]
    by_name = {str(row["variant"]): row for row in summary}
    with path.open("w", encoding="utf-8") as f:
        f.write("| Variant | Coll% | Near% | Unsafe% | MinDist | MinTTC | Delay | Intervention |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for name in order:
            row = by_name.get(name)
            if row is None:
                continue
            f.write(
                f"| {name} | {row['collision_pct']} | {row['near_miss_pct']} | "
                f"{row['unsafe_pct']} | {row['min_distance_m']} | {row['min_ttc_s']} | "
                f"{row['delay_s']} | {row['intervention']} |\n"
            )


def plot_results(path: Path, summary: List[Dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [str(r["variant"]) for r in summary]
        coll = [float(r["collision_pct"]) for r in summary]
        unsafe = [float(r["unsafe_pct"]) for r in summary]
        dist = [float(r["min_distance_m"]) for r in summary]
        delay = [float(r["delay_s"]) for r in summary]

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        x = np.arange(len(labels))
        axes[0].bar(x - 0.18, coll, width=0.36, label="Collision %", color="#b84545")
        axes[0].bar(x + 0.18, unsafe, width=0.36, label="Unsafe %", color="#d99b45")
        axes[0].set_ylabel("Rate (%)")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend()
        axes[1].bar(x - 0.18, dist, width=0.36, label="Min distance (m)", color="#3f7f93")
        axes[1].bar(x + 0.18, delay, width=0.36, label="Delay (s)", color="#6f6f6f")
        axes[1].set_ylabel("Mean value")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()
        fig.suptitle("Mechanism Ablation v2: deterministic kinematic stress tests")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[plot skipped] {exc}")


def run_all() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    cases = build_cases()
    variants = build_variants()
    rows: List[Dict[str, object]] = []
    total = len(cases) * len(variants)
    idx = 0
    for variant in variants:
        for case in cases:
            idx += 1
            row = run_case(case, variant)
            rows.append(row)
            print(
                f"[{idx:03d}/{total}] {variant.name:<24} {case.case_id:<24} "
                f"coll={row['collision']} near={row['near_miss']} "
                f"dmin={row['min_distance_m']:>5} ttc={row['min_ttc_s']:>5}"
            )
    summary = summarize(rows, ["variant", "variant_family"])
    scenario_summary = summarize(rows, ["variant", "scenario"])
    return rows, summary, scenario_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="Optional output directory.")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else RESULT_ROOT / f"mechanism_ablation_v2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary, scenario_summary = run_all()
    write_csv(out_dir / "full.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "scenario_summary.csv", scenario_summary)
    write_markdown(out_dir / "paper_table.md", summary)
    plot_results(out_dir / "mechanism_ablation_v2.png", summary)

    config = {
        "created_at": ts,
        "collision_distance_m": 2.2,
        "near_miss_distance_m": 4.0,
        "cases": [c.case_id for c in build_cases()],
        "variants": [asdict(v) for v in build_variants()],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RESULT_ROOT / "mechanism_ablation_v2_latest.txt").write_text(str(out_dir), encoding="utf-8")
    print(f"\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
