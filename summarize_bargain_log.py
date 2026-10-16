#!/usr/bin/env python3
"""Summarize the CARLA bargaining log produced by pnp_infer_action_e2e_bargaining."""

import argparse
import math
from pathlib import Path

import numpy as np


def finite(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def main():
    parser = argparse.ArgumentParser(description='Summarize bargain_log.npy')
    parser.add_argument('log', nargs='?', default='Result/carla_eval_test/bargain_log.npy')
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise FileNotFoundError(f'No bargain log found at {path}')

    entries = np.load(path, allow_pickle=True).tolist()
    if not entries:
        print('No bargaining entries found.')
        return

    active = [e for e in entries if e.get('status') != 'skipped']
    skipped = [e for e in entries if e.get('status') == 'skipped']
    deltas = [float(e.get('max_waypoint_delta', 0.0)) for e in active]
    ttcs = finite([float(e.get('min_ttc', float('inf'))) for e in active])
    converged = [bool(e.get('converged', False)) for e in active]
    modes = {}
    for entry in active:
        mode = entry.get('input_mode', 'unknown')
        modes[mode] = modes.get(mode, 0) + 1
    reasons = {}
    for entry in skipped:
        reason = entry.get('reason', 'unknown')
        reasons[reason] = reasons.get(reason, 0) + 1

    print(f'log: {path}')
    print(f'total log entries: {len(entries)}')
    print(f'active bargaining steps: {len(active)}')
    print(f'skipped steps: {len(skipped)}')
    if reasons:
        print(f'skip reasons: {reasons}')
    print(f'input modes: {modes}')
    if converged:
        print(f'convergence rate: {100.0 * sum(converged) / len(converged):.2f}%')
    if deltas:
        print(f'mean max waypoint delta: {np.mean(deltas):.4f} m')
        print(f'max waypoint delta: {np.max(deltas):.4f} m')
    if ttcs:
        print(f'mean finite min TTC: {np.mean(ttcs):.4f} s')
        print(f'min finite TTC: {np.min(ttcs):.4f} s')
    else:
        print('finite TTC: none')


if __name__ == '__main__':
    main()
