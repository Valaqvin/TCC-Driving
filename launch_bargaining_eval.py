#!/usr/bin/env python3
"""
launch_bargaining_eval.py — Launch CARLA evaluation with Rubinstein bargaining.

Usage:
    python launch_bargaining_eval.py [--carla-port 2000] [--show-viz]

This script launches the leaderboard evaluator with:
    - The PnP agent that uses pnp_infer_action_e2e_bargaining
    - The bargaining-integrated config
    - 2 ego vehicles (minimum needed for meaningful bargaining)

Prerequisites:
    - CARLA 0.9.15 server running on the specified port
    - Pre-trained perception checkpoint at: ckpt/codriving/perception
    - Pre-trained planning checkpoint at: ckpt/codriving/planning/planner_epoch_27.ckpt
    - Route file and scenario config file
"""

import os
import sys
import argparse
import subprocess

# Adjust path to find the leaderboard evaluator
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVALUATOR_PATH = os.path.join(
    PROJECT_ROOT, "simulation", "leaderboard", "leaderboard", "leaderboard_evaluator.py"
)
AGENT_PATH = os.path.join(
    PROJECT_ROOT, "simulation", "leaderboard", "team_code", "pnp_agent_e2e.py"
)
AGENT_CONFIG = os.path.join(
    PROJECT_ROOT, "simulation", "leaderboard", "team_code", "agent_config",
    "pnp_config_bargaining_5_10.yaml"
)
ROUTES_FILE = os.path.join(
    PROJECT_ROOT, "simulation", "leaderboard", "data", "routes_codriving.xml"
)
SCENARIOS_FILE = os.path.join(
    PROJECT_ROOT, "simulation", "leaderboard", "data", "scenarios", "no_scenarios.json"
)

PYTHONPATHS = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "simulation", "leaderboard"),
    os.path.join(PROJECT_ROOT, "simulation", "leaderboard", "team_code", "utils"),
    os.path.join(PROJECT_ROOT, "simulation", "scenario_runner"),
    os.path.join(PROJECT_ROOT, "simulation", "carla", "WindowsNoEditor", "PythonAPI", "carla"),
]


def main():
    os.chdir(PROJECT_ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        PYTHONPATHS + [os.environ.get("PYTHONPATH", "")]
    )
    conda_prefix = sys.prefix
    os.environ["PATH"] = os.pathsep.join([
        conda_prefix,
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
        os.path.join(conda_prefix, "Library", "usr", "bin"),
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Scripts"),
        os.environ.get("PATH", ""),
    ])
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    parser = argparse.ArgumentParser(description="Launch bargaining evaluation")
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", default=2000, type=int)
    parser.add_argument("--routes", default=ROUTES_FILE)
    parser.add_argument("--scenarios", default=SCENARIOS_FILE)
    parser.add_argument("--ego-num", default=2, type=int,
                        help="Number of ego vehicles (default: 2 for bargaining)")
    parser.add_argument("--repetitions", default=1, type=int)
    parser.add_argument("--record", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default=0, type=int)
    parser.add_argument("--show-viz", action="store_true",
                        help="Show visualization window")
    parser.add_argument("--timeout", default="60000")
    parser.add_argument("--debug", default=1, type=int)
    args = parser.parse_args()

    args.routes = os.path.abspath(args.routes)
    args.scenarios = os.path.abspath(args.scenarios)
    args.record = os.path.abspath(args.record) if args.record else ""
    args.checkpoint = os.path.abspath(args.checkpoint) if args.checkpoint else ""
    output_root = os.path.abspath(args.record or os.path.join(PROJECT_ROOT, "Result", "carla_eval_test"))
    checkpoint_endpoint = os.path.abspath(args.checkpoint or os.path.join(output_root, "results.json"))

    cmd = [
        sys.executable, EVALUATOR_PATH,
        f"--host={args.carla_host}",
        f"--port={args.carla_port}",
        f"--routes={args.routes}",
        f"--scenarios={args.scenarios}",
        f"--agent={AGENT_PATH}",
        f"--agent-config={AGENT_CONFIG}",
        f"--ego-num={args.ego_num}",
        f"--repetitions={args.repetitions}",
        f"--timeout={args.timeout}",
        f"--debug={args.debug}",
        f"--resume={args.resume}",
        f"--checkpoint={checkpoint_endpoint}",
    ]

    if args.record:
        cmd.append(f"--record={args.record}")
    if args.show_viz:
        cmd.append("--track=SENSORS")

    os.environ.setdefault("SAVE_PATH", output_root)
    os.environ.setdefault("RESULT_ROOT", output_root)
    os.environ.setdefault("CHECKPOINT_ENDPOINT", checkpoint_endpoint)
    os.environ.setdefault("ROUTES", os.path.abspath(args.routes))
    os.environ.setdefault("SCENARIOS", os.path.abspath(args.scenarios))

    print("=" * 60)
    print("Launching CoLMDriver evaluation with Rubinstein bargaining")
    print("=" * 60)
    print(f"  Agent:       {AGENT_PATH}")
    print(f"  Config:      {AGENT_CONFIG}")
    print(f"  Routes:      {args.routes}")
    print(f"  Scenarios:   {args.scenarios}")
    print(f"  Ego vehicles: {args.ego_num}")
    print(f"  CARLA:       {args.carla_host}:{args.carla_port}")
    print()
    print("  Bargaining module: active")
    print("    Protocol:  Rubinstein alternating-offers")
    print("    Discount:  TTC-based delta = 1 - exp(-alpha * TTC)")
    print("    Max rounds: configured in pnp_config_bargaining_5_10.yaml")
    print("    Logging:   terminal + bargain_log.npy")
    print("=" * 60)

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
