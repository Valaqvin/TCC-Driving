#!/usr/bin/env python3
"""Run closed-loop CARLA evaluations for baseline and bargaining variants."""

import argparse
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATOR_PATH = PROJECT_ROOT / "simulation" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
AGENT_PATH = PROJECT_ROOT / "simulation" / "leaderboard" / "team_code" / "pnp_agent_e2e.py"
CONFIG_DIR = PROJECT_ROOT / "simulation" / "leaderboard" / "team_code" / "agent_config"
DEFAULT_ROUTES = PROJECT_ROOT / "simulation" / "leaderboard" / "data" / "routes_smoke_town10.xml"
DEFAULT_SCENARIOS = PROJECT_ROOT / "simulation" / "leaderboard" / "data" / "scenarios" / "no_scenarios.json"

VARIANTS = {
    "baseline": CONFIG_DIR / "pnp_config_codriving_eval_2ego.yaml",
    "bargaining": CONFIG_DIR / "pnp_config_bargaining_5_10.yaml",
}

PYTHONPATHS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "simulation" / "leaderboard",
    PROJECT_ROOT / "simulation" / "leaderboard" / "team_code" / "utils",
    PROJECT_ROOT / "simulation" / "scenario_runner",
    PROJECT_ROOT / "simulation" / "carla" / "WindowsNoEditor" / "PythonAPI" / "carla",
    Path(os.environ.get("CARLA_ROOT", "")) / "PythonAPI" / "carla",
]


def port_open(host, port, timeout=2.0):
    if host in {"localhost", "127.0.0.1"}:
        host = "127.0.0.1"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except OSError:
        return False


def prepare_env(output_root, checkpoint_endpoint, routes, scenarios):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in PYTHONPATHS] + [env.get("PYTHONPATH", "")])
    conda_prefix = sys.prefix
    env["PATH"] = os.pathsep.join([
        conda_prefix,
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
        os.path.join(conda_prefix, "Library", "usr", "bin"),
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Scripts"),
        env.get("PATH", ""),
    ])
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["SAVE_PATH"] = str(output_root)
    env["RESULT_ROOT"] = str(output_root)
    env["CHECKPOINT_ENDPOINT"] = str(checkpoint_endpoint)
    env["ROUTES"] = str(routes)
    env["SCENARIOS"] = str(scenarios)
    return env


def validate_inputs(config, routes, scenarios):
    required = [
        EVALUATOR_PATH,
        AGENT_PATH,
        config,
        routes,
        scenarios,
        PROJECT_ROOT / "ckpt" / "codriving" / "perception" / "net_epoch_bestval_at16.pth",
        PROJECT_ROOT / "ckpt" / "codriving" / "planning" / "epoch_26.ckpt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def run_variant(args, variant):
    config = VARIANTS[variant]
    routes = Path(args.routes).resolve()
    scenarios = Path(args.scenarios).resolve()
    validate_inputs(config, routes, scenarios)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() / f"{variant}_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_endpoint = output_root / "results.json"
    env = prepare_env(output_root, checkpoint_endpoint, routes, scenarios)

    cmd = [
        sys.executable, str(EVALUATOR_PATH),
        f"--host={args.carla_host}",
        f"--port={args.carla_port}",
        f"--routes={routes}",
        f"--scenarios={scenarios}",
        f"--agent={AGENT_PATH}",
        f"--agent-config={config}",
        f"--ego-num={args.ego_num}",
        f"--repetitions={args.repetitions}",
        f"--timeout={args.timeout}",
        f"--debug={args.debug}",
        f"--resume={args.resume}",
        f"--checkpoint={checkpoint_endpoint}",
    ]
    if args.record:
        record_dir = output_root / "recordings"
        record_dir.mkdir(parents=True, exist_ok=True)
        cmd.append(f"--record={record_dir}")

    print("=" * 72)
    print(f"Variant: {variant}")
    print(f"Config:  {config}")
    print(f"Routes:  {routes}")
    print(f"Output:  {output_root}")
    print("=" * 72)
    print(" ".join(str(c) for c in cmd))

    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)


def main():
    parser = argparse.ArgumentParser(description="Run baseline/bargaining CARLA evals")
    parser.add_argument("--variant", choices=["baseline", "bargaining", "suite"], default="suite")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--routes", default=str(DEFAULT_ROUTES))
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    parser.add_argument("--ego-num", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", default="60000")
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "Result" / "closed_loop_suite"))
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-port-check", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.skip_port_check:
        if not port_open(args.carla_host, args.carla_port):
            raise RuntimeError(
                f"CARLA is not reachable at {args.carla_host}:{args.carla_port}. "
                "Start the CARLA server first, or pass --skip-port-check if you "
                "know the evaluator should connect differently.")

    variants = ["baseline", "bargaining"] if args.variant == "suite" else [args.variant]
    for variant in variants:
        code = run_variant(args, variant)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
