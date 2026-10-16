#!/usr/bin/env python3
"""Preflight checks before running CARLA closed-loop experiments on a server."""

import argparse
import importlib
import os
import socket
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    "ckpt/codriving/perception/config.yaml",
    "ckpt/codriving/perception/net_epoch_bestval_at16.pth",
    "ckpt/codriving/planning/epoch_26.ckpt",
    "simulation/leaderboard/team_code/agent_config/pnp_config_bargaining_5_10.yaml",
    "simulation/leaderboard/team_code/agent_config/pnp_config_codriving_eval_2ego.yaml",
    "simulation/leaderboard/data/routes_smoke_town10.xml",
    "simulation/leaderboard/data/scenarios/no_scenarios.json",
    "launch_closed_loop_suite.py",
    "check_bargaining_integration.py",
]


def check_file(path):
    full = PROJECT_ROOT / path
    if not full.exists():
        raise FileNotFoundError(str(full))
    print(f"OK file: {path}")


def check_import(name):
    module = importlib.import_module(name)
    location = getattr(module, "__file__", "built-in")
    version = getattr(module, "__version__", "")
    suffix = f" version={version}" if version else ""
    print(f"OK import: {name} ({location}){suffix}")
    return module


def check_carla_server(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect((host, port))
    finally:
        sock.close()

    carla = check_import("carla")
    client = carla.Client(host, port)
    client.set_timeout(5.0)
    server_version = client.get_server_version()
    world = client.get_world()
    print(f"OK CARLA server: version={server_version} map={world.get_map().name}")


def run(cmd):
    print("RUN " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Server preflight for CoLMDriver experiments")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--skip-carla", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    print(f"python: {sys.executable}")
    print(f"version: {sys.version}")
    print(f"cwd: {PROJECT_ROOT}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"CARLA_ROOT={os.environ.get('CARLA_ROOT', '')}")

    for path in REQUIRED_FILES:
        check_file(path)

    check_import("numpy")
    torch = check_import("torch")
    check_import("yaml")
    check_import("cv2")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch cuda device count: {torch.cuda.device_count()}")
        print(f"torch cuda device 0: {torch.cuda.get_device_name(0)}")

    if not args.skip_carla:
        check_carla_server(args.host, args.port)

    run([sys.executable, "-m", "py_compile",
         "simulation/leaderboard/team_code/pnp_infer_action_e2e_bargaining.py",
         "opencood/models/comm_modules/rubinstein_bargaining.py",
         "launch_closed_loop_suite.py",
         "summarize_bargain_log.py",
         "summarize_closed_loop_results.py",
         "summarize_rollout_trace.py"])

    if not args.skip_smoke:
        run([sys.executable, "check_bargaining_integration.py"])
        run([sys.executable, "launch_closed_loop_suite.py",
             "--variant", "suite", "--dry-run",
             "--carla-host", args.host, "--carla-port", str(args.port)])

    print("PRECHECK_OK")


if __name__ == "__main__":
    main()
