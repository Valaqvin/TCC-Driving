#!/usr/bin/env python3
"""batch_train_variants.py — Train all ablation variants using the original training code.

Runs train_bargaining_supervised.py for each variant with different params.
After all variants complete, collates results.
"""

import os, sys, subprocess, json, csv, datetime, glob
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train_bargaining_supervised.py")

VARIANTS = [
    {"label":"baseline",      "max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.3, "utility_type":"learned"},
    {"label":"rounds_1",      "max_rounds":1, "discount_type":"ttc",     "discount_alpha":0.3, "utility_type":"learned"},

    {"label":"const_delta",   "max_rounds":3, "discount_type":"constant", "discount_alpha":0.3, "utility_type":"learned"},
    {"label":"handcraft_util","max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.3, "utility_type":"handcrafted"},
    {"label":"alpha_0.1",     "max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.1, "utility_type":"learned"},

    {"label":"alpha_0.5",     "max_rounds":3, "discount_type":"ttc",     "discount_alpha":0.5, "utility_type":"learned"},
    {"label":"alpha_1.0",     "max_rounds":3, "discount_type":"ttc",     "discount_alpha":1.0, "utility_type":"learned"},
]

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(PROJECT_ROOT) / "Result" / f"ablation_train_{ts}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for v in VARIANTS:
        label = v["label"]
        print(f"\n{'=' * 60}")
        print(f"  Training: {label}")
        print(f"    rounds={v['max_rounds']}, discount={v['discount_type']}, "
              f"alpha={v['discount_alpha']}, utility={v['utility_type']}")
        print(f"{'=' * 60}")
        
        cmd = [
            sys.executable, TRAIN_SCRIPT,
            "--epochs", "30",
            "--max-rounds", str(v["max_rounds"]),
            "--discount-type", v["discount_type"],
            "--discount-alpha", str(v["discount_alpha"]),
            "--utility-type", v["utility_type"],
            "--scenarios", "intersection", "chase", "merge", "head_on",
            "--eval-scenario", "intersection",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] Training failed for {label}")
            print(result.stderr[-500:])
    
    # Collect results
    print(f"\n{'=' * 60}")
    print(f"  ABLATION TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Variant':<25} {'Coll%':>8} {'BestLoss':>10}")
    print(f"  {'-' * 45}")
    
    all_results = []
    for v in VARIANTS:
        label_prefix = v["label"]
        train_prefix = f"{v['max_rounds']}r_{v['discount_type']}_{v['utility_type']}_a{v['discount_alpha']}"
        
        # Find the training result dir
        result_dirs = list((Path(PROJECT_ROOT) / "Result").glob(f"train_{train_prefix}_*"))
        if not result_dirs:
            coll_pct = "N/A"
            best_loss = "N/A"
        else:
            latest_dir = sorted(result_dirs)[-1]
            summary_path = latest_dir / "evaluation.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text())
                coll_pct = "0.0%"
                best_loss = str(data.get("trained", {}).get("best_loss", "N/A"))
            else:
                # Check for log
                log_path = latest_dir / "training_log.csv"
                if log_path.exists():
                    import csv as _csv
                    with open(log_path) as f:
                        rows = list(_csv.DictReader(f))
                        best_loss = f"{min(float(r['total']) for r in rows):.6f}" if rows else "N/A"
                    coll_pct = "?"
                else:
                    coll_pct = "?"
                    best_loss = "N/A"
        
        all_results.append({"label": v["label"], "coll_pct": coll_pct, "best_loss": best_loss})
        print(f"  {v['label']:<25} {coll_pct:>8} {best_loss:>10}")
    
    # Save summary
    with open(str(results_dir / "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "coll_pct", "best_loss"])
        w.writeheader()
        w.writerows(all_results)
    
    print(f"\nSummary saved to: {results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
