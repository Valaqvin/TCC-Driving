import os, sys, json, math, datetime, csv, argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc,
)

OFFER_LABELS = ["speed_factor", "lateral_shift", "time_headway", "yield_will"]

SCENARIOS = {
    "intersection": {
        "a": {"start": [-30.0, -2.0], "target": [0.0, 0.0], "speed": 8.0},
        "b": {"start": [-2.0, -30.0], "target": [0.0, 0.0], "speed": 8.0},
        "desc": "Two vehicles cross at 90 degrees, classic intersection conflict.",
    },
    "chase": {
        "a": {"start": [-25.0, 0.0], "target": [15.0, 0.0], "speed": 10.0},
        "b": {"start": [-15.0, 0.0], "target": [15.0, 0.0], "speed": 6.0},
        "desc": "Faster behind slower. Negotiate overtake or fallback.",
    },
    "merge": {
        "a": {"start": [-25.0, -3.0], "target": [5.0, 0.0], "speed": 8.0},
        "b": {"start": [-30.0, 0.0], "target": [10.0, 0.0], "speed": 7.0},
        "desc": "Vehicle A merges from side lane into B lane.",
    },
    "head_on": {
        "a": {"start": [-20.0, 0.0], "target": [10.0, 0.0], "speed": 8.0},
        "b": {"start": [20.0, 0.0], "target": [-10.0, 0.0], "speed": 8.0},
        "desc": "Head-on approach. Must negotiate to avoid collision.",
    },
}


class SimState:
    """Two vehicles approaching a conflict point. Simple motion model."""
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario]
        self.dt = dt; self.name = scenario; self.step = 0
        self.collision = False; self.finished = False

        a, b = cfg["a"], cfg["b"]
        self.pos = [np.array(a["start"], dtype=np.float32),
                    np.array(b["start"], dtype=np.float32)]
        tgt_a = np.array(a["target"], dtype=np.float32) - self.pos[0]
        tgt_b = np.array(b["target"], dtype=np.float32) - self.pos[1]
        self.spd = [a["speed"], b["speed"]]
        self.hdg = [math.atan2(tgt_a[1], tgt_a[0]), math.atan2(tgt_b[1], tgt_b[0])]
        self._update_wps()

        # History buffers
        self.h = {k: [] for k in [
            "step", "ttc", "px0","py0","s0","hdg0","px1","py1","s1","hdg1",
            "dA0","dA1","dA2","dA3","dB0","dB1","dB2","dB3","converged"]}

    def _update_wps(self):
        self.wps = [np.zeros((10, 2), dtype=np.float32) for _ in range(2)]
        for v in range(2):
            for i in range(10):
                t = (i + 1) * self.dt
                self.wps[v][i] = [self.spd[v]*t*math.cos(self.hdg[v]),
                                  self.spd[v]*t*math.sin(self.hdg[v])]

    def adjust(self, vid, dx, dy):
        spdc = dx*math.cos(self.hdg[vid]) + dy*math.sin(self.hdg[vid])
        self.spd[vid] = max(0.3, min(15.0, self.spd[vid] + spdc * 2.0))

    def step_sim(self):
        if self.finished: return
        for v in range(2):
            self.pos[v] += np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),
                                     self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps()
        self.step += 1
        d = np.linalg.norm(self.pos[0] - self.pos[1])
        if d < 1.0: self.collision = True; self.finished = True
        if self.step > 200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished = True

    def get_state(self):
        pos = torch.from_numpy(np.array(self.pos)).float()
        vel = torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],
                            [self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]], dtype=torch.float32)
        hdg = torch.tensor(self.hdg, dtype=torch.float32)
        wps = torch.stack([torch.from_numpy(self.wps[0]),torch.from_numpy(self.wps[1])], dim=0)
        ttc_m = compute_pairwise_ttc(pos, vel, hdg)
        mt = ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,"record_len":torch.tensor([2]),"ttc":mt}


def run_scenario(variant, max_steps=80, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = SimState(variant)
    barg = RubinsteinBargaining({
        "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
        "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0}).to(device)
    barg.eval()

    log = []
    print(f"  {'Step':>5} | {'TTC':>7} | {'d':>6} | {'Offers_A':>32} | {'Offers_B':>32} | {'dWP':>7}")
    print("  " + "-"*100)

    while not sim.finished and sim.step < max_steps:
        s = sim.get_state()
        pos, vel, hdg, wps, rl = s["positions"].to(device), s["velocities"].to(device), s["headings"].to(device), s["waypoints"].to(device), s["record_len"].to(device)
        feat = torch.randn(2, 256, 48, 48, device=device)

        with torch.no_grad():
            out = barg(spatial_features=feat, waypoints=wps, positions=pos, velocities=vel, headings=hdg, record_len=rl)

        adj = out["adjusted_waypoints"]
        off = out["bargaining_offers"]; acc = out["acceptance_mask"]
        meta = out["metadata"]; conv = meta[0]["converged"]; nr = meta[0]["n_rounds"]
        tt = s["ttc"]
        delta = barg.discount(compute_pairwise_ttc(pos, vel, hdg))[0,1].item()
        oa = off[0].cpu().numpy(); ob = off[1].cpu().numpy()
        wpd = (adj - wps).norm(dim=-1).mean().item()

        log.append({
            "step":sim.step,"ttc":tt,"delta":delta,
            "sa":sim.spd[0],"sb":sim.spd[1],
            "pa":sim.pos[0].tolist(),"pb":sim.pos[1].tolist(),
            "oa":oa.tolist(),"ob":ob.tolist(),"wpd":wpd,"nr":nr,"conv":conv})

        # History
        h = sim.h
        h["step"].append(sim.step); h["ttc"].append(tt)
        h["px0"].append(sim.pos[0][0]); h["py0"].append(sim.pos[0][1])
        h["s0"].append(sim.spd[0]); h["hdg0"].append(sim.hdg[0])
        h["px1"].append(sim.pos[1][0]); h["py1"].append(sim.pos[1][1])
        h["s1"].append(sim.spd[1]); h["hdg1"].append(sim.hdg[1])
        h["dA0"].append(float(oa[0])); h["dA1"].append(float(oa[1]))
        h["dA2"].append(float(oa[2])); h["dA3"].append(float(oa[3]))
        h["dB0"].append(float(ob[0])); h["dB1"].append(float(ob[1]))
        h["dB2"].append(float(ob[2])); h["dB3"].append(float(ob[3]))
        h["converged"].append(conv)

        print(f"  {sim.step:>5} | {tt:>7.2f} | {delta:>6.3f} | [{oa[0]:.3f},{oa[1]:.3f},{oa[2]:.3f},{oa[3]:.3f}] | [{ob[0]:.3f},{ob[1]:.3f},{ob[2]:.3f},{ob[3]:.3f}] | {wpd:>7.4f}")

        for v in range(2):
            # adj may have shape (1, 10, 2) since module only adjusts ego waypoints
            if v < adj.shape[0]:
                dxy = (adj[v,0] - wps[v,0]).cpu().numpy()
            else:
                dxy = -(adj[0,0] - wps[0,0]).cpu().numpy() * 0.5
            sim.adjust(v, dxy[0], dxy[1])
        sim.step_sim()

    # Summary
    tt_vals = [x["ttc"] for x in log]
    min_d = min(math.sqrt((x["pa"][0]-x["pb"][0])**2+(x["pa"][1]-x["pb"][1])**2) for x in log) if log else 99
    summary = {
        "scenario":variant,"n_steps":len(log),"collision":sim.collision,
        "min_distance":round(min_d,2),
        "avg_ttc":round(float(np.mean(tt_vals)),2),"min_ttc":round(float(min(tt_vals)),2),
        "avg_delta":round(float(np.mean([x["delta"] for x in log])),3),
        "conv_rate":round(float(np.mean([x["conv"] for x in log])*100),1),
        "final_sa":round(log[-1]["sa"],2) if log else 0,
        "final_sb":round(log[-1]["sb"],2) if log else 0}
    print(); print("  " + "="*50); print("  Summary"); print("  " + "="*50)
    for k,v in summary.items(): print(f"    {k}: {v}")
    print("  " + "="*50)
    return log, sim, summary


def save_results(log, sim, summary, show_plot=True):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path(PROJECT_ROOT) / "Result" / f"test_{ts}"
    res_dir.mkdir(parents=True, exist_ok=True)

    (res_dir / "scenario_desc.txt").write_text(SCENARIOS[summary["scenario"]]["desc"])
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if log:
        with open(str(res_dir / "bargaining_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=log[0].keys())
            w.writeheader(); w.writerows(log)

    h = sim.h
    with open(str(res_dir / "kinematic_history.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(h.keys())
        for i in range(len(h["step"])): w.writerow([h[k][i] for k in h])

    np.save(str(res_dir / "bargain_log.npy"), np.array(log))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [x["step"] for x in log]
        fig, (a1,a2) = plt.subplots(2,1,figsize=(10,6),sharex=True)
        a1.plot(steps, [x["ttc"] for x in log],"b-o",ms=3); a1.set_ylabel("TTC (s)"); a1.grid(alpha=0.3)
        a2.plot(steps, [x["delta"] for x in log],"r-o",ms=3); a2.set_xlabel("Step"); a2.set_ylabel("delta")
        a2.grid(alpha=0.3)
        fig.suptitle(f"TTC -> Discount ({summary['scenario']})")
        plt.tight_layout(); plt.savefig(str(res_dir/"discount_over_time.png"),dpi=150); plt.close()

        oa = np.array([x["oa"] for x in log])
        ob = np.array([x["ob"] for x in log])
        fig, (a1,a2) = plt.subplots(2,1,figsize=(10,8))
        for i in range(4):
            a1.plot(steps, oa[:,i], label=OFFER_LABELS[i])
        a1.set_title("A Offers"); a1.legend(); a1.grid(alpha=0.3)
        for i in range(4):
            a2.plot(steps, ob[:,i], label=OFFER_LABELS[i])
        a2.set_xlabel("Step"); a2.set_title("B Offers"); a2.legend(); a2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(str(res_dir/"offers_over_time.png"),dpi=150); plt.close()

        plt.figure(figsize=(10,4))
        plt.plot(steps,[x["wpd"] for x in log],"g-o",ms=3)
        plt.xlabel("Step"); plt.ylabel("dWP (m)"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(str(res_dir/"waypoint_adjustment.png"),dpi=150); plt.close()

        plt.figure(figsize=(8,8))
        plt.plot(h["px0"],h["py0"],"b.-",label="A",ms=3)
        plt.plot(h["px1"],h["py1"],"r.-",label="B",ms=3)
        plt.plot(h["px0"][0],h["py0"][0],"bs",ms=8,label="A start")
        plt.plot(h["px1"][0],h["py1"][0],"rs",ms=8,label="B start")
        plt.axis("equal"); plt.grid(alpha=0.3); plt.legend()
        plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.title(f"Trajectories ({summary['scenario']})")
        plt.tight_layout(); plt.savefig(str(res_dir/"trajectories.png"),dpi=150); plt.close()

        print(f"  Plots -> {res_dir}")
    except Exception as e:
        print(f"  Plot skip: {e}")

    (Path(PROJECT_ROOT)/"Result"/"latest.txt").write_text(str(res_dir))
    print(f"  All data -> {res_dir}")
    return res_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=80)
    ap.add_argument("--variant",type=str,default="intersection",choices=list(SCENARIOS))
    ap.add_argument("--no-plot",action="store_true")
    ap.add_argument("--all-variants",action="store_true")
    args = ap.parse_args()

    if args.all_variants:
        for v in SCENARIOS:
            print(f"\n{'='*60}\n  Scenario: {v}\n{'='*60}")
            log, sim, sm = run_scenario(v, args.steps)
            save_results(log,sim,sm,show_plot=False)
        return

    print("="*60)
    print("  Rubinstein Bargaining Module Test")
    print(f"    Variant: {args.variant}  Steps: {args.steps}")
    print(f"    Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*60)
    log, sim, sm = run_scenario(args.variant, args.steps)
    res_dir = save_results(log, sim, sm, show_plot=not args.no_plot)
    print(f"\n  Result: {res_dir}")
    print(f"  Run: python \"{__file__}\" --variant=chase")

if __name__ == "__main__":
    main()
