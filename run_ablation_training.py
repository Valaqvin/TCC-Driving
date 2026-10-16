#!/usr/bin/env python3
"""run_ablation_training.py — Train and evaluate ablation variants.

Trains each variant for 30 epochs, evaluates on 4 scenarios,
and produces a comparison table.

Usage:
    python run_ablation_training.py
    python run_ablation_training.py --epochs 50
"""

import os, sys, json, math, datetime, csv, copy
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, UtilityEstimator, OfferDecoder,
    TTCBasedDiscount, compute_pairwise_ttc,
)

# ========== Simulator ==========
SCENARIOS = {
    "intersection":{"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase":{"a":{"start":[-25,0],"target":[15,0],"speed":6},"b":{"start":[-15,0],"target":[15,0],"speed":10}},
    "merge":{"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on":{"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimState:
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario];self.dt=dt;self.name=scenario;self.step=0
        self.collision=False;self.finished=False
        a,b=cfg["a"],cfg["b"]
        self.pos=[np.array(a["start"],np.float32),np.array(b["start"],np.float32)]
        ta=np.array(a["target"],np.float32)-self.pos[0]
        tb=np.array(b["target"],np.float32)-self.pos[1]
        self.spd=[a["speed"],b["speed"]]
        self.hdg=[math.atan2(ta[1],ta[0]),math.atan2(tb[1],tb[0])]
        self._update_wps()
    def _update_wps(self):
        self.wps=[np.zeros((10,2),np.float32) for _ in range(2)]
        for v in range(2):
            for i in range(10):
                t=(i+1)*self.dt
                self.wps[v][i]=[self.spd[v]*t*math.cos(self.hdg[v]),self.spd[v]*t*math.sin(self.hdg[v])]
    def adjust(self,vid,dx,dy):
        spdc=dx*math.cos(self.hdg[vid])+dy*math.sin(self.hdg[vid])
        self.spd[vid]=max(0.3,min(15.0,self.spd[vid]+spdc*2.0))
    def step_sim(self):
        if self.finished:return
        for v in range(2):
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps();self.step+=1
        d=np.linalg.norm(self.pos[0]-self.pos[1])
        if d<1.0:self.collision=True;self.finished=True
        if self.step>200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished=True
    def get_state(self):
        pos=torch.from_numpy(np.array(self.pos)).float().to(DEVICE)
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],
                          [self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],
                         dtype=torch.float32,device=DEVICE)
        wps=torch.from_numpy(np.array(self.wps)).float().to(DEVICE)
        return {"positions":pos,"velocities":vel,"waypoints":wps,"headings":torch.tensor(self.hdg,device=DEVICE),
                "record_len":torch.tensor([2],device=DEVICE),"ttc":float(compute_pairwise_ttc(pos,vel,torch.tensor(self.hdg,device=DEVICE)).min().cpu().numpy())}


# ========== Variants ==========
VARIANTS = [
    {"label":"baseline","discount_type":"ttc","utility_type":"learned","max_rounds":3,"discount_alpha":0.3},
    {"label":"rounds_1","discount_type":"ttc","utility_type":"learned","max_rounds":1,"discount_alpha":0.3},
    {"label":"const_delta","discount_type":"constant","utility_type":"learned","max_rounds":3,"discount_alpha":0.3},
    {"label":"handcraft_util","discount_type":"ttc","utility_type":"handcrafted","max_rounds":3,"discount_alpha":0.3},
    {"label":"alpha_0.1","discount_type":"ttc","utility_type":"learned","max_rounds":3,"discount_alpha":0.1},
    {"label":"alpha_1.0","discount_type":"ttc","utility_type":"learned","max_rounds":3,"discount_alpha":1.0},
]


class ConstantDiscount(nn.Module):
    def __init__(self, delta=0.9):
        super().__init__();self.delta=nn.Parameter(torch.tensor(delta),requires_grad=False)
    def forward(self, ttc):return self.delta.expand_as(ttc)


class HandcraftedUtility(nn.Module):
    def forward(self, state, offer):
        return torch.sigmoid(10.0 * (offer[:, 3] - 0.5 * offer[:, 0]) - 2.0)


def create_variant(**kwargs):
    label = kwargs.get("label","variant")
    discount_type = kwargs.get("discount_type","ttc")
    utility_type = kwargs.get("utility_type","learned")
    max_rounds = kwargs.get("max_rounds",3)
    discount_alpha = kwargs.get("discount_alpha",0.3)
    
    cfg = {"state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":max_rounds,
           "discount_alpha":discount_alpha,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0}
    model = RubinsteinBargaining(cfg).to(DEVICE)
    if discount_type == "constant":
        model.discount = ConstantDiscount(delta=0.9).to(DEVICE)
    if utility_type == "handcrafted":
        model.utility_estimator = HandcraftedUtility().to(DEVICE)
    model.label = label
    return model


def run_episode(model, scenario, max_steps=60):
    sim = SimState(scenario); conv=0; wpds=[]; ttc_log=[]
    while not sim.finished and sim.step < max_steps:
        s = sim.get_state()
        feat = torch.randn(2, 256, 48, 48, device=DEVICE)
        with torch.no_grad():
            out = model(spatial_features=feat, waypoints=s["waypoints"],
                        positions=s["positions"], velocities=s["velocities"],
                        headings=s["headings"], record_len=s["record_len"])
        wpd = (out["adjusted_waypoints"]-s["waypoints"]).norm(dim=-1).mean().item()
        wpds.append(wpd); ttc_log.append(s["ttc"])
        if out["metadata"][0]["converged"]: conv+=1
        for v in range(2):
            if v<out["adjusted_waypoints"].shape[0]:
                dxy=(out["adjusted_waypoints"][v,0]-s["waypoints"][v,0]).cpu().numpy()
            else:
                dxy=-(out["adjusted_waypoints"][0,0]-s["waypoints"][0,0]).cpu().numpy()*0.5
            sim.adjust(v,dxy[0],dxy[1])
        sim.step_sim()
    n=max(1,sim.step)
    return {"label":model.label,"scenario":scenario,"steps":sim.step,"collision":sim.collision,
            "conv":round(conv/n*100,2),"min_ttc":round(min(ttc_log)if ttc_log else 0,3),
            "avg_wpd":round(np.mean(wpds)if wpds else 0,4)}


def evaluate_model(model, scenarios, trials=5):
    """Evaluate model on each scenario with N trials."""
    results = []
    orig_rng = torch.get_rng_state()
    for sc in scenarios:
        for t in range(trials):
            torch.manual_seed(t)
            r = run_episode(model, sc)
            r["trial"] = t
            results.append(r)
    torch.set_rng_state(orig_rng)
    return results


def train_model(model, n_epochs=30, lr=1e-3, ep_per_epoch=5):
    """Train model using supervised learning (adapted from train_bargaining_supervised.py)."""
    model.train()
    for p in model.state_encoder.parameters(): p.requires_grad = False
    opt = optim.Adam([
        {"params": model.offer_decoder.parameters(), "lr": lr},
        {"params": model.utility_estimator.parameters(), "lr": lr},
    ])
    mse = nn.MSELoss()
    margin = nn.MarginRankingLoss(margin=0.5)
    
    scenarios = list(SCENARIOS.keys())
    log = {"epoch":[],"offer_loss":[],"util_loss":[],"total":[]}
    best_loss = float("inf")
    
    for ep in range(n_epochs):
        ep_off_loss, ep_util_loss, ep_total = 0, 0, 0
        for _ in range(ep_per_epoch):
            sc = np.random.choice(scenarios)
            sim = SimState(sc)
            while not sim.finished and sim.step < 50:
                s = sim.get_state()
                feat = torch.randn(2, 256, 48, 48, device=DEVICE)
                ttc = s["ttc"]
                
                # Generate GT offers
                gt = torch.zeros(2, 4, device=DEVICE)
                if ttc < 3.0:
                    gt[0] = torch.tensor([0.15, 0.3, 0.5, 0.8])
                    gt[1] = torch.tensor([0.75, 0.3, 0.5, 0.2])
                elif ttc < 6.0:
                    ratio = min(1.0, sim.spd[1] / max(sim.spd[0], 0.1))
                    gt[0] = torch.tensor([0.4*ratio, 0.3, 0.5, 0.6])
                    gt[1] = torch.tensor([0.4*(1-ratio), 0.3, 0.5, 0.4])
                elif ttc < 10.0:
                    gt[:] = torch.tensor([0.45, 0.3, 0.5, 0.5])
                else:
                    gt[:] = torch.tensor([0.5, 0.3, 0.5, 0.5])
                
                # Forward
                states = model.state_encoder(feat)
                pred = model.offer_decoder(states)
                off_loss = mse(pred, gt)
                
                # Utility ranking
                bad = 1.0 - gt
                u_good = model.utility_estimator(states.detach(), gt).view(-1)
                u_bad = model.utility_estimator(states.detach(), bad).view(-1)
                target = torch.ones(2, device=DEVICE)
                util_loss = margin(u_good, u_bad, target)
                
                total = off_loss + 0.5 * util_loss
                opt.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                ep_off_loss += off_loss.item()
                ep_util_loss += util_loss.item()
                ep_total += total.item()
                
                # Sim step
                with torch.no_grad():
                    model.eval()
                    out = model(spatial_features=feat, waypoints=s["waypoints"],
                                positions=s["positions"], velocities=s["velocities"],
                                headings=s["headings"], record_len=s["record_len"])
                    model.train()
                for v in range(2):
                    if v<out["adjusted_waypoints"].shape[0]:
                        dxy=(out["adjusted_waypoints"][v,0]-s["waypoints"][v,0]).cpu().numpy()
                    else:
                        dxy=-(out["adjusted_waypoints"][0,0]-s["waypoints"][0,0]).cpu().numpy()*0.5
                    sim.adjust(v,dxy[0],dxy[1])
                sim.step_sim()
        
        avg_off = ep_off_loss/(ep_per_epoch*50)
        avg_util = ep_util_loss/(ep_per_epoch*50)
        avg_total = ep_total/(ep_per_epoch*50)
        log["epoch"].append(ep);log["offer_loss"].append(avg_off)
        log["util_loss"].append(avg_util);log["total"].append(avg_total)
        
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep:>3d} | offer={avg_off:.6f} | util={avg_util:.6f} | total={avg_total:.6f}")
        if avg_total < best_loss:
            best_loss = avg_total
    
    model.eval()
    return model, log, best_loss


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=5, help="eval trials per scenario")
    ap.add_argument("--variants", nargs="+", default=[v["label"] for v in VARIANTS])
    args = ap.parse_args()
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path(PROJECT_ROOT) / "Result" / f"ablation_train_{ts}"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for v in VARIANTS:
        if v["label"] not in args.variants:
            continue
        label = v["label"]
        print(f"\n{'='*60}")
        print(f"  Training: {label}")
        print(f"    discount={v['discount_type']}, utility={v['utility_type']}, "
              f"rounds={v['max_rounds']}, alpha={v['discount_alpha']}")
        print(f"{'='*60}")
        
        model = create_variant(**v)
        print(f"  Model created. Training for {args.epochs} epochs...")
        model, log, best_loss = train_model(model, n_epochs=args.epochs)
        
        # Save checkpoint
        ckpt_path = str(res_dir / f"{label}_final.pth")
        torch.save({
            "offer_decoder": model.offer_decoder.state_dict(),
            "utility_estimator": model.utility_estimator.state_dict(),
            "config": v, "log": log, "best_loss": best_loss,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")
        
        # Evaluate
        print(f"  Evaluating on {len(SCENARIOS)} scenarios x {args.trials} trials...")
        eval_results = evaluate_model(model, list(SCENARIOS.keys()), trials=args.trials)
        for r in eval_results:
            r["variant"] = label
            all_results.append(r)
        
        # Print eval summary
        coll = sum(1 for r in eval_results if r["collision"])
        total = len(eval_results)
        avg_ttc = np.mean([r["min_ttc"] for r in eval_results])
        avg_wpd = np.mean([r["avg_wpd"] for r in eval_results])
        print(f"  Results: collision={coll}/{total} ({coll/total*100:.1f}%), "
              f"avg_ttc={avg_ttc:.3f}s, avg_wpd={avg_wpd:.4f}m")
    
    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"  ABLATION TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Variant':<20} {'Coll%':>8} {'AvgTTC':>8} {'AvgWPD':>8} {'BestLoss':>10}")
    print(f"  {'-'*54}")
    
    from collections import defaultdict
    g = defaultdict(list)
    for r in all_results: g[r["variant"]].append(r)
    
    summary_rows = []
    for label, results in sorted(g.items()):
        coll = sum(1 for r in results if r["collision"])
        total = len(results)
        coll_pct = coll/total*100
        avg_ttc = np.mean([r["min_ttc"] for r in results])
        avg_wpd = np.mean([r["avg_wpd"] for r in results])
        
        # Find best loss from checkpoint
        ckpt = torch.load(str(res_dir / f"{label}_final.pth"), map_location="cpu", weights_only=False)
        bl = ckpt["best_loss"]
        
        summary_rows.append({"label":label,"coll_pct":coll_pct,"avg_ttc":avg_ttc,
                             "avg_wpd":avg_wpd,"best_loss":bl})
        print(f"  {label:<20} {coll_pct:>7.1f}% {avg_ttc:>7.3f}s {avg_wpd:>7.4f}m {bl:>9.6f}")
    
    # Save summary
    with open(str(res_dir / "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label","coll_pct","avg_ttc","avg_wpd","best_loss"])
        w.writeheader(); w.writerows(summary_rows)
    
    # Save full results
    with open(str(res_dir / "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","label","scenario","trial","steps","collision","conv","min_ttc","avg_wpd"])
        w.writeheader(); w.writerows(all_results)
    
    print(f"\nResults saved to: {res_dir}")
    print(f"Summary: {res_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
