# -*- coding: utf-8 -*-
"""test_bargaining_ablation.py - Automated ablation study for Rubinstein bargaining.

Tests all combinations of key design choices:
  1. Discount type:   TTC-based vs constant
  2. Bargaining rounds: 1 vs 3
  3. Utility type:      learned vs handcrafted
  4. Discount alpha:    0.1 / 0.3 / 0.5 / 1.0 (TTC-only)

Each combination runs on all 4 scenarios. Results saved to Result/ablation_*/.
"""

import os, sys, json, math, datetime, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc,
)

SCENARIOS = {
    "intersection": {"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase": {"a":{"start":[-25,0],"target":[15,0],"speed":6},"b":{"start":[-15,0],"target":[15,0],"speed":10}},
    "merge": {"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on": {"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
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
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),
                                   self.spd[v]*self.dt*math.sin(self.hdg[v])])
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
        hdg=torch.tensor(self.hdg,dtype=torch.float32,device=DEVICE)
        wps=torch.stack([torch.from_numpy(self.wps[0]).to(DEVICE),torch.from_numpy(self.wps[1]).to(DEVICE)],dim=0)
        ttc_m=compute_pairwise_ttc(pos,vel,hdg)
        mt=ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,
                "record_len":torch.tensor([2],device=DEVICE),"ttc":mt,"spd":list(self.spd)}


# ── Ablation variants ──
class HandcraftedUtility(nn.Module):
    def forward(self, agent_state, offer):
        balance = 1.0 - torch.abs(offer - 0.5).mean(dim=-1, keepdim=True)
        return balance


class ConstantDiscount(nn.Module):
    def __init__(self, delta=0.9):
        super().__init__(); self.delta = delta
    def forward(self, ttc):
        return torch.full_like(ttc, self.delta, device=ttc.device)


def create_variant(label, discount_type="ttc", max_rounds=3, utility_type="learned", discount_alpha=0.3):
    cfg = {"state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":max_rounds,
           "discount_alpha":discount_alpha,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0}
    model = RubinsteinBargaining(cfg).to(DEVICE); model.eval()
    if discount_type == "constant": model.discount = ConstantDiscount(delta=0.9).to(DEVICE)
    if utility_type == "handcrafted": model.utility_estimator = HandcraftedUtility().to(DEVICE)
    model.label = label; return model


ABLATIONS = [
    {"name":"discount_ttc_vs_const","desc":"TTC-based vs constant delta=0.9","configs":[
        {"label":"ttc_default","discount_type":"ttc","discount_alpha":0.3},
        {"label":"constant_delta","discount_type":"constant"},
    ]},
    {"name":"rounds_1_vs_3","desc":"1 bargaining round vs 3 rounds","configs":[
        {"label":"rounds_3","discount_type":"ttc","max_rounds":3,"discount_alpha":0.3},
        {"label":"rounds_1","discount_type":"ttc","max_rounds":1,"discount_alpha":0.3},
    ]},
    {"name":"utility_learned_vs_hand","desc":"Learned vs hand-crafted utility","configs":[
        {"label":"utility_learned","discount_type":"ttc","utility_type":"learned","discount_alpha":0.3},
        {"label":"utility_hand","discount_type":"ttc","utility_type":"handcrafted","discount_alpha":0.3},
    ]},
    {"name":"alpha_0p1_0p3_0p5_1p0","desc":"Discount alpha values","configs":[
        {"label":"alpha_0.1","discount_type":"ttc","discount_alpha":0.1},
        {"label":"alpha_0.3","discount_type":"ttc","discount_alpha":0.3},
        {"label":"alpha_0.5","discount_type":"ttc","discount_alpha":0.5},
        {"label":"alpha_1.0","discount_type":"ttc","discount_alpha":1.0},
    ]},
]


def run_episode(model, scenario, max_steps=60, seed=None):
    if seed is not None: torch.manual_seed(seed)
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


def run_all(scenarios=None, max_steps=60, checkpoint=None, trials=1):
    if scenarios is None: scenarios = list(SCENARIOS)
    results = []
    seed_counter = [0]
    print(f"Running {len(ABLATIONS)} ablations x {len(scenarios)} scenarios ({len(scenarios)*sum(len(a['configs']) for a in ABLATIONS)} total runs)...\n")
    for ab in ABLATIONS:
        print(f"[{ab['name']}] {ab['desc']}")
        for cfg in ab["configs"]:
            model = create_variant(**cfg)
            if checkpoint:
                model = load_ckpt(model, checkpoint)
            for sc in scenarios:
                for t in range(trials):
                    if trials > 1:
                        torch.manual_seed(seed_counter[0])
                        seed_counter[0] += 1
                        trial_model = create_variant(**cfg)
                    else:
                        trial_model = model
                    if checkpoint:
                        trial_model = load_ckpt(trial_model, checkpoint)
                    r = run_episode(trial_model, sc, max_steps, seed=None)
                    r["ablation"] = ab["name"]
                    r["trial"] = t
                    results.append(r)
                    coll="Y" if r["collision"] else "."
                    print(f"  {r['label']:<20} {sc:<15} trial={t:>2d}  coll={coll}  conv={r['conv']:>6.2f}%  ttc={r['min_ttc']:>7.3f}s  wpd={r['avg_wpd']:>7.4f}")
        print()
    return results


def save_results(results):
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rd=Path(PROJECT_ROOT)/"Result"/f"ablation_{ts}";rd.mkdir(parents=True,exist_ok=True)

    # Full CSV
    with open(str(rd/"ablation_full.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["ablation","label","scenario","trial","steps","collision","conv","min_ttc","avg_wpd"])
        w.writeheader();w.writerows(results)

    # Summary by (ablation, label)
    from collections import defaultdict
    g=defaultdict(list)
    for r in results: g[(r["ablation"],r["label"])].append(r)

    rows=[]
    for k,v in sorted(g.items()):
        rows.append({"ablation":k[0],"label":k[1],"n":len(v),"trials":len(set(r.get("trial",0)for r in v)),
                     "coll%":round(sum(r["collision"]for r in v)/len(v)*100,1),
                     "conv%":round(sum(r["conv"]for r in v)/len(v),1),
                     "ttc":round(sum(r["min_ttc"]for r in v)/len(v),3),
                     "wpd":round(sum(r["avg_wpd"]for r in v)/len(v),4)})

    with open(str(rd/"ablation_summary.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["ablation","label","n","trials","coll%","conv%","ttc","wpd"])
        w.writeheader();w.writerows(rows)

    json.dump(results,open(str(rd/"ablation_full.json"),"w"),indent=2)

    # Markdown table for paper
    with open(str(rd/"paper_table.md"),"w") as f:
        f.write("| Ablation | Variant | Coll% | Conv% | MinTTC | AvgWPD |\n|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f'| {r["ablation"]} | {r["label"]} | {r["coll%"]} | {r["conv%"]} | {r["ttc"]} | {r["wpd"]} |\n')

    # Plot
    try:
        import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,6))
        x=range(len(rows));labels=[f"{r['ablation']}\n{r['label']}"for r in rows]
        ax.bar([i-0.2 for i in x],[r["conv%"]for r in rows],0.4,label="Conv%",color="green",alpha=0.7)
        ax.bar([i+0.2 for i in x],[r["ttc"]for r in rows],0.4,label="MinTTC",color="blue",alpha=0.7)
        ax.set_xticks(x);ax.set_xticklabels(labels,fontsize=8);ax.legend();ax.grid(alpha=0.3)
        plt.suptitle("Ablation Study — Rubinstein Bargaining");plt.tight_layout()
        plt.savefig(str(rd/"ablation_plot.png"),dpi=150);plt.close()
    except Exception as e: print(f"[Plot] {e}")

    print(f"Results -> {rd}")
    (Path(PROJECT_ROOT)/"Result"/"latest.txt").write_text(str(rd))


def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "offer_decoder" in ckpt:
        model.offer_decoder.load_state_dict(ckpt["offer_decoder"])
    if "utility_estimator" in ckpt:
        model.utility_estimator.load_state_dict(ckpt["utility_estimator"])
    return model

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=60)
    ap.add_argument("--trials",type=int,default=1,help="random seeds per config/scenario")
    ap.add_argument("--scenarios",nargs="+",default=list(SCENARIOS),choices=list(SCENARIOS))
    ap.add_argument("--checkpoint",type=str,default=None)
    args=ap.parse_args()
    ckpt_note = " [trained]" if args.checkpoint else ""
    print(f"Scenarios: {args.scenarios}, Steps: {args.steps}{ckpt_note}\n")
    r=run_all(scenarios=args.scenarios,max_steps=args.steps,checkpoint=args.checkpoint,trials=args.trials)
    save_results(r)

if __name__=="__main__":
    main()
