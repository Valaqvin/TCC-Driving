# -*- coding: utf-8 -*-
"""
collect_distill_data.py - Collect (state, offer) pairs for LLM distillation.

Runs the 4 kinematic scenarios with the trained bargaining model,
saving state snapshots for LLM annotation.

Output: Result/distill_data/scenes.json (500+ scenes)
"""

import os, sys, json, math, random, datetime
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIOS = {
    "intersection": {"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase": {"a":{"start":[-25,0],"target":[15,0],"speed":6},"b":{"start":[-15,0],"target":[15,0],"speed":10}},
    "merge": {"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on": {"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
}

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
        hdg=torch.tensor(self.hdg,dtype=torch.float32,device=DEVICE)
        wps=torch.stack([torch.from_numpy(self.wps[0]).to(DEVICE),torch.from_numpy(self.wps[1]).to(DEVICE)],dim=0)
        ttc_m=compute_pairwise_ttc(pos,vel,hdg)
        mt=ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,
                "record_len":torch.tensor([2],device=DEVICE),"ttc":mt,"spd":list(self.spd)}


def collect_scenes(num_scenes=500, max_steps=50):
    """Run episodes and collect (state, offer, outcome) for each step."""
    model = RubinsteinBargaining({
        "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
        "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
    }).to(DEVICE)
    model.eval()

    scenes = []
    scenarios = list(SCENARIOS.keys())

    for sid in range(num_scenes):
        sc = random.choice(scenarios)
        sim = SimState(sc)
        scene_data = []

        while not sim.finished and sim.step < max_steps:
            state = sim.get_state()
            feat = torch.randn(2, 256, 48, 48, device=DEVICE)

            with torch.no_grad():
                out = model(spatial_features=feat, waypoints=state["waypoints"],
                           positions=state["positions"], velocities=state["velocities"],
                           headings=state["headings"], record_len=state["record_len"])
                init_offers = out["bargaining_offers"].cpu().numpy().tolist()

            entry = {
                "scene_id": sid, "step": sim.step, "scenario": sc,
                "ttc": state["ttc"],
                "pos_a": sim.pos[0].tolist(), "pos_b": sim.pos[1].tolist(),
                "speed_a": sim.spd[0], "speed_b": sim.spd[1],
                "heading_a": sim.hdg[0], "heading_b": sim.hdg[1],
                "offer_a": init_offers[0], "offer_b": init_offers[1],
                "converged": out["metadata"][0]["converged"],
            }
            scene_data.append(entry)

            for v in range(2):
                dxy = (out["adjusted_waypoints"][v,0]-state["waypoints"][v,0]).cpu().numpy() if v < out["adjusted_waypoints"].shape[0] else -(out["adjusted_waypoints"][0,0]-state["waypoints"][0,0]).cpu().numpy()*0.5
                sim.adjust(v, dxy[0], dxy[1])
            sim.step_sim()

        for entry in scene_data:
            entry["collision"] = sim.collision
            scenes.append(entry)

        if (sid + 1) % 100 == 0:
            print(f"[Data] {sid+1}/{num_scenes} episodes processed, {len(scenes)} scenes collected")

    return scenes


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scenes", type=int, default=500)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    output = args.output or str(Path(PROJECT_ROOT) / "Result" / "distill_data" / "scenes.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"[Data] Collecting {args.num_scenes} scenes from {len(SCENARIOS)} scenarios...")
    scenes = collect_scenes(args.num_scenes)

    json.dump(scenes, open(output, "w"), indent=2)
    print(f"[Data] Saved {len(scenes)} scenes to {output}")


if __name__ == "__main__":
    main()
