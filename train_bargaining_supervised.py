# -*- coding: utf-8 -*-
"""
train_bargaining_supervised.py - Supervised training of the Rubinstein bargaining module.
Uses rule-based ground truth offers to train offer_decoder and utility_estimator.
The rules encode socially optimal behavior: faster vehicle yields in conflict.
"""

import os, sys, json, math, datetime, csv, argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

class ConstantDiscount(nn.Module):
    def __init__(self, delta=0.9):
        super().__init__();self.delta=nn.Parameter(torch.tensor(delta),requires_grad=False)
    def forward(self, ttc):return self.delta.expand_as(ttc)


class HandcraftedUtility(nn.Module):
    def forward(self, state, offer):
        return torch.sigmoid(10.0 * (offer[:, 3] - 0.5 * offer[:, 0]) - 2.0)

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc,
)

SCENARIOS = {
    "intersection": {"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase": {"a":{"start":[-25,0],"target":[15,0],"speed":6},"b":{"start":[-15,0],"target":[15,0],"speed":10}},
    "merge": {"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on": {"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
}

class SimState:
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario]; self.dt=dt; self.name=scenario; self.step=0
        self.collision=False; self.finished=False
        a,b=cfg["a"],cfg["b"]
        self.pos=[np.array(a["start"],np.float32),np.array(b["start"],np.float32)]
        tgt_a=np.array(a["target"],np.float32)-self.pos[0]
        tgt_b=np.array(b["target"],np.float32)-self.pos[1]
        self.spd=[a["speed"],b["speed"]]
        self.hdg=[math.atan2(tgt_a[1],tgt_a[0]),math.atan2(tgt_b[1],tgt_b[0])]
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
        if self.finished: return
        for v in range(2):
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps();self.step+=1
        d=np.linalg.norm(self.pos[0]-self.pos[1])
        if d<1.0:self.collision=True;self.finished=True
        if self.step>200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished=True
    def get_state(self):
        pos=torch.from_numpy(np.array(self.pos)).float()
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],
                          [self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],dtype=torch.float32)
        hdg=torch.tensor(self.hdg,dtype=torch.float32)
        wps=torch.stack([torch.from_numpy(self.wps[0]),torch.from_numpy(self.wps[1])],dim=0)
        ttc_m=compute_pairwise_ttc(pos,vel,hdg)
        mt=ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,
                "record_len":torch.tensor([2]),"ttc":mt,"speeds":[self.spd[0],self.spd[1]]}


def compute_ground_truth(state):
    """
    Rule-based expert: generates socially optimal offers.
    
    Rule: When TTC < threshold, the faster vehicle yields.
    
    Offer space: [speed_factor, lateral_shift, time_headway, yield_willingness]
    Returns: (2,4) tensor, (2,) weights
    """
    ttc = state["ttc"]
    sa, sb = state["speeds"]
    dev = state["positions"].device if hasattr(state["positions"],"device") else "cpu"
    
    if ttc < 3.0:
        if sa > sb:
            gt = [[0.15, 0.35, 0.80, 0.85], [0.75, 0.10, 0.25, 0.15]]
        else:
            gt = [[0.75, 0.10, 0.25, 0.15], [0.15, 0.35, 0.80, 0.85]]
        w = [2.0, 2.0]
    elif ttc < 6.0:
        ra = sb / (sa + sb + 1e-6)
        rb = sa / (sa + sb + 1e-6)
        gt = [[0.3+0.4*ra, 0.25, 0.6, 0.5+0.3*ra],
              [0.3+0.4*rb, 0.25, 0.6, 0.5+0.3*rb]]
        w = [1.5, 1.5]
    elif ttc < 10.0:
        gt = [[0.45, 0.15, 0.55, 0.55], [0.45, 0.15, 0.55, 0.55]]
        w = [1.0, 1.0]
    else:
        gt = [[0.50, 0.10, 0.50, 0.50], [0.50, 0.10, 0.50, 0.50]]
        w = [0.5, 0.5]
    
    return torch.tensor(gt, dtype=torch.float32, device=dev), torch.tensor(w, device=dev)


def collect_episode(model, scenario, max_steps, device):
    """Run episode and collect (state_feature -> gt_offer) pairs."""
    sim = SimState(scenario)
    feats, gts = [], []
    while not sim.finished and sim.step < max_steps:
        state = sim.get_state()
        state["positions"] = state["positions"].to(device)
        state["velocities"] = state["velocities"].to(device)
        # Synthetic BEV features (2 agents, 256 channels, 48x48)
        feat = torch.randn(2, 256, 48, 48, device=device)
        gt, _ = compute_ground_truth(state)
        feats.append(feat.cpu()); gts.append(gt.cpu())
        sim.adjust(0, 0, 0); sim.adjust(1, 0, 0)
        sim.step_sim()
    return torch.stack(feats), torch.stack(gts)


class BargainingTrainer:
    def __init__(self, device=None, lr=1e-3, max_rounds=3, discount_type="ttc", discount_alpha=0.3, utility_type="learned"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        
        self.model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":max_rounds,
            "discount_alpha":discount_alpha,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(self.device)
        if discount_type == "constant":
            self.model.discount = ConstantDiscount(delta=0.9).to(self.device)
        if utility_type == "handcrafted":
            self.model.utility_estimator = HandcraftedUtility().to(self.device)
        self.model.train()
        
        # Freeze state_encoder, only train decoder + estimator
        for p in self.model.state_encoder.parameters(): p.requires_grad = False
        
        util_params = [] if utility_type == "handcrafted" else [{"params": self.model.utility_estimator.parameters(), "lr": lr}]
        self.opt = optim.Adam([
            {"params": self.model.offer_decoder.parameters(), "lr": lr},
        ] + util_params)
        self.mse = nn.MSELoss()
        self.margin = nn.MarginRankingLoss(margin=0.5)
        
        self.log = {"epoch":[],"offer_loss":[],"util_loss":[],"total":[]}
        self.best_loss = float("inf")
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{max_rounds}r_{discount_type}_{utility_type}_a{discount_alpha}"
        self.res_dir = Path(PROJECT_ROOT) / "Result" / f"train_{label}_{ts}"
        (self.res_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.res_dir / "plots").mkdir(exist_ok=True)
        print(f"[Train] Results -> {self.res_dir}")
    
    def step(self, features, gt_offers):
        """Single training step."""
        features = features.to(self.device)
        gt_offers = gt_offers.to(self.device)
        
        # Flatten: (T,2,C,H,W) -> (T*2,C,H,W)
        B = features.shape[0] * features.shape[1]
        feat_flat = features.view(B, 256, 48, 48)
        gt_flat = gt_offers.view(B, 4)
        
        states = self.model.state_encoder(feat_flat)  # (B, 64)
        pred = self.model.offer_decoder(states)        # (B, 4)
        offer_loss = self.mse(pred, gt_flat)
        
        # Utility ranking loss: good offers > bad offers
        bad = 1.0 - gt_flat
        u_good = self.model.utility_estimator(states, gt_flat).squeeze(-1)
        u_bad = self.model.utility_estimator(states, bad).squeeze(-1)
        util_loss = self.margin(u_good, u_bad, torch.ones_like(u_good))
        
        total = offer_loss + 0.5 * util_loss
        
        self.opt.zero_grad(); total.backward()
        nn.utils.clip_grad_norm_(
            list(self.model.offer_decoder.parameters()) +
            list(self.model.utility_estimator.parameters()), 1.0)
        self.opt.step()
        
        return {"offer_loss":offer_loss.item(),"util_loss":util_loss.item(),"total":total.item()}
    
    def train(self, n_epochs=50, ep_per_epoch=5, scenarios=None):
        if scenarios is None: scenarios = ["intersection","merge"]
        print(f"[Train] {n_epochs} epochs, {ep_per_epoch} ep/epoch, scenarios={scenarios}")
        
        for ep in range(n_epochs):
            losses = []
            for _ in range(ep_per_epoch):
                sc = np.random.choice(scenarios)
                feats, gts = collect_episode(self.model, sc, 40, self.device)
                losses.append(self.step(feats, gts))
            
            avg = {k: np.mean([l[k] for l in losses]) for k in losses[0]}
            self.log["epoch"].append(ep)
            [self.log[k].append(avg[k]) for k in ["offer_loss","util_loss","total"]]
            
            if ep % max(1, n_epochs//10) == 0 or ep == n_epochs-1:
                print(f"  Ep {ep:>3d} | Offer:{avg['offer_loss']:.6f} | Util:{avg['util_loss']:.6f} | Total:{avg['total']:.6f}")
            
            if avg["total"] < self.best_loss:
                self.best_loss = avg["total"]
                torch.save({
                    "offer_decoder": self.model.offer_decoder.state_dict(),
                    "utility_estimator": self.model.utility_estimator.state_dict(),
                    "opt": self.opt.state_dict(),
                    "best_loss": self.best_loss,
                }, str(self.res_dir / "checkpoints" / "best_model.pth"))
        
        torch.save(self.model.offer_decoder.state_dict(),
                   str(self.res_dir / "checkpoints" / "offer_decoder_final.pth"))
        torch.save(self.model.utility_estimator.state_dict(),
                   str(self.res_dir / "checkpoints" / "utility_estimator_final.pth"))
        
        self._save_log_csv()
        self._save_config()
        print(f"[Train] Done. Best loss: {self.best_loss:.6f}")
    
    def _save_log_csv(self):
        import csv
        p = str(self.res_dir / "training_log.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.log.keys())
            w.writeheader()
            for i in range(len(self.log["epoch"])):
                w.writerow({k: self.log[k][i] for k in self.log})
        print(f"[Log] Training log -> {p}")

    def _save_config(self):
        cfg = {"learning_rate": self.lr, "device": str(self.device),
               "state_dim": 64, "offer_dim": 4, "loss_offer_weight": 1.0,
               "loss_utility_weight": 0.5,
               "rule": "Faster vehicle yields when TTC below 3 seconds."}
        import json
        json.dump(cfg, open(str(self.res_dir / "config.json"), "w"), indent=2)
        print(f"[Config] Saved -> {self.res_dir / 'config.json'}")

    def _plot_curve(self):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, (a1,a2) = plt.subplots(1,2,figsize=(12,4))
            e = self.log["epoch"]
            a1.plot(e,self.log["offer_loss"],"b-",label="Offer")
            a1.plot(e,self.log["util_loss"],"r-",label="Utility")
            a1.set_xlabel("Epoch");a1.set_ylabel("Loss");a1.legend();a1.grid(alpha=0.3)
            a2.plot(e,self.log["total"],"g-",label="Total",lw=2)
            a2.set_xlabel("Epoch");a2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(str(self.res_dir/"plots"/"training_curve.png"),dpi=150);plt.close()
        except Exception as ex: print(f"[Plot] {ex}")
    
    def evaluate(self, scenario="intersection", max_steps=60):
        """Compare trained vs untrained model."""
        base = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(self.device)
        base.eval(); self.model.eval()
        
        def run(mod, label):
            sim = SimState(scenario); conv=0; steps=0
            wpds=[]; ttc_log=[]
            while not sim.finished and sim.step < max_steps:
                s = sim.get_state()
                pos=s["positions"].to(self.device);vel=s["velocities"].to(self.device)
                hdg=s["headings"].to(self.device);wps=s["waypoints"].to(self.device)
                feat=torch.randn(2,256,48,48,device=self.device)
                with torch.no_grad():
                    out=mod(spatial_features=feat,waypoints=wps,positions=pos,velocities=vel,headings=hdg,
                            record_len=torch.tensor([2],device=self.device))
                wpd=(out["adjusted_waypoints"]-wps).norm(dim=-1).mean().item()
                wpds.append(wpd); ttc_log.append(s["ttc"])
                if out["metadata"][0]["converged"]: conv+=1
                steps+=1
                for v in range(2):
                    if v<out["adjusted_waypoints"].shape[0]:
                        dxy=(out["adjusted_waypoints"][v,0]-wps[v,0]).cpu().numpy()
                    else:
                        dxy=-(out["adjusted_waypoints"][0,0]-wps[0,0]).cpu().numpy()*0.5
                    sim.adjust(v,dxy[0],dxy[1])
                sim.step_sim()
            return {"label":label,"steps":steps,"collision":sim.collision,
                    "conv":conv/max(1,steps)*100,"minttc":min(ttc_log)if ttc_log else 0,
                    "avgwpd":np.mean(wpds)if wpds else 0}
        
        bl=run(base,"Baseline");tr=run(self.model,"Trained")
        eval_res = {"eval_scenario": scenario,
                    "baseline": {k: bl[k] for k in bl if k != "label"},
                    "trained": {k: tr[k] for k in tr if k != "label"}}
        import json
        json.dump(eval_res, open(str(self.res_dir / "evaluation.json"), "w"), indent=2)
        print(f"[Eval] Results -> {self.res_dir / 'evaluation.json'}")
        print(f"\n{'Metric':<20}{'Baseline':>12}{'Trained':>12}")
        print("-"*44)
        for k in ["steps","collision","conv","minttc","avgwpd"]:
            v1,v2=bl[k],tr[k]
            if isinstance(v1, float): s1 = f"{v1:.4f}"
            else: s1 = str(v1)
            if isinstance(v2, float): s2 = f"{v2:.4f}"
            else: s2 = str(v2)
            print(f"  {k:<20}{s1:>12}{s2:>12}")
        
        # Comparison plot
        try:
            import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(10,5))
            res=[bl,tr]
            for r in res:
                c="blue" if "Base" in r["label"] else "green"
                ls="--" if "Base" in r["label"] else "-"
                ax.bar(r["label"],[r["conv"],abs(r["avgwpd"])*20],color=c,alpha=0.5)
            ax.set_ylabel("Conv Rate (%) / Scaled WP Delta");ax.set_title(f"Eval: {scenario}")
            plt.tight_layout()
            plt.savefig(str(self.res_dir/"plots"/"eval_comparison.png"),dpi=150);plt.close()
        except: pass
        return bl, tr


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--scenarios",nargs="+",default=["intersection","merge"],choices=list(SCENARIOS))
    ap.add_argument("--max-rounds",type=int,default=3)
    ap.add_argument("--discount-type",type=str,default="ttc",choices=["ttc","constant"])
    ap.add_argument("--discount-alpha",type=float,default=0.3)
    ap.add_argument("--utility-type",type=str,default="learned",choices=["learned","handcrafted"])
    ap.add_argument("--eval-scenario",default="intersection",choices=list(SCENARIOS))
    ap.add_argument("--eval",action="store_true",help="Skip train, only eval from checkpoint")
    args=ap.parse_args()
    
    t=BargainingTrainer(lr=args.lr,max_rounds=args.max_rounds,discount_type=args.discount_type,discount_alpha=args.discount_alpha,utility_type=args.utility_type)
    if not args.eval:
        t.train(n_epochs=args.epochs,ep_per_epoch=5,scenarios=args.scenarios)
    all_evals=[]
    for sc in list(SCENARIOS.keys()):
        r=t.evaluate(scenario=sc,max_steps=60)
        all_evals.append(r)
    import json
    json.dump(all_evals,open(str(Path(t.res_dir)/"all_evals.json"),"w"),indent=2)
    print(f"\nDone. Results: {t.res_dir}")

if __name__=="__main__":
    main()
