# -*- coding: utf-8 -*-
"""
train_bargaining_ppo.py - PPO training with curriculum learning and entropy scheduling.

Improvements over REINFORCE training (train_bargaining_rl.py):
  1. PPO (clipped surrogate objective) - much more stable updates
  2. GAE (Generalized Advantage Estimation) - better advantage estimates
  3. Curriculum learning - progressive scenario difficulty
  4. Adaptive entropy scheduling - decays exploration over time

Usage:
    python train_bargaining_ppo.py --epochs=200
    python train_bargaining_ppo.py --curriculum
    python train_bargaining_ppo.py --resume <policy.pth>

Output:
    Result/ppo_train_YYYYMMDD_HHMMSS/
"""

import os, sys, json, math, datetime, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curriculum: (scenario, epochs) or None = all
CURRICULUM = [
    ("intersection", 40),   # easy, longer for stable base
    ("chase", 20),          # medium
    ("merge", 20),          # medium
    ("head_on", 40),        # hard, shorter with smoother transition
    (None, 80),            # all mixed, same total
]

SCENARIOS = {
    "intersection":{"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase":{"a":{"start":[-25,0],"target":[15,0],"speed":10},"b":{"start":[-15,0],"target":[15,0],"speed":6}},
    "merge":{"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on":{"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
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


class StochasticPolicy(nn.Module):
    def __init__(self, state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=-1.0):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, offer_dim),
        )
        self.log_std = nn.Parameter(torch.full((offer_dim,), init_log_std))
    def forward(self, state, deterministic=False):
        mean = self.mean_net(state)
        if deterministic:
            return torch.sigmoid(mean), None
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        action_logit = dist.rsample()
        action = torch.sigmoid(action_logit)
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        return action, log_prob
    def get_log_prob(self, state, action):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        action_logit = torch.log(action / (1 - action + 1e-8) + 1e-8)
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        return log_prob
    def get_entropy(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        return dist.entropy().sum(dim=-1).mean()


def compute_step_reward(converged, speed_a, speed_b, n_rounds, dist, ttc):
    r = 0.0
    if converged:
        r += 1.0
    if ttc < 3.0 and not converged:
        r -= 2.0
    if ttc < 1.5 and not converged:
        r -= 3.0
    avg_speed = (speed_a + speed_b) / 2.0
    r += 0.02 * min(avg_speed / 10.0, 1.0)
    r += 0.02 * max(0, 1.0 - dist / 50.0)
    r -= 0.1 * n_rounds / 3.0
    return r



class PPOTrainer:
    def __init__(self, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=10,
                 entropy_init=0.1, entropy_min=0.001,
                 use_curriculum=True):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.entropy_init = entropy_init
        self.entropy_min = entropy_min
        self.use_curriculum = use_curriculum
        self.transition_window = 15  # smooth transition epochs between stages
        self.epoch = 0
        self.total_epochs = 1

        # Model
        self.model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(DEVICE)
        self.model.train()
        for p in self.model.state_encoder.parameters():
            p.requires_grad = False

        # Policy + Value
        self.policy = StochasticPolicy(
            state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=0.0
        ).to(DEVICE)
        self.value_net = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1),
        ).to(DEVICE)

        # Optimizer
        self.optimizer = optim.Adam([
            {"params": self.policy.parameters(), "lr": lr},
            {"params": self.value_net.parameters(), "lr": lr},
        ])

        # Logging
        self.log = {"epoch":[],"reward":[],"collision_pct":[],"policy_loss":[],"value_loss":[],"entropy":[],"kl":[]}
        self.best_reward = -float("inf")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.res_dir = Path(PROJECT_ROOT) / "Result" / f"ppo_train_{ts}"
        (self.res_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        print(f"[PPO] Results -> {self.res_dir}")

    def collect_episode(self, scenario, max_steps=50):
        sim = SimState(scenario)
        trajectory = []

        while not sim.finished and sim.step < max_steps:
            state = sim.get_state()
            feat = torch.randn(2, 256, 48, 48, device=DEVICE)
            with torch.no_grad():
                state_enc = self.model.state_encoder(feat)

            init_offers, log_prob = self.policy(state_enc)
            pos = state["positions"]; vel = state["velocities"]
            hdg = state["headings"]; wps = state["waypoints"]

            with torch.no_grad():
                feat2 = torch.randn(2, 256, 48, 48, device=DEVICE)
                out = self.model(spatial_features=feat2, waypoints=wps,
                                 positions=pos, velocities=vel, headings=hdg,
                                 record_len=torch.tensor([2], device=DEVICE))
                offer_to_wp = self.model.offer_to_waypoint(init_offers)
                offset = offer_to_wp * self.model.waypoint_offset_scale
                adj_wps = wps.clone()
                adj_wps[0, 0, 0] += offset[0, 0]; adj_wps[0, 0, 1] += offset[0, 1]
                adj_wps[0, 1, 0] += offset[0, 2]; adj_wps[0, 1, 1] += offset[0, 3]

            for v in range(2):
                if v < adj_wps.shape[0]:
                    dxy = (adj_wps[v,0] - wps[v,0]).cpu().numpy()
                else:
                    dxy = -(adj_wps[0,0] - wps[0,0]).cpu().numpy() * 0.5
                sim.adjust(v, dxy[0], dxy[1])

            dist = np.linalg.norm(sim.pos[0] - sim.pos[1])
            conv = out["metadata"][0]["converged"] if "metadata" in out else False
            reward = compute_step_reward(
                converged=conv, speed_a=sim.spd[0], speed_b=sim.spd[1],
                n_rounds=out["metadata"][0]["n_rounds"] if "metadata" in out else 0,
                dist=dist, ttc=state["ttc"])

            with torch.no_grad():
                value = self.value_net(state_enc[0:1]).item()

            trajectory.append({
                "state_enc": state_enc.detach().cpu(),
                "action": init_offers.detach().cpu(),
                "log_prob": log_prob.detach().cpu(),
                "reward": reward,
                "value": value,
                "done": sim.finished or sim.step >= max_steps - 1,
            })
            sim.step_sim()

        # Terminal reward
        final = -10.0 if sim.collision else 1.0
        if trajectory:
            trajectory[-1]["reward"] += final
        return trajectory, sim

    def process_trajectories(self, trajectories):
        all_s, all_a, all_lp, all_r, all_v, all_d = [],[],[],[],[],[]
        for traj in trajectories:
            for t in traj:
                all_s.append(t["state_enc"][0:1]); all_a.append(t["action"][0:1])
                all_lp.append(t["log_prob"][0:1]); all_r.append(t["reward"])
                all_v.append(t.get("value",0)); all_d.append(1 if t.get("done",False) else 0)
        if not all_s:
            return None, None, None, None, None
        s = torch.cat(all_s).to(DEVICE)
        a = torch.cat(all_a).to(DEVICE)
        lp = torch.cat(all_lp).to(DEVICE).squeeze()
        r = torch.tensor(all_r, dtype=torch.float32, device=DEVICE)
        v = torch.tensor(all_v, dtype=torch.float32, device=DEVICE)
        d = torch.tensor(all_d, dtype=torch.float32, device=DEVICE)
        adv = self.compute_gae(r, v, d)
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return s, a, lp, ret, adv

    def compute_gae(self, rewards, values, dones):
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        return adv

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        if states is None: return {}
        total_policy = 0.0; total_value = 0.0; total_entropy = 0.0; total_kl = 0.0
        for _ in range(self.ppo_epochs):
            lp = self.policy.get_log_prob(states, actions)
            entropy = self.policy.get_entropy(states)
            ratio = torch.exp(lp - old_log_probs.detach())
            obj1 = ratio * advantages
            obj2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            p_loss = -torch.min(obj1, obj2).mean()
            vals = self.value_net(states).squeeze(-1)
            v_loss = F.mse_loss(vals, returns)
            ec = max(self.entropy_min, self.entropy_init * (1 - self.epoch / self.total_epochs))
            loss = p_loss + 0.5 * v_loss - ec * entropy
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value_net.parameters()), 0.5)
            self.optimizer.step()
            total_policy += p_loss.item(); total_value += v_loss.item()
            total_entropy += entropy.item()
            with torch.no_grad():
                kl = (old_log_probs - lp).mean().item()
                total_kl += kl
                if kl > 0.02:
                    break
        n = _ + 1
        return {"policy_loss": total_policy / n, "value_loss": total_value / n,
                "entropy": total_entropy / n, "kl": total_kl / n}

    def train(self, n_epochs=200, ep_per_epoch=5):
        self.total_epochs = n_epochs
        print(f"[PPO] Training: {n_epochs} epochs, {ep_per_epoch} ep/epoch")
        print(f"  Entropy: {self.entropy_init} -> {self.entropy_min}, Curriculum: {self.use_curriculum}")
        hdr = "  {:>5} | {:>8} | {:>6} | {:>6} | {:>9} | {:>8} | {:>6}"
        print(hdr.format("Ep", "Reward", "Coll%", "Conv%", "P_Loss", "Entropy", "KL"))
        print("  " + "-" * 60)

        curriculum_stage = 0
        stage_epoch = 0

        for ep in range(n_epochs):
            self.epoch = ep

            # Curriculum learning
            if self.use_curriculum:
                stage_epoch += 1
                if curriculum_stage < len(CURRICULUM) and stage_epoch > CURRICULUM[curriculum_stage][1]:
                    curriculum_stage += 1
                    stage_epoch = 0
                curr_scenario = CURRICULUM[min(curriculum_stage, len(CURRICULUM)-1)][0]
                scenarios = [curr_scenario] if curr_scenario else list(SCENARIOS)
                weights = None
                if curriculum_stage > 0 and stage_epoch < self.transition_window:
                    prev_scenario = CURRICULUM[curriculum_stage - 1][0]
                    mix = stage_epoch / self.transition_window
                    if prev_scenario and curr_scenario:
                        scenarios = [prev_scenario, curr_scenario]
                        weights = [1.0 - mix, mix]
            else:
                scenarios = list(SCENARIOS)
                weights = None

            trajectories = []
            ep_collisions = []
            for _ in range(ep_per_epoch):
                sc = np.random.choice(scenarios, p=weights) if weights else np.random.choice(scenarios)
                traj, sim = self.collect_episode(sc, 50)
                trajectories.append(traj)
                ep_collisions.append(sim.collision)

            states, actions, log_probs, returns, adv = self.process_trajectories(trajectories)
            stats = self.ppo_update(states, actions, log_probs, returns, adv)
            total_r = sum(sum(t["reward"] for t in traj) for traj in trajectories) / len(trajectories)
            avg_coll = np.mean(ep_collisions) * 100
            avg_conv = sum(1 for t in trajectories if not t[-1].get("done", True)) / len(trajectories) * 100

            self.log["epoch"].append(ep); self.log["reward"].append(total_r)
            self.log["collision_pct"].append(avg_coll)
            self.log["policy_loss"].append(stats.get("policy_loss",0))
            self.log["value_loss"].append(stats.get("value_loss",0))
            self.log["entropy"].append(stats.get("entropy",0))
            self.log["kl"].append(stats.get("kl",0))

            if ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1:
                print(hdr.format(ep, f"{total_r:.3f}", f"{avg_coll:.1f}", f"{avg_conv:.1f}",
                      f"{stats.get('policy_loss',0):.5f}", f"{stats.get('entropy',0):.5f}",
                      f"{stats.get('kl',0):.5f}"))

            if total_r > self.best_reward:
                self.best_reward = total_r
                torch.save(self.policy.state_dict(),
                           str(self.res_dir / "checkpoints" / "best_policy.pth"))
                torch.save({
                    "epoch": ep,
                    "best_reward": self.best_reward,
                    "policy": self.policy.state_dict(),
                    "value_net": self.value_net.state_dict(),
                    "model": self.model.state_dict(),
                    "scenarios": SCENARIOS,
                    "curriculum": CURRICULUM,
                }, str(self.res_dir / "checkpoints" / "best_checkpoint_full.pth"))

        torch.save(self.policy.state_dict(), str(self.res_dir / "checkpoints" / "final_policy.pth"))
        torch.save({
            "epoch": n_epochs - 1,
            "best_reward": self.best_reward,
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "model": self.model.state_dict(),
            "scenarios": SCENARIOS,
            "curriculum": CURRICULUM,
        }, str(self.res_dir / "checkpoints" / "final_checkpoint_full.pth"))
        self._save_log()
        self._plot_curves()
        print(f"[PPO] Done. Best avg reward: {self.best_reward:.3f}")

    def _save_log(self):
        with open(str(self.res_dir / "ppo_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.log.keys()))
            w.writeheader()
            for i in range(len(self.log["epoch"])):
                w.writerow({k: self.log[k][i] for k in self.log})
        json.dump({k: v for k, v in self.log.items()},
                  open(str(self.res_dir / "ppo_log.json"), "w"))
        print(f"[Log] Saved to {self.res_dir}")

    def _plot_curves(self):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            e = self.log["epoch"]
            d = [("reward","Reward","blue"),("collision_pct","Collision %","red"),
                 ("policy_loss","Policy Loss","green"),("value_loss","Value Loss","orange"),
                 ("entropy","Entropy","purple"),("kl","KL Div","brown")]
            for ax, (k, lbl, c) in zip(axes.flat, d):
                ax.plot(e, self.log[k], color=c); ax.set_xlabel("Epoch")
                ax.set_ylabel(lbl); ax.grid(alpha=0.3)
            plt.suptitle("PPO Training Curves"); plt.tight_layout()
            plt.savefig(str(self.res_dir / "ppo_curves.png"), dpi=150); plt.close()
            print(f"[Plot] -> {self.res_dir / 'ppo_curves.png'}")
        except Exception as ex: print(f"[Plot] {ex}")

    def evaluate(self, max_steps=60):
        self.policy.eval()
        model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(DEVICE)
        model.eval()
        class _OfferDec(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return torch.sigmoid(self.m(x))
        model.offer_decoder = _OfferDec(self.policy.mean_net)
        model.discount = self.model.discount
        print("[Eval] PPO policy across scenarios:")
        for sc in list(SCENARIOS):
            sim = SimState(sc); conv=0; steps=0; ttcs=[]; wpds=[]
            while not sim.finished and sim.step < max_steps:
                s = sim.get_state(); feat = torch.randn(2,256,48,48,device=DEVICE)
                with torch.no_grad():
                    out = model(spatial_features=feat, waypoints=s["waypoints"],
                               positions=s["positions"], velocities=s["velocities"],
                               headings=s["headings"], record_len=s["record_len"])
                ttcs.append(s["ttc"]); wpds.append((out["adjusted_waypoints"]-s["waypoints"]).norm(dim=-1).mean().item())
                if out["metadata"][0]["converged"]: conv+=1; steps+=1
                for v in range(2):
                    dxy = (out["adjusted_waypoints"][v,0]-s["waypoints"][v,0]).cpu().numpy() if v < out["adjusted_waypoints"].shape[0] else -(out["adjusted_waypoints"][0,0]-s["waypoints"][0,0]).cpu().numpy()*0.5
                    sim.adjust(v, dxy[0], dxy[1])
                sim.step_sim()
            c = "Y" if sim.collision else "."; n = max(1, steps)
            print(f"  {sc:<15} steps={steps:>3d} coll={c} conv={round(conv/n*100,1):>6.1f}% ttc={round(min(ttcs) if ttcs else 0,3):>6.3f}s wpd={round(np.mean(wpds) if wpds else 0,4):.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="PPO training for bargaining")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    args = ap.parse_args()
    print("=" * 60); print("  PPO: Stable Self-Play Bargaining Training")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Curriculum: {not args.no_curriculum}")
    print("=" * 60); print()
    trainer = PPOTrainer(lr=args.lr, use_curriculum=not args.no_curriculum)
    trainer.train(n_epochs=args.epochs, ep_per_epoch=5)
    trainer.evaluate()
    print()
    print(f"Done. Results: {trainer.res_dir}")

if __name__ == "__main__":
    main()
