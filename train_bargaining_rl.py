# -*- coding: utf-8 -*-
"""
train_bargaining_rl.py - Self-play REINFORCE training for the Rubinstein bargaining module.

The offer_decoder acts as a stochastic policy pi(a|s) with learnable log_std.
The alternating_offers_protocol + kinematic simulation form the environment.

Reward design (multi-objective):
  R = w_conv * conv + w_coll * coll + w_eff * efficiency + w_speed * speed_penalty

Usage:
    python train_bargaining_rl.py --epochs=50
    python train_bargaining_rl.py --resume <ckpt_path>

Output:
    Result/rl_train_YYYYMMDD_HHMMSS/  with logs and model checkpoints
"""

import os, sys, json, math, datetime, csv
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import accumulate
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining, compute_pairwise_ttc, alternating_offers_protocol,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIOS = {
    "intersection": {
        "a": {"start":[-30,-2],"target":[0,0],"speed":8},
        "b": {"start":[-2,-30],"target":[0,0],"speed":8},
        "desc": "Perpendicular crossing at an intersection.",
    },
    "chase": {
        "a": {"start":[-25,0],"target":[15,0],"speed":6},
        "b": {"start":[-15,0],"target":[15,0],"speed":10},
        "desc": "Faster vehicle catching up from behind.",
    },
    "merge": {
        "a": {"start":[-25,-3],"target":[5,0],"speed":8},
        "b": {"start":[-30,0],"target":[10,0],"speed":7},
        "desc": "Lane merging from a side lane.",
    },
    "head_on": {
        "a": {"start":[-20,0],"target":[10,0],"speed":8},
        "b": {"start":[20,0],"target":[-10,0],"speed":8},
        "desc": "Head-on approach.",
    },
}


class SimState:
    """Kinematic simulation of two vehicles approaching a conflict point."""
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario]; self.dt=dt; self.name=scenario; self.step=0
        self.collision=False; self.finished=False
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
        if self.finished: return
        for v in range(2):
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),
                                   self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps(); self.step+=1
        d=np.linalg.norm(self.pos[0]-self.pos[1])
        if d<1.0: self.collision=True; self.finished=True
        if self.step>200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished=True
    def get_state(self):
        pos=torch.from_numpy(np.array(self.pos)).float().to(DEVICE)
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],
                          [self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],
                         dtype=torch.float32,device=DEVICE)
        hdg=torch.tensor(self.hdg,dtype=torch.float32,device=DEVICE)
        wps=torch.stack([torch.from_numpy(self.wps[0]).to(DEVICE),
                         torch.from_numpy(self.wps[1]).to(DEVICE)],dim=0)
        ttc_m=compute_pairwise_ttc(pos,vel,hdg)
        mt=ttc_m[ttc_m.isfinite()].min().item() if ttc_m.isfinite().any() else 100.0
        return {"positions":pos,"velocities":vel,"headings":hdg,"waypoints":wps,
                "record_len":torch.tensor([2],device=DEVICE),"ttc":mt,"spd":list(self.spd)}



# ── RL Policy Components ──

class StochasticPolicy(nn.Module):
    """
    Stochastic policy wrapping the offer_decoder with learnable exploration noise.
    
    The policy outputs a diagonal Gaussian over the 4D offer space (logit-space).
    Sigmoid is applied to the sampled action to produce bounded offers in [0,1].
    
    pi(a|s) = Normal(mean=decoder(s), std=exp(log_std))
    """
    def __init__(self, state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=0.0):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, offer_dim),
        )
        self.log_std = nn.Parameter(torch.full((offer_dim,), init_log_std))
    
    def forward(self, state, deterministic=False):
        """
        Args:
            state: (B, state_dim) bargaining state embeddings
            deterministic: if True, return mean action without noise
        Returns:
            action: (B, offer_dim) sigmoid-bounded offers in [0,1]
            log_prob: (B,) log probability of sampled action (or None if deterministic)
        """
        mean = self.mean_net(state)
        if deterministic:
            return torch.sigmoid(mean), None
        
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        action_logit = dist.rsample()  # reparameterized sample
        action = torch.sigmoid(action_logit)
        
        # Log probability with change of variables for sigmoid
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        # Jacobian correction for sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        
        return action, log_prob
    
    def get_log_prob(self, state, action):
        """Compute log pi(action | state) for a given action."""
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        # Inverse sigmoid to get logit
        action_logit = torch.log(action / (1 - action + 1e-8) + 1e-8)
        log_prob = dist.log_prob(action_logit).sum(dim=-1)
        log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
        return log_prob


class ValueBaseline(nn.Module):
    """
    State-value estimator V(s) for advantage computation in REINFORCE.
    Trained with MSE loss against Monte Carlo returns.
    """
    def __init__(self, state_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        """(B, state_dim) -> (B, 1)"""
        return self.net(state)


def compute_return(rewards, gamma=0.95):
    """Compute discounted cumulative returns from a list of rewards."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_step_reward(converged, collision, speed_a, speed_b, n_rounds, dist):
    """
    Multi-objective reward for one simulation step.
    
    Returns a scalar reward in roughly [-1, 1].
    """
    r = 0.0
    
    # Agreement reward (sparse, only when convergence happens)
    if converged:
        r += 1.0
    
    # Collision penalty (terminal)
    if collision:
        r -= 2.0
    
    # Efficiency reward (dense)
    avg_speed = (speed_a + speed_b) / 2.0
    r += 0.05 * min(avg_speed / 10.0, 1.0)
    
    # Distance-based progress reward (dense)
    r += 0.01 * min(dist / 50.0, 1.0)
    
    # Round penalty (dense, small)
    r -= 0.05 * n_rounds / 3.0
    
    return r



class REINFORCETrainer:
    """
    Self-play REINFORCE training for the bargaining module.
    
    Algorithm:
      1. Sample initial offers from stochastic policy pi(a|s)
      2. Run bargaining protocol (deterministic, part of environment)
      3. Step kinematic simulation, get reward
      4. Store (state, action, reward) trajectory
      5. At episode end: compute returns, advantages, update policy
    
    Value baseline: separate V(s) network trained with MSE.
    """
    def __init__(self, lr=3e-4, gamma=0.95, entropy_coef=0.05):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Base bargaining model (provides state encoder and utility estimator)
        self.model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(DEVICE)
        self.model.train()
        
        # Freeze state_encoder (transfer from pretrained or keep fixed)
        for p in self.model.state_encoder.parameters():
            p.requires_grad = False
        
        # Stochastic policy (replaces the original offer_decoder)
        self.policy = StochasticPolicy(
            state_dim=64, offer_dim=4, hidden_dim=128, init_log_std=0.0
        ).to(DEVICE)
        
        # Value baseline
        self.value_net = ValueBaseline(state_dim=64, hidden_dim=128).to(DEVICE)
        
        # Optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Logging
        self.log = defaultdict(list)
        self.best_reward = -float('inf')
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.res_dir = Path(PROJECT_ROOT) / "Result" / f"rl_train_{ts}"
        (self.res_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        print(f"[RL] Results -> {self.res_dir}")
    
    def collect_episode(self, scenario, max_steps=50):
        """Run one episode, return trajectory of (state, action, reward, info)."""
        sim = SimState(scenario)
        trajectory = []
        
        while not sim.finished and sim.step < max_steps:
            state = sim.get_state()
            
            # Get bargaining state embedding
            with torch.no_grad():
                feat = torch.randn(2, 256, 48, 48, device=DEVICE)
                state_enc = self.model.state_encoder(feat)  # (2, 64)
            
            # Sample initial offers from policy
            init_offers, log_prob = self.policy(state_enc)  # (2, 4), (2,)
            
            # Run bargaining protocol with sampled offers
            with torch.no_grad():
                pos = state["positions"]; vel = state["velocities"]
                hdg = state["headings"]; wps = state["waypoints"]
                feat_in = torch.randn(2, 256, 48, 48, device=DEVICE)
                
                # Get the bargaining output
                out = self.model(
                    spatial_features=feat_in, waypoints=wps,
                    positions=pos, velocities=vel, headings=hdg,
                    record_len=torch.tensor([2], device=DEVICE),
                )
                
                # But we want to use our SAMPLED offers, not the model's default.
                # So we replace the model's offers with our policy's offers.
                # The waypoint adjustment comes from our policy's offers.
                offer_to_wp = self.model.offer_to_waypoint(init_offers)
                offset = offer_to_wp * self.model.waypoint_offset_scale
                
                adj_wps = wps.clone()
                adj_wps[0, 0, 0] += offset[0, 0]
                adj_wps[0, 0, 1] += offset[0, 1]
                adj_wps[0, 1, 0] += offset[0, 2]
                adj_wps[0, 1, 1] += offset[0, 3]
            
            # Apply adjustment to simulation
            for v in range(2):
                if v < adj_wps.shape[0]:
                    dxy = (adj_wps[v,0] - wps[v,0]).cpu().numpy()
                else:
                    dxy = -(adj_wps[0,0] - wps[0,0]).cpu().numpy() * 0.5
                sim.adjust(v, dxy[0], dxy[1])
            
            # Compute per-step reward BEFORE stepping (based on current state)
            dist = np.linalg.norm(sim.pos[0] - sim.pos[1])
            conv = out["metadata"][0]["converged"] if "metadata" in out else False
            reward = compute_step_reward(
                converged=conv,
                collision=False,  # evaluated before step
                speed_a=sim.spd[0], speed_b=sim.spd[1],
                n_rounds=out["metadata"][0]["n_rounds"] if "metadata" in out else 0,
                dist=dist,
            )
            
            trajectory.append({
                "state_enc": state_enc.detach().cpu(),
                "action": init_offers.detach().cpu(),
                "log_prob": log_prob.detach().cpu(),
                "reward": reward,
            })
            
            sim.step_sim()
        
        # Terminal reward
        final_reward = -10.0 if sim.collision else 1.0
        if trajectory:
            trajectory[-1]["reward"] += final_reward
        
        return trajectory, sim
    
    def update(self, trajectory):
        """REINFORCE update with baseline and entropy bonus."""
        if not trajectory:
            return {}
        
        B = len(trajectory)
        device = DEVICE
        
        # Stack trajectory data
        states = torch.stack([t["state_enc"][0:1] for t in trajectory]).to(device)
        states = states.squeeze(1)  # (B, 64) — use agent 0 as ego
        actions = torch.stack([t["action"][0:1] for t in trajectory]).squeeze(1).to(device)
        log_probs_old = torch.stack([t["log_prob"][0:1] for t in trajectory]).squeeze(1).to(device)
        rewards = torch.tensor([t["reward"] for t in trajectory], device=device, dtype=torch.float32)
        
        # Discounted returns
        returns = torch.zeros_like(rewards)
        G = 0.0
        for i in range(B - 1, -1, -1):
            G = rewards[i] + self.gamma * G
            returns[i] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize
        
        # Value baseline
        values = self.value_net(states).squeeze(-1)  # (B,)
        advantages = returns - values.detach()
        
        # Policy loss (REINFORCE)
        log_probs = self.policy.get_log_prob(states, actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy bonus (encourage exploration)
        with torch.no_grad():
            mean = self.policy.mean_net(states)
            std = torch.exp(self.policy.log_std).expand_as(mean)
            entropy = Normal(mean, std).entropy().mean()
        
        # Total loss
        total_loss = policy_loss - self.entropy_coef * entropy
        
        # Update policy
        self.policy_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_opt.step()
        
        # Update value baseline
        value_loss = F.mse_loss(values, returns)
        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }
    
    def train(self, n_epochs=100, ep_per_epoch=5, scenarios=None):
        """Main training loop."""
        if scenarios is None:
            scenarios = ["intersection", "merge"]
        
        print()
        print("  {:>5} | {:>8} | {:>5} | {:>7} | {:>8} | {:>8}".format("Ep", "Reward", "Coll", "Conv%", "Loss", "Entropy"))
        print("-" * 55)
        
        for ep in range(n_epochs):
            ep_rewards = []; ep_collisions = []; ep_losses = []
            
            for _ in range(ep_per_epoch):
                sc = np.random.choice(scenarios)
                trajectory, sim = self.collect_episode(sc, max_steps=50)
                stats = self.update(trajectory)
                
                total_r = sum(t["reward"] for t in trajectory)
                ep_rewards.append(total_r)
                ep_collisions.append(sim.collision)
                ep_losses.append(stats.get("policy_loss", 0))
            
            avg_reward = np.mean(ep_rewards)
            avg_coll = np.mean(ep_collisions) * 100
            avg_loss = np.mean(ep_losses)
            entropy = stats.get("entropy", 0)
            conv_pct = (1 - avg_coll / 100) * 100 if avg_coll < 100 else 0
            
            self.log["epoch"].append(ep)
            self.log["reward"].append(avg_reward)
            self.log["collision_pct"].append(avg_coll)
            self.log["policy_loss"].append(avg_loss)
            self.log["entropy"].append(entropy)
            
            if ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1:
                print(f"  {ep:>5} | {avg_reward:>+8.3f} | {avg_coll:>5.1f}% | {conv_pct:>7.1f} | {avg_loss:>8.5f} | {entropy:>8.5f}")
            
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                torch.save(self.policy.state_dict(),
                           str(self.res_dir / "checkpoints" / "best_policy.pth"))
        
        torch.save(self.policy.state_dict(),
                   str(self.res_dir / "checkpoints" / "final_policy.pth"))
        self._save_log()
        self._plot_curve()
        print()
        print(f"[RL] Done. Best avg reward: {self.best_reward:.3f}")
    
    def _save_log(self):
        with open(str(self.res_dir / "rl_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.log.keys()))
            w.writeheader()
            for i in range(len(self.log["epoch"])):
                w.writerow({k: self.log[k][i] for k in self.log})
        
        json.dump({k: v for k, v in self.log.items()},
                  open(str(self.res_dir / "rl_log.json"), "w"))
        print(f"[Log] Saved to {self.res_dir}")
    
    def _plot_curve(self):
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            e = self.log["epoch"]
            axes[0].plot(e, self.log["reward"], "b-")
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Avg Reward"); axes[0].grid(alpha=0.3)
            
            axes[1].plot(e, self.log["collision_pct"], "r-")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Collision %"); axes[1].grid(alpha=0.3)
            
            axes[2].plot(e, self.log["entropy"], "g-")
            axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Policy Entropy"); axes[2].grid(alpha=0.3)
            
            plt.suptitle("REINFORCE Training Curves")
            plt.tight_layout()
            plt.savefig(str(self.res_dir / "rl_curves.png"), dpi=150); plt.close()
            print(f"[Plot] -> {self.res_dir / 'rl_curves.png'}")
        except Exception as ex:
            print(f"[Plot] {ex}")


    def evaluate(self, scenarios=None, max_steps=60):
        """Evaluate trained policy on multiple scenarios."""
        if scenarios is None:
            scenarios = list(SCENARIOS.keys())
        self.policy.eval()
        model = RubinsteinBargaining({
            "state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":3,
            "discount_alpha":0.3,"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0,
        }).to(DEVICE)
        model.eval()
        model.offer_decoder = self.policy.mean_net
        model.discount = self.model.discount
        
        print("\n[Eval] Trained policy results:")
        print("  {:>15} | {:>5} | {:>5} | {:>7} | {:>8} | {:>8}".format("Scenario","Steps","Coll","Conv%","MinTTC","AvgWPD"))
        print("  " + "-" * 55)
        for sc in scenarios:
            sim = SimState(sc); conv=0; steps=0; ttcs=[]; wpds=[]
            while not sim.finished and sim.step < max_steps:
                s = sim.get_state()
                feat = torch.randn(2, 256, 48, 48, device=DEVICE)
                with torch.no_grad():
                    out = model(spatial_features=feat, waypoints=s["waypoints"],
                               positions=s["positions"], velocities=s["velocities"],
                               headings=s["headings"], record_len=s["record_len"])
                ttcs.append(s["ttc"])
                wpds.append((out["adjusted_waypoints"]-s["waypoints"]).norm(dim=-1).mean().item())
                if out["metadata"][0]["converged"]: conv+=1
                steps+=1
                for v in range(2):
                    if v < out["adjusted_waypoints"].shape[0]:
                        dxy=(out["adjusted_waypoints"][v,0]-s["waypoints"][v,0]).cpu().numpy()
                    else:
                        dxy=-(out["adjusted_waypoints"][0,0]-s["waypoints"][0,0]).cpu().numpy()*0.5
                    sim.adjust(v, dxy[0], dxy[1])
                sim.step_sim()
            coll = "Y" if sim.collision else "."
            print(f"  {sc:<15} | {steps:>5d} | {coll:>5} | {round(conv/max(1,steps)*100,1):>7.1f} | {round(min(ttcs) if ttcs else 0,3):>8.3f} | {round(np.mean(wpds) if wpds else 0,4):>8.4f}")

def main():
    ap = argparse.ArgumentParser(description="REINFORCE training for bargaining")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--scenarios", nargs="+", default=["intersection","chase","merge","head_on"],
                    choices=list(SCENARIOS))
    ap.add_argument("--load-policy", type=str, default=None, help="Load pretrained policy weights")
    args = ap.parse_args()
    
    print("=" * 55)
    print("  REINFORCE: Self-Play Bargaining Training")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Gamma: {args.gamma}")
    print(f"  Scenarios: {args.scenarios}")
    print("=" * 55)
    print()
    
    trainer = REINFORCETrainer(lr=args.lr, gamma=args.gamma)
    if args.load_policy:
        ckpt = torch.load(args.load_policy, map_location=DEVICE, weights_only=False)
        if "offer_decoder" in ckpt:
            sd = {}
            for k, v in ckpt["offer_decoder"].items():
                k2 = k.replace("decoder.", "") if k.startswith("decoder.") else k
                sd[k2] = v
            trainer.policy.mean_net.load_state_dict(sd)
            print(f"[Joint] Loaded supervised offer_decoder from {args.load_policy}")
        else:
            trainer.policy.load_state_dict(ckpt)
            print(f"[Joint] Loaded policy from {args.load_policy}")
    trainer.train(n_epochs=args.epochs, ep_per_epoch=5, scenarios=args.scenarios)
    print()
    print(f"Done. Results: {trainer.res_dir}")
if __name__ == "__main__":
    main()
