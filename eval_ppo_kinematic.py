
import os, sys, math, json, argparse
import torch
import numpy as np
from opencood.models.comm_modules.rubinstein_bargaining import RubinsteinBargaining

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260608
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],[self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],dtype=torch.float32,device=DEVICE)
        wps=torch.from_numpy(np.array(self.wps)).float().to(DEVICE)
        return {"positions":pos,"velocities":vel,"waypoints":wps,"headings":torch.tensor(self.hdg,device=DEVICE),"record_len":torch.tensor([2],device=DEVICE)}

def run_episode(model, scenario, max_steps=60):
    sim = SimState(scenario); conv=0; wpds=[]
    while not sim.finished and sim.step < max_steps:
        s = sim.get_state()
        feat = torch.randn(2, 256, 48, 48, device=DEVICE)
        with torch.no_grad():
            out = model(spatial_features=feat, waypoints=s["waypoints"], positions=s["positions"], velocities=s["velocities"], headings=s["headings"], record_len=s["record_len"])
        wpd = (out["adjusted_waypoints"]-s["waypoints"]).norm(dim=-1).mean().item()
        wpds.append(wpd)
        if out["metadata"][0]["converged"]: conv+=1
        for v in range(2):
            if v<out["adjusted_waypoints"].shape[0]:
                dxy=(out["adjusted_waypoints"][v,0]-s["waypoints"][v,0]).cpu().numpy()
            else:
                dxy=-(out["adjusted_waypoints"][0,0]-s["waypoints"][0,0]).cpu().numpy()*0.5
            sim.adjust(v,dxy[0],dxy[1])
        sim.step_sim()
    n=max(1,sim.step)
    return {"steps":sim.step,"collision":sim.collision,"conv":round(conv/n*100,2)}

class ConstantDiscount(torch.nn.Module):
    def __init__(self, delta=0.9):
        super().__init__();self.delta=torch.nn.Parameter(torch.tensor(delta),requires_grad=False)
    def forward(self, ttc):return self.delta.expand_as(ttc)

class HandcraftedUtility(torch.nn.Module):
    def forward(self, state, offer):
        return torch.sigmoid(10.0*(offer[:,3]-0.5*offer[:,0])-2.0)

variants = [
    ("TTC + 3 rounds + a=0.3 (baseline)", {"max_rounds":3,"discount_type":"ttc","discount_alpha":0.3,"utility_type":"learned"}),
    ("Constant + 3 rounds + a=0.3", {"max_rounds":3,"discount_type":"constant","discount_alpha":0.3,"utility_type":"learned"}),
    ("TTC + 1 round + a=0.3", {"max_rounds":1,"discount_type":"ttc","discount_alpha":0.3,"utility_type":"learned"}),
    ("TTC + 3 rounds + a=0.1", {"max_rounds":3,"discount_type":"ttc","discount_alpha":0.1,"utility_type":"learned"}),
    ("TTC + 3 rounds + a=1.0", {"max_rounds":3,"discount_type":"ttc","discount_alpha":1.0,"utility_type":"learned"}),
]

def create_model(vcfg, checkpoint_state):
    cfg = {"state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":vcfg["max_rounds"],"discount_alpha":vcfg["discount_alpha"],"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0}
    model = RubinsteinBargaining(cfg).to(DEVICE)
    model.eval()
    if vcfg["discount_type"] == "constant":
        model.discount = ConstantDiscount(delta=0.9).to(DEVICE)
    if vcfg["utility_type"] == "handcrafted":
        model.utility_estimator = HandcraftedUtility().to(DEVICE)
    new_sd = {}
    for k, v in checkpoint_state.items():
        if k.startswith("mean_net."):
            new_sd["decoder." + k[9:]] = v
    model.offer_decoder.load_state_dict(new_sd, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bargaining PPO policy in the kinematic simulator.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained PPO checkpoint.")
    parser.add_argument("--trials", type=int, default=25, help="Number of trials per scenario and variant.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    trials = args.trials
    print()
    print("=== PPO KINEMATIC VALIDATION (" + str(trials) + "x/scenario, seed=" + str(SEED) + ") ===")
    print()
    hdr = "  {:<40} {:>5} {:>5} {:>5} {:>5} {:>8}".format("Variant", "Int", "Chse", "Mrg", "HOn", "Coll%")
    print(hdr)
    print("  " + "-" * 70)

    summary = {
        "seed": SEED,
        "trials_per_scenario": trials,
        "scenario_order": list(SCENARIOS.keys()),
        "variants": [],
    }

    for name, vcfg in variants:
        model = create_model(vcfg, checkpoint)
        results = []
        for sc in list(SCENARIOS.keys()):
            for _ in range(trials):
                r = run_episode(model, sc, 60)
                model.eval()
                r["scenario"] = sc
                results.append(r)
        stats = {}
        for sc in list(SCENARIOS.keys()):
            sc_r = [r for r in results if r["scenario"] == sc]
            coll = sum(1 for r in sc_r if r["collision"])
            stats[sc] = coll
        total_coll = sum(1 for r in results if r["collision"])
        total_all = len(results)
        coll_pct = total_coll * 100 // total_all
        line = "  {:<40} {:>5} {:>5} {:>5} {:>5} {:>7}%".format(name, stats["intersection"], stats["chase"], stats["merge"], stats["head_on"], coll_pct)
        print(line)
        summary["variants"].append({
            "name": name,
            "config": vcfg,
            "collisions": stats,
            "total_collisions": total_coll,
            "total_trials": total_all,
            "collision_pct": total_coll / total_all * 100.0,
        })

    print()
    print("  Legend: Int=Intersection, Chse=Chase, Mrg=Merge, HOn=Head-on")
    print("  Values are collisions out of " + str(trials) + " trials")

    out_path = os.path.join("Result", "ppo_kinematic_latest.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  JSON -> " + out_path)


if __name__ == "__main__":
    main()
