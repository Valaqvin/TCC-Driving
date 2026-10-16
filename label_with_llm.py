# -*- coding: utf-8 -*-
"""
label_with_llm.py - Label driving scenes with Qwen2.5-1.5B-Instruct.

Uses a local LLM to annotate socially optimal offers for each scene.
The LLM outputs a JSON with optimal speed_factor, lateral_shift,
time_headway, and yield_willingness for both vehicles.

Usage:
    python label_with_llm.py --scenes Result/distill_data/scenes.json
    python label_with_llm.py --max-scenes=50  (test with 50 scenes)

Output: Result/distill_data/labeled_scenes.json
"""

import os, sys, json, math, time, re
from pathlib import Path
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_TEMPLATE = """You are a traffic negotiation expert. Two vehicles approach a conflict zone.

Scenario: {scenario}
Vehicle A: position ({ax},{ay}), speed {as_}m/s
Vehicle B: position ({bx},{by}), speed {bs_}m/s

Negotiation parameters (each 0-1):
1. speed_factor: 0=stop, 1=full speed
2. lateral_shift: 0=no shift, 1=max shift
3. time_headway: 0=aggressive, 1=conservative
4. yield_willingness: 0=never yield, 1=always yield

Social rules: faster vehicle yields when TTC<3s. Both adjust moderately at TTC<6s.

Output ONLY valid JSON with format: {{"vehicle_a":{{"speed_factor":0.x,"lateral_shift":0.x,"time_headway":0.x,"yield_willingness":0.x}},"vehicle_b":{{"speed_factor":0.x,"lateral_shift":0.x,"time_headway":0.x,"yield_willingness":0.x}}}}"""


def load_model():
    import os as _os
    local = _os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/manual_v1")
    if _os.path.isdir(local):
        print(f"[LLM] Loading from local cache: {local}")
        tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    else:
        print(f"[LLM] Loading Qwen/Qwen2.5-1.5B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
    print(f"[LLM] Loaded on {model.device}")
    return model, tokenizer


def label_scene(scene, model, tokenizer):
    prompt = PROMPT_TEMPLATE.format(
        scenario=scene["scenario"],
        ax=scene["pos_a"][0], ay=scene["pos_a"][1],
        as_=scene["speed_a"],
        bx=scene["pos_b"][0], by=scene["pos_b"][1],
        bs_=scene["speed_b"],
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        offers = json.loads(response[start:end])
        return offers
    except (ValueError, json.JSONDecodeError):
        return None

def label_all(scenes_path, output_path, max_scenes=None, batch_size=1):
    model, tokenizer = load_model()
    scenes = json.load(open(scenes_path))
    output = scenes[:max_scenes] if max_scenes else scenes
    total = len(output)
    print(f"[Label] Processing {total}/{len(scenes)} scenes...")

    for i, scene in enumerate(output):
        if scene.get("llm_offers"):
            continue
        offers = label_scene(scene, model, tokenizer)
        scene["llm_offers"] = offers if offers else None
        scene["llm_labeled"] = offers is not None

        if (i + 1) % 10 == 0 or (i + 1) == total:
            json.dump(output, open(output_path, "w"), indent=2)
            n_lab = sum(1 for s in output if s.get("llm_labeled"))
            print(f"  [{i+1}/{total}] {n_lab} labeled, saved to {output_path}")

    n_lab = sum(1 for s in output if s.get("llm_labeled"))
    print(f"[Label] Complete. {n_lab}/{len(output)} scenes labeled")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=str, default=str(Path(PROJECT_ROOT) / "Result" / "distill_data" / "scenes.json"))
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args()

    output = str(Path(args.scenes).parent / "labeled_scenes.json")
    label_all(args.scenes, output, args.max_scenes)


if __name__ == "__main__":
    main()
