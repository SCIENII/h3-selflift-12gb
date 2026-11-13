# -*- coding: utf-8 -*-
"""SelfLift on RTX 3060 Laptop 12GB — reproducible benchmark harness.

Builds the MiniMax-H3 graph in ComfyUI *API* format, submits it, samples VRAM
while it runs, and records wall-clock / peak-VRAM / output files to results/.

Usage:
    python bench.py --list
    python bench.py --run baseline4
    python bench.py --run all --repeat 1
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- fixed fixtures
UNET = "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_V = "minimax_h3_video_vae_fp16.safetensors"
VAE_A = "minimax_h3_audio_vae_fp32.safetensors"
LORA_4S = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
LORA_8S = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
UPS_3D = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
UPS_2D = "minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors"

PROMPT = (
    "A lone lighthouse keeper climbs the spiral stairs at dusk, warm lamp light "
    "raking across weathered stone, sea spray visible through the window. "
    "The camera tracks upward behind him, then settles on his face as he reaches "
    "the lamp room. The word \"SELFLIFT\" is painted on a brass plate beside the door "
    "and stays sharp and legible in every frame. "
    "Audio: low wind, distant foghorn every two seconds, footsteps on iron stairs, "
    "a slow solo cello underneath."
)

W, H, LEN = 864, 480, 73  # 16:9, multiple of 32, 73 frames @24fps = 3s (17k+5 grid)


def _base_nodes(cfg):
    """Model-loading + conditioning subgraph shared by every configuration."""
    steps = cfg["steps"]
    n = {
        "127": {"class_type": "UNETLoader",
                "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "128": {"class_type": "CLIPLoader",
                "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "119": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_V}},
        "120": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_A}},
        "134": {"class_type": "MiniMaxH3TurboLoRA",
                "inputs": {"model": ["127", 0], "lora_name": cfg["lora"],
                           "strength": 1.0, "low_vram": True}},
        "131": {"class_type": "MiniMaxH3ImageToVideo",
                "inputs": {"clip": ["128", 0], "vae": ["119", 0], "prompt": PROMPT,
                           "width": W, "height": H, "length": LEN}},
        "124": {"class_type": "BasicScheduler",
                "inputs": {"model": ["134", 0], "scheduler": "simple",
                           "steps": steps, "denoise": 1.0}},
        "129": {"class_type": "RandomNoise",
                "inputs": {"noise_seed": cfg.get("seed", 42)}},
        "122": {"class_type": "VAEDecode",
                "inputs": {"samples": ["125", 0], "vae": ["119", 0]}},
        "121": {"class_type": "VAEDecodeAudio",
                "inputs": {"samples": ["125", 0], "vae": ["120", 0]}},
        "130": {"class_type": "CreateVideo",
                "inputs": {"images": ["122", 0], "audio": ["121", 0], "fps": 24}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["130", 0],
                          "filename_prefix": cfg["name"], "format": "auto", "codec": "auto"}},
    }
    return n


def build_graph(cfg):
    """Assemble a full API-format prompt for one configuration."""
    n = _base_nodes(cfg)
    if cfg["mode"] == "baseline":
        n["135"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
        n["126"] = {"class_type": "BasicGuider",
                    "inputs": {"model": ["134", 0], "conditioning": ["131", 0]}}
        n["125"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["129", 0], "guider": ["126", 0],
                               "sampler": ["135", 0], "sigmas": ["124", 0],
                               "latent_image": ["131", 1]}}
    elif cfg["mode"] == "selflift":
        # SelfLift demands the stock Euler KSAMPLER; turbo LoRA still supplies the
        # acceleration, the schedule stays the model's own.
        n["137"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        n["131n"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["131", 0]}}
        model = ["134", 0]
        if cfg.get("tst_tau"):
            n["138"] = {"class_type": "SelfLiftH3TST",
                        "inputs": {"model": ["134", 0], "tau": cfg["tst_tau"],
                                   "log_diagnostics": True}}
            model = ["138", 0]
        n["139"] = {"class_type": "SelfLiftH3Sampler",
                    "inputs": {"model": model, "positive": ["131", 0],
                               "negative": ["131n", 0], "vae": ["119", 0],
                               "latent_image": ["131", 1], "sampler": ["137", 0],
                               "sigmas": ["124", 0], "seed": cfg.get("seed", 42),
                               "cfg": 1.0,
                               "transition_step": cfg["transition_step"],
                               "lowres_scale": cfg["lowres_scale"],
                               "rho": cfg["rho"], "w_min": cfg["w_min"],
                               "w_max": cfg["w_max"],
                               "upscaler_model": cfg.get("upscaler_model", "none"),
                               "upscaler_unload": True,
                               "highres_tiling": cfg.get("highres_tiling", False)}}
        # SelfLift integrates the whole schedule itself, so the outer
        # SamplerCustomAdvanced / BasicGuider / RandomNoise are not used; the
        # decoders read SelfLift's output latent directly.
        n["122"]["inputs"]["samples"] = ["139", 0]
        n["121"]["inputs"]["samples"] = ["139", 0]
        del n["129"]  # RandomNoise unused: SelfLift takes the seed itself
    else:
        raise ValueError(cfg["mode"])
    return n


def _renumber_latent(_n):
    """SelfLift path: sampler output 0 is the final latent; nothing else to fix."""
    return


# ---------------------------------------------------------------- http helpers
def _get(path, timeout=120):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def _post(path, payload, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _vram():
    try:
        s = _get("/system_stats", timeout=10)
        d = s["devices"][0]
        return d["vram_total"], d["vram_free"]
    except Exception:
        return None, None


def run_one(cfg, timeout=3600):
    graph = build_graph(cfg)
    # wait for an idle queue so submission latency never pollutes the timing
    for _ in range(600):
        q = _get("/queue", timeout=30)
        if not q["queue_running"] and not q["queue_pending"]:
            break
        time.sleep(1)
    resp = _post("/prompt", {"prompt": graph, "client_id": "selflift-bench"})
    pid = resp.get("prompt_id")
    if not pid:
        return {"name": cfg["name"], "status": "submit_failed", "resp": resp}

    total, peak_used, min_free = None, 0, None
    t_start = None  # set when the queue actually begins executing our prompt
    while True:
        if t_start is not None and time.time() - t_start > timeout:
            return {"name": cfg["name"], "status": "timeout", "prompt_id": pid}
        hist = _get(f"/history/{pid}", timeout=60)
        if t_start is None:
            q = _get("/queue", timeout=30)
            if any(item[1] == pid for item in q["queue_running"]):
                t_start = time.time()
            elif pid in hist:
                t_start = time.time()  # finished between polls
        total, free = _vram()
        if free is not None and t_start is not None:
            used = total - free
            peak_used = max(peak_used, used)
            min_free = free if min_free is None else min(min_free, free)
        if pid in hist:
            h = hist[pid]
            st = h.get("status", {})
            wall = time.time() - t_start if t_start else None
            files = []
            for _nid, out in (h.get("outputs") or {}).items():
                for key in ("images", "gifs", "video", "videos"):
                    for f in out.get(key, []) or []:
                        files.append(f"{f.get('subfolder','')}/{f.get('filename','')}")
            return {"name": cfg["name"], "status": st.get("status_str", "done"),
                    "ok": st.get("completed", False),
                    "exec_s": round(wall, 2) if wall else None,
                    "peak_vram_gb": round(peak_used / 1024**3, 2) if peak_used else None,
                    "min_free_gb": round(min_free / 1024**3, 2) if min_free else None,
                    "files": files, "cfg": cfg,
                    "errors": [m for m in st.get("messages", []) if m[0] == "execution_error"]}
        time.sleep(0.3)


CONFIGS = {
    # reference point: turbo LoRA only, no progressive resolution
    "baseline4": dict(name="sl_base4", mode="baseline", steps=4, lora=LORA_4S),
    "baseline8": dict(name="sl_base8", mode="baseline", steps=8, lora=LORA_8S),
    # learned latent lift (skips the VAE round trip entirely)
    "lift3d": dict(name="sl_lift3d", mode="selflift", steps=4, lora=LORA_4S,
                   transition_step=3, lowres_scale=0.5, rho=0.0, w_min=1.0, w_max=1.0,
                   upscaler_model=UPS_3D),
    "lift2d": dict(name="sl_lift2d", mode="selflift", steps=4, lora=LORA_4S,
                   transition_step=3, lowres_scale=0.5, rho=0.0, w_min=1.0, w_max=1.0,
                   upscaler_model=UPS_2D),
    # pivot sweeps for the learned route
    "lift3d_t2": dict(name="sl_lift3d_t2", mode="selflift", steps=4, lora=LORA_4S,
                      transition_step=2, lowres_scale=0.5, rho=0.0, w_min=1.0, w_max=1.0,
                      upscaler_model=UPS_3D),
    "lift3d_s04": dict(name="sl_lift3d_s04", mode="selflift", steps=4, lora=LORA_4S,
                       transition_step=3, lowres_scale=0.4, rho=0.0, w_min=1.0, w_max=1.0,
                       upscaler_model=UPS_3D),
    "lift3d_tst": dict(name="sl_lift3d_tst", mode="selflift", steps=4, lora=LORA_4S,
                       transition_step=3, lowres_scale=0.5, rho=0.0, w_min=1.0, w_max=1.0,
                       upscaler_model=UPS_3D, tst_tau=0.2),
    # SelfLift-zero parity route (VAE round trip), for the honest comparison
    "zero_r06": dict(name="sl_zero_r06", mode="selflift", steps=4, lora=LORA_4S,
                     transition_step=3, lowres_scale=0.5, rho=0.6, w_min=1.0, w_max=1.0),
    # 8-step non-turbo LoRA, same comparison
    "base8": dict(name="sl_base8b", mode="baseline", steps=8, lora=LORA_8S),
    "lift3d_8s": dict(name="sl_lift3d_8s", mode="selflift", steps=8, lora=LORA_8S,
                      transition_step=6, lowres_scale=0.5, rho=0.0, w_min=1.0, w_max=1.0,
                      upscaler_model=UPS_3D),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    if a.list:
        for k, v in CONFIGS.items():
            print(f"{k:16s} {json.dumps(v, ensure_ascii=False)}")
        return
    keys = list(CONFIGS) if a.run in (None, "all") else [a.run]
    for rep in range(a.repeat):
        for k in keys:
            cfg = dict(CONFIGS[k])
            cfg["name"] = f'{cfg["name"]}_r{rep}' if a.repeat > 1 else cfg["name"]
            r = run_one(cfg)
            print(json.dumps(r, ensure_ascii=False)[:400], flush=True)
            with open(OUT / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
