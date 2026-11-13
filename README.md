# h3-selflift-12gb

**SelfLift for MiniMax-H3 on a 12 GB laptop GPU — medians, not single shots.**

The paper reports 41.5 % / 44.1 % latency cuts; a H3 director-panel blog post
reports ~54 % on a 2 MP 10-second clip. Both are real. Neither is what you get
if you copy their parameters onto an RTX 3060 Laptop at 864×480 — and if you
measure with one run per config, you will conclude the exact opposite of the
truth (I did, twice).

This repo is the harness, the raw numbers, and the break-even model that tells
you when SelfLift pays on hardware like this.

Measured on: RTX 3060 **Laptop** 12 GB (GA106, driver 610.62), i5-8500,
32 GB DDR4, Windows, ComfyUI 0.33.0, torch 2.9.1+cu130,
`--reserve-vram 1.5 --vram-headroom 1.0 --async-offload 2 --use-sage-attention`.

## Result: medians of 3, queue drained between runs

864×480, 73 frames (3 s @ 24 fps), seed 42, turbo LoRA:

| Configuration | Median | Range | vs its baseline | Peak VRAM |
|---|---:|---|---:|---:|
| Baseline, 4-step LoRA | 147.98 s | 136.4–151.6 | — | 9.63 GB |
| **SelfLift, 4-step, learned lift, t=3/4** | **137.03 s** | 129.9–142.4 | **−7.4 %** | 9.72 GB |
| Baseline, 8-step LoRA | 201.91 s | 192.2–202.1 | — | 9.78 GB |
| **SelfLift, 8-step, learned lift, t=6/8** | **152.23 s** | 149.6–162.7 | **−24.6 %** | 9.71 GB |

SelfLift wins in both regimes, and **the win grows with step count** — 7 % at
4 steps, 25 % at 8. That is exactly what the mechanism predicts, and it is the
single most useful thing in this repo: the speedup is not a fixed number you
can quote, it is `transition_step / N` of the denoiser work, minus a
roughly-constant transition tax.

## The trap: single runs give you the wrong sign

My first pass ran each config once. It produced:

```
baseline4  126.22 s   ← "wins"
lift3d     131.07 s   ← "loses by 3.8 %"
```

Three repeats of the *same two configs* produced medians of 147.98 s and
137.03 s — **the opposite ordering**, with per-run spread of 10 %:

```
baseline4 runs:  151.59  136.38  147.98
lift3d    runs:  137.03  129.88  142.39
```

Laptop thermals plus Windows memory pressure plus ComfyUI's dynamic-VRAM
staging decisions move a single run by more than the effect size. Anything in
this repo built on one run is labelled as such; the table above is medians.
If a SelfLift benchmark from anywhere quotes a number without a range, assume
this trap applies to it too.

## Why the paper-faithful route is the worst one here

SelfLift-zero (the training-free route with the VAE re-encode anchor) is what
you reach for first, because it needs no extra checkpoint. On this card it is
the slowest option in the sweep — slower than just doubling your step count:

| Route (4-step, t=3/4, scale 0.5) | Exec | Transition cost | Note |
|---|---:|---:|---|
| Baseline, no SelfLift | 126.2 s | — | single run |
| Learned lift `3d_conv`, `rho=0` | 131.1 s | **5.8 s** | single run |
| Learned lift + TST τ=0.2 | 169.3 s | 5.2 s | TST alone ≈ +29 % |
| **SelfLift-zero**, `rho=0.6` | 201.4 s | **63.5 s** | VAE round trip dominates |

The transition stage is where the money goes:

```
learned lift:  low 35.1s | transition  4.2s | high 12.9s
SelfLift-zero: low 38.6s | transition 63.5s | high 18.5s
```

`paired_lifts` builds a pixel anchor by decoding the low-res latent, upscaling
73 frames to 864×480, and re-encoding. That is ~60 s of pure added latency on
this machine, and a 4-step schedule doesn't bank enough denoiser time to pay
for it. Set `upscaler_model=<3d_conv>`, `rho=0` and the transition collapses to
~5 s — that single change is the difference between "SelfLift is a slowdown"
and "SelfLift is a 25 % speedup".

**Free physical RAM during that VAE pass: 1.6–2.2 GB**, on a 32 GB box with a
46.9 GB pagefile already in use. On this class of machine the binding
constraint is RAM, not the 12 GB of VRAM.

## Break-even model

```
saving  = (transition_step / N) × T_full_step
tax     = T_transition        (≈5 s learned, ≈60 s VAE route here)
win     ⇔ transition_step/N × T_full_step > tax
```

With `T_full_step ≈ 20 s` at this resolution: the VAE route needs ~3 steps at
full res just to break even (`t/N ≈ 0.24`), which is why it loses at both 4 and
8 steps. The learned route needs only `t/N ≳ 0.25` — the 8-step run at t=6/8
sits far past that. **Scale N up or scale the tax down; there is no third lever.**

## Compatibility landmine

`minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors` **fails to load** in
`comfyui-SelfLift`:

```
ValueError: SelfLift: upscaler checkpoint is missing conv_in.weight
  h3_upscaler.py:290 in _normalize_checkpoint_dtype
```

The 2d variant ships weights under a `resizer.` prefix; the loader doesn't
strip it. The 3d variant has no prefix and loads fine. Verified by reading the
safetensors headers directly (262 tensors prefixed vs 322 unprefixed).
Workaround:

```python
from safetensors.torch import load_file, save_file
sd = load_file("minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors")
save_file({k[8:] if k.startswith("resizer.") else k: v for k, v in sd.items()},
          "minimax_h3_latent_upscaler_2d_conv_v1_bf16_unprefixed.safetensors")
```

## Reproducing

```bash
cd ComfyUI/custom_nodes && git clone https://github.com/facok/comfyui-SelfLift.git
# weights → models/latent_upscale_models/
#   minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors

python bench.py --list                       # single-shot sweep
python bench.py --run all
python repeat_runs.py --reps 3               # ← the numbers that matter
python extract_timings.py <console.log>      # per-stage decomposition
python make_tables.py                        # regenerate markdown from data
```

`bench.py` builds the graph in API format, **drains the queue and blocks until
`/queue` reports idle before submitting**, and starts its clock when ComfyUI
actually picks the prompt up. That queue-drain is not cosmetic: an earlier
version measured 151 s baseline vs 137 s SelfLift purely because two jobs
interleaved in the queue.

## Honest limits

- One seed, one prompt, one resolution per row. Solid on **cost**, silent on
  **quality** — no FID, no VBench, no human preference study. `presets/` holds
  the winning graph so you can eyeball output yourself.
- Repeats run sequentially in one block, so thermal drift is a live confound.
  Within-config spread is 5–10 %; the 8-step result (25 %) clears it easily, the
  4-step result (7 %) is marginal and deserves a longer run before you quote it.
- Peak VRAM is `/system_stats` polling, not `torch.cuda.max_memory_allocated`.
  A floor, ±0.3 GB.
- Laptop GA106 ≠ desktop GA106: fewer SMs, lower clocks. Expect a desktop 3060
  to move both columns; the ordering won't flip.
- TST measured **+29 %** here against an upstream claim of 1–2 %. Budget for it
  as a real cost if you want its temporal-stability effect.

## Layout

```
bench.py             graph builder + harness; all model ids in one fixtures block
repeat_runs.py       median-of-N on the decisive configs
extract_timings.py   per-stage cost reconstruction from a console log
make_tables.py       regenerate README tables from results/
docs/                measurements.md, raw console log, parsed stage JSON
presets/             winning config as API-format JSON
results/             results.jsonl, medians.json, protocol notes
```

## Credits

- [SelfLift paper](https://arxiv.org/abs/2609.02036) — Wen et al.
- [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) — the ComfyUI port measured here
- [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) — learned latent lift
- [MiniMax-H3](https://www.minimax.io/blog/minimax-h3)

MIT for the harness and docs. The node pack, upscaler weights and base model
keep their own licenses.
