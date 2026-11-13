# Raw measurements

Everything here comes from `docs/comfyui-console.log` (re-parsed by
`extract_timings.py`) and `results/*.json`. Nothing is rounded or estimated
except where marked.

## Decisive result: medians of 3

`repeat_runs.py --reps 3`, queue drained before every submission, clock starts
when ComfyUI picks the prompt up.

| config | runs (s) | median | vs same-step baseline | peak VRAM |
|---|---|---:|---:|---:|
| `baseline4` (4-step LoRA) | 151.59 / 136.38 / 147.98 | 147.98 | — | 9.63 GB |
| `lift3d` (4-step + SelfLift t=3) | 137.03 / 129.88 / 142.39 | 137.03 | **−7.4 %** | 9.72 GB |
| `base8` (8-step LoRA) | 202.08 / 201.91 / 192.16 | 201.91 | — | 9.78 GB |
| `lift3d_8s` (8-step + SelfLift t=6) | 162.68 / 152.23 / 149.60 | 152.23 | **−24.6 %** | 9.71 GB |

The win scales with step count: 7 % at 4 steps, 25 % at 8. Same card, same
resolution, same prompt, same seed.

### Why this section exists: single runs invert the sign

The first pass ran each configuration once and produced:

```
baseline4  126.22 s   ← appears to win
lift3d     131.07 s   ← appears to lose by 3.8 %
sl_zero_r06 201.37 s
sl_lift3d_8s 177.03 s vs sl_base8b 181.55 s
```

Repeating only the first two, three times each, produced the opposite ordering
(147.98 vs 137.03) with 10 % within-config spread. Every config in this repo
with spread ≥ the claimed effect is flagged; the 4-step result is marginal
(7 % effect, 10 % spread) and the 8-step result is not (25 % effect, 5 % spread).

## Single-run matrix (`results/results.jsonl`)

One run each — useful only for the shape of the cost structure, not for
comparing configurations. 864×480, 73 frames, 24 fps, seed 42.

| id | what changed | exec (s) | peak VRAM (GB) |
|---|---|---:|---:|
| `sl_base4` | baseline, 4-step turbo LoRA | 126.22 | 9.83 |
| `sl_lift3d` | + learned lift `3d_conv`, t=3, scale 0.5, rho=0 | 131.07 | 9.58 |
| `sl_lift3d_s04` | same, scale 0.4 | 135.41 | 9.62 |
| `sl_lift3d_tst` | same as lift3d + TST τ=0.2 | 169.33 | 9.79 |
| `sl_lift3d_t2` | learned lift, t=2 (2 low / 2 high) | 173.52 | 9.67 |
| `sl_base8b` | baseline, **8-step** turbo LoRA | 181.55 | 9.45 |
| `sl_lift3d_8s` | learned lift on the 8-step LoRA, t=6 | 177.03 | 9.65 |
| `sl_zero_r06` | **SelfLift-zero**, rho=0.6 (VAE round trip) | 201.37 | 9.60 |
| `sl_lift2d` | learned lift `2d_conv` | *failed* | — |

### `sl_lift2d` failure

```
ValueError: SelfLift: upscaler checkpoint is missing conv_in.weight
  h3_upscaler.py:290 in _normalize_checkpoint_dtype
```

The 2d_conv checkpoint stores weights under a `resizer.` prefix; the 3d_conv
one does not. Verified by reading the safetensors headers directly:

```
minimax_h3_latent_upscaler_2d_conv_v1_bf16 -> 262 tensors
  resizer.conv_in.weight, resizer.conv_out.weight, resizer.embed.linear.0.weight, ...
minimax_h3_latent_upscaler_3d_conv_v1_bf16 -> 322 tensors
  conv_in.weight, conv_out.weight, embed.0.weight, ...
```

Re-saving the 2d variant with the prefix stripped produces a loadable file
(262 → 262 tensors, keys matched). See the README snippet.

## Per-stage decomposition

From the console log, the three stages of a SelfLift run. `transition` has no
counterpart in the baseline — it is pure added cost.

| route | low-res prefix | transition | high-res tail | sum |
|---|---:|---:|---:|---:|
| learned lift `3d_conv`, t=3 scale 0.5 | 35.1 | **4.2** | 12.9 | 52.2 |
| learned lift, t=3 scale 0.5 (2nd run) | 42.2 | 5.8 | 11.7 | 59.7 |
| learned lift, t=3 scale 0.4 | 35.9 | 6.2 | 12.9 | 55.0 |
| learned lift, t=2 | 49.1 | 2.7 | 28.3 | 80.1 |
| learned lift, 8-step LoRA, t=6 | 58.1 | 5.2 | 25.6 | 88.9 |
| **SelfLift-zero, rho=0.6 (VAE)** | 38.6 | **63.5** | 18.5 | 120.6 |

The VAE route's transition is ~13× the learned route's. That single row is the
entire reason the paper-faithful configuration loses on this hardware, and it is
also why dropping `rho` to 0 with an external upscaler is not a micro-tune but a
structural change.

`t=2` is a visible trap: the transition is cheap (2.7 s) but 2 of 4 evaluations
now run at full resolution (28.3 s tail), so the prefix saved almost nothing.

Stage sums don't equal `exec_s`: model loading, text encoding, final VAE decode
and muxing sit outside the three marked stages.

## Run-to-run noise

Repeating the *same* configuration varies materially:

```
low_resolution totals observed: 35.1  35.9  38.6  41.1  42.2  49.1  57.3  58.1 s
baseline4 totals observed:     136.38  147.98  151.59 s
```

Thermals (laptop chassis), Windows memory pressure, and ComfyUI's dynamic-VRAM
staging decisions. This is why `repeat_runs.py` exists and why the README
reports medians with the range attached.

## Environment

```
GPU     NVIDIA GeForce RTX 3060 Laptop GPU, 12288 MiB, cc 8.6, driver 610.62
CPU     Intel Core i5-8500 @ 3.00 GHz
RAM     31.9 GB (46.9 GB pagefile allocated, 2.8 GB peak in use)
OS      Windows, ComfyUI 0.33.0, Python 3.12.9, torch 2.9.1+cu130
flags   --reserve-vram 1.5 --vram-headroom 1.0 --async-offload 2
        --use-sage-attention --cache-none
models  MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors (14.06 GB)
        qwen3vl_32b_minimax_h3_int8_convrot.safetensors (25.28 GB)
        minimax_h3_video_vae_fp16 / minimax_h3_audio_vae_fp32
        minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16 (4-step)
        minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16 (8-step)
```

Free physical RAM during the VAE-route transition measured **1.6–2.2 GB**. On a
32 GB machine that is the binding constraint, not the 12 GB of VRAM.
