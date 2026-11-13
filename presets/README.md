# presets

API-format ComfyUI graphs. Load with `POST /prompt` as
`{"prompt": <file contents>}`, or open the corresponding workflow in the UI.

| file | steps | SelfLift | median exec | note |
|---|---:|---|---:|---|
| `preset_8step_t6_selflift.json` | 8 | t=6/8, learned lift | 152.2 s | **recommended** — −24.6 % vs its baseline, 5 % run spread |
| `preset_4step_t3_selflift.json` | 4 | t=3/4, learned lift | 137.0 s | −7.4 %, but spread is 9 %; the win is marginal |
| `preset_8step_baseline.json` | 8 | none | 201.9 s | the thing you're comparing against |

All three: 864×480, 73 frames (3 s @ 24 fps), seed 42, 16:9, `simple` scheduler,
same prompt (embedded in the graph — edit `131.inputs.prompt`).

## Required models

```
diffusion_models/  MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors
text_encoders/     qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae/               minimax_h3_video_vae_fp16.safetensors
                   minimax_h3_audio_vae_fp32.safetensors
loras/             minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors   (4-step presets)
                   minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors   (8-step presets)
latent_upscale_models/
                   minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors       (SelfLift presets)
custom_nodes/      comfyui-SelfLift  (facok)
                   ComfyUI-MiniMax-H3-Turbo  (for the baseline preset's sampler)
```

## Tuning knobs that matter

| input | recommended | effect |
|---|---|---|
| `transition_step` | `steps - 2` | Fraction of evaluations at low res. The win scales with `t/N`; going to `t=2` of 4 loses because 2 full-res steps overwhelm what the prefix saved. |
| `lowres_scale` | `0.5` | 0.4 measured slower, not faster. The transition cost doesn't shrink with the prefix. |
| `rho` | `0.0` **with** an external upscaler | Anything above 0 turns on the VAE round trip (60 s here) and reverts you to the slow route. |
| `upscaler_model` | `3d_conv` | The `2d_conv` checkpoint fails to load — see README. |
| `upscaler_unload` | `true` | Frees VRAM before the high-res stage. |

Do not set `rho=0` with `upscaler_model=none`; the node rejects it, because
that combination disables every lift path.
