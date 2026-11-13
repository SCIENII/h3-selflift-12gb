# -*- coding: utf-8 -*-
"""Emit the recommended graphs as API-format JSON presets.

Run from the repo root:  python export_presets.py
"""
import json
from pathlib import Path

import bench

HERE = Path(__file__).parent
OUT = HERE / "presets"
OUT.mkdir(exist_ok=True)

# primary: the configuration with the clearest win (8-step, t=6, −24.6 % median)
PRIMARY = dict(name="preset_8step_t6", mode="selflift", steps=8, lora=bench.LORA_8S,
               transition_step=6, lowres_scale=0.5, rho=0.0, w_min=1.0, w_max=1.0,
               upscaler_model=bench.UPS_3D)

# secondary: 4-step variant (−7.4 % median, within the noise band — flagged as such)
SECONDARY = dict(name="preset_4step_t3", mode="selflift", steps=4, lora=bench.LORA_4S,
                 transition_step=3, lowres_scale=0.5, rho=0.0, w_min=1.0, w_max=1.0,
                 upscaler_model=bench.UPS_3D)

# reference: what you are comparing against
BASELINE = dict(name="preset_8step_baseline", mode="baseline", steps=8, lora=bench.LORA_8S)


def main():
    for path, cfg in (("preset_8step_t6_selflift.json", PRIMARY),
                      ("preset_4step_t3_selflift.json", SECONDARY),
                      ("preset_8step_baseline.json", BASELINE)):
        g = bench.build_graph(cfg)
        (OUT / path).write_text(json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path:38s} {len(g):2d} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
