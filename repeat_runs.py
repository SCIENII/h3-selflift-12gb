# -*- coding: utf-8 -*-
"""Repeat runs on the key configurations and report medians.

Single-shot timings on this class of hardware vary by ±15 %, enough to flip the
sign of a 4 % difference. This script is the one that decides the conclusion.

    python repeat_runs.py [--reps 3] [--keys baseline4,lift3d]
"""
import argparse
import json
import statistics
import sys

import bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--keys", default="baseline4,lift3d,lift3d_8s,base8")
    a = ap.parse_args()
    keys = [k.strip() for k in a.keys.split(",") if k.strip()]
    out = {}
    for k in keys:
        ts, peaks = [], []
        for r in range(a.reps):
            cfg = dict(bench.CONFIGS[k])
            cfg["name"] = "%s_rep%d" % (cfg["name"], r)
            res = bench.run_one(cfg)
            if res.get("exec_s"):
                ts.append(res["exec_s"])
                peaks.append(res.get("peak_vram_gb"))
            print("%-12s rep%d  exec=%s peak=%s  %s"
                  % (k, r, res.get("exec_s"), res.get("peak_vram_gb"), res.get("status")),
                  flush=True)
        if ts:
            out[k] = {"runs": ts, "median": statistics.median(ts),
                      "min": min(ts), "max": max(ts),
                      "peak_gb": max(p for p in peaks if p)}
    print(json.dumps(out, indent=2))
    with open(bench.OUT / "medians.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
