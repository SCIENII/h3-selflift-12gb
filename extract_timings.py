# -*- coding: utf-8 -*-
"""Extract per-stage SelfLift timings from a ComfyUI console log.

The node pack logs `[SelfLift timing] ...` lines per stage and `Prompt executed
in N seconds` at the end of each prompt, so the log is enough to reconstruct
where the time went without instrumenting the sampler.

    python extract_timings.py <comfy.log> [--json out.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PLAN = re.compile(r"\[SelfLift plan\] (.+)$")
TIM = re.compile(r"\[SelfLift timing\] (.+?):\s*([0-9.]+)s")
DONE = re.compile(r"Prompt executed in ([0-9.]+) seconds")


def parse(path):
    runs, cur = [], None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw).strip()
        m = PLAN.search(line)
        if m:
            # a plan line opens a new SelfLift run
            cur = {"plan": m.group(1).strip(), "stages": {}}
            runs.append(cur)
            continue
        m = TIM.search(line)
        if m and cur is not None and "total" in m.group(1):
            stage = m.group(1).split(" total")[0]
            cur["stages"][stage] = float(m.group(2))
            continue
        m = DONE.search(line)
        if m:
            if cur is not None:
                cur["prompt_executed_s"] = float(m.group(1))
                cur = None
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = parse(a.log)
    if not runs:
        print("no SelfLift runs found")
        return 1
    for i, r in enumerate(runs, 1):
        st = r["stages"]
        low, tr, hi = st.get("low_resolution"), st.get("transition"), st.get("high_resolution")
        parts = [f"run {i}"]
        if low is not None:
            parts.append(f"low={low:.1f}s")
        if tr is not None:
            parts.append(f"transition={tr:.1f}s")
        if hi is not None:
            parts.append(f"high={hi:.1f}s")
        if "prompt_executed_s" in r:
            parts.append(f"total={r['prompt_executed_s']:.1f}s")
        print("  ".join(parts))
        print(f"    plan: {r['plan'][:150]}")
    if a.json:
        Path(a.json).write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
