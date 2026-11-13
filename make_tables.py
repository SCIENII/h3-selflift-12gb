# -*- coding: utf-8 -*-
"""Render results/medians.json into the markdown table used in the README.

Reads both results/results.jsonl (single-shot matrix) and results/medians.json
(repeated runs), so the README is always regenerated from data rather than
hand-edited.

    python make_tables.py > docs/tables.md
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"


def fmt(v, digits=1):
    return "—" if v is None else f"{v:.{digits}f}"


def load_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main():
    rows = load_jsonl(RES / "results.jsonl")
    med = {}
    if (RES / "medians.json").exists():
        med = json.loads((RES / "medians.json").read_text(encoding="utf-8"))

    base = next((r for r in rows if r["name"] == "sl_base4" and r.get("exec_s")), None)

    print("## Single-run matrix\n")
    print("| id | exec (s) | vs baseline | peak VRAM (GB) | status |")
    print("|---|---:|---:|---:|---|")
    for r in rows:
        e = r.get("exec_s")
        if base and e:
            d = (e - base["exec_s"]) / base["exec_s"] * 100
            delta = f"{d:+.1f} %"
        else:
            delta = "—"
        print(f"| `{r['name']}` | {fmt(e, 2)} | {delta} | "
              f"{fmt(r.get('peak_vram_gb'), 2)} | {r.get('status')} |")

    if med:
        print("\n## Medians of repeated runs\n")
        print("| config | median (s) | min | max | spread | peak VRAM (GB) |")
        print("|---|---:|---:|---:|---:|---:|")
        order = ["baseline4", "lift3d", "lift3d_8s", "base8"]
        for k in [k for k in order if k in med] + [k for k in med if k not in order]:
            d = med[k]
            spread = (d["max"] - d["min"]) / d["median"] * 100 if d.get("median") else 0
            print(f"| `{k}` | {fmt(d['median'], 2)} | {fmt(d['min'], 2)} | "
                  f"{fmt(d['max'], 2)} | {spread:.1f} % | {fmt(d.get('peak_gb'), 2)} |")

        # compare each SelfLift config against the baseline of the SAME step count
        pairs = [("lift3d", "baseline4"), ("lift3d_8s", "base8")]
        for sl, bl in pairs:
            if sl in med and bl in med:
                b = med[bl]["median"]
                d = (med[sl]["median"] - b) / b * 100
                print(f"\n`{sl}` vs `{bl}`: **{d:+.1f} %**")


if __name__ == "__main__":
    main()
