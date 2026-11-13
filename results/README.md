# results

Two files, two different jobs:

| file | what it is | how much to trust it |
|---|---|---|
| `results.jsonl` | single run per configuration, appended by `bench.py` | **cost structure only** — do not compare configs against each other |
| `medians.json` | median of 3 runs on the decisive configs, from `repeat_runs.py` | the actual conclusions |

Fields in `results.jsonl`:

| field | meaning |
|---|---|
| `name` | configuration id from `bench.py --list` |
| `status` | `success` / `error` / `timeout` / `submit_failed` |
| `exec_s` | wall clock from the moment ComfyUI *started executing* to completion. Submission and queue-wait latency are excluded. |
| `peak_vram_gb` | highest `vram_total - vram_free` sampled from `/system_stats` during the run (~3 Hz). A floor, not a true peak. |
| `min_free_gb` | lowest free VRAM observed |
| `files` | output files written by `SaveVideo` |
| `cfg` | exact configuration submitted |

## Measurement protocol

- Queue is drained and the harness blocks until `/queue` reports idle before
  each submission, so nothing interleaves.
- The clock starts when the prompt appears in `queue_running`, not at submit
  time.
- No warmup run is discarded: model loading is part of what a user actually
  pays. But it also means the first config in a batch is penalised — run
  `repeat_runs.py` and compare medians, not first runs.
- Fixed fixtures: 864×480, 73 frames, 24 fps, seed 42, one prompt (in
  `bench.py`). Changing any of these shifts all numbers.

## The one number to remember

Within-config spread on this laptop is **5–10 %**. Any claimed improvement
smaller than that, from a single run, is noise. In this repo that rule caught a
sign inversion: single runs said SelfLift lost by 3.8 %, three repeats each said
it won by 7.4 %.

Do not compare these numbers against the paper's 41.5 % / 44.1 % or a blog's
54 %: those are an 8-step image model and a 2 MP 10-second H3 clip. Different
regime, different ratio — the scaling law in the README's break-even section is
the transferable part.
