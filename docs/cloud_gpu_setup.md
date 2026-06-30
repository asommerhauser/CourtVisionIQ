# Running the full train + predictions on a paid cloud GPU

The local PC can't train this. This guide gets the whole pipeline — **train 2** (availability
masking + capacity bump) plus the **full holdout prediction run** — onto a rented GPU cheaply.

The model stack is *small* (6–12 transformer heads, ~3–9M params each, trained sequentially on
one GPU). It is more I/O- and overhead-bound than compute-bound, so you do **not** need an
A100/H100 — a single 24 GB consumer card is plenty and far cheaper.

---

## 1. What to rent (cheap + fast + enough capacity)

**Recommended: [RunPod](https://www.runpod.io) + 1× RTX 4090 (24 GB).** Per-second billing,
persistent network volumes (upload the 4 GB of data once, reuse it), SSH + JupyterLab, and the
`tensorflow[and-cuda]` wheel in `requirements-gpu.txt` brings its own CUDA so there's nothing to
install at the driver level.

| GPU | VRAM | ~$/hr (community) | When to pick it |
|---|---|---|---|
| **RTX 4090** | 24 GB | ~$0.34–0.44 | **Default.** Fast, cheap, fits train 2 comfortably. |
| RTX A40 / A6000 | 48 GB | ~$0.40–0.79 | Want to crank batch sizes way up (more games per forward pass → faster). |
| A100 80 GB | 80 GB | ~$1.2–1.9 | Only if you scale capacity/holdout a lot. Usually overkill here. |
| H100 80 GB | 80 GB | ~$2–3 | Overkill for a model this small — don't bother. |

**Alternatives:** [Vast.ai](https://vast.ai) is usually a bit cheaper (a marketplace — pick a
host with good reliability), [Lambda](https://lambdalabs.com) is clean on-demand A100/H100.
RunPod is the best balance of cheap + simple for this job.

**Rough cost:** train ~2–4 h + full 100-game eval ~1–3 h ≈ **$3–10 total** on a 4090, including
some iteration. Stop the pod the moment you've pulled results back (see §6).

---

## 2. Get the code onto the pod

The repo is on GitHub at `github.com/asommerhauser/CourtVisionIQ`. The vocabs
(`encoder/vocabs/*.json`) are committed, so a clone brings everything **except** the data.

```bash
cd /workspace                      # a persistent network volume mount on RunPod
git clone https://github.com/asommerhauser/CourtVisionIQ.git
cd CourtVisionIQ
```

Make sure `main` is pushed first (it carries train 2 + the `eval-all` command):
```bash
# on your PC
git push origin main
```

---

## 3. Get the data onto the pod

Training reads the **cleaned season CSVs** in `data/` (`season2003.csv … season2023.csv`, ~4 GB
total). These are *not* in git — transfer them separately. You do **not** need `RawData/` (3.7 GB)
or `PlayByPlayLogs/` (0.7 GB) — those were only for the one-time cleaning step, already done. So
the bulky raw data stays home; only the ~4 GB of cleaned CSVs travel, and they gzip to ~1 GB.

> **"Can't I just ship the preprocessed tensors instead?"** Not cleanly. (1) The local
> `data/processed/*.npz` are *train-1* tensors — they have no `avail_mask`, so they'd train without
> the masking; they'd have to be regenerated with the train-2 code first. (2) The prediction/eval
> step reads the **real** holdout games from these CSVs to score against and to seed each sim, and
> the global `game_id` numbering depends on the full file set — tensors can't do that. (3)
> `full_train.py train` re-preprocesses from `data/` on every run. The tensors are also only ~half
> the size of the gzipped CSVs (~0.55 GB vs ~1 GB), so shipping the CSVs once is the simpler win.

Compress first (play-by-play text shrinks ~3–4×):
```bash
# on your PC, from the repo root
tar -czf seasons.tgz data/season*.csv          # ~1–1.3 GB
```

Then move it across — easiest is RunPod's peer-to-peer `runpodctl`:
```bash
# on your PC
runpodctl send seasons.tgz                       # prints a one-time code
# on the pod
runpodctl receive <code>
mkdir -p data && tar -xzf seasons.tgz            # restores data/season*.csv
```
(Alternatives: drag-and-drop into JupyterLab, `rclone` from an S3/B2/Drive bucket, or
`scp`/`rsync` to the pod's SSH endpoint.)

> Tip: put `data/` on the **persistent network volume** so you only upload once — future pods
> just re-clone the code and reuse the same data.

---

## 4. Environment + GPU check

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt              # pulls tensorflow[and-cuda]==2.20.0 (bundles CUDA)
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```
You should see one GPU listed. Mixed precision is already on by default in the train code.

---

## 5. Run it

```bash
python full_train.py setup        # compute the mid-2023 cut + 100-game holdout (no re-clean)
python full_train.py train        # train 2: every head, fresh, recency-weighted -> ./artifacts_full2
python full_train.py eval-all     # predict ALL 100 holdout games straight through (no stopping)
```

- **`train`** writes weights to `./artifacts_full2` and a per-head report under `reports/` tagged
  `full_train_2`. If it dies mid-run, just re-run `train` — it resumes at the next unfinished head.
- **`eval-all`** is the new paid-GPU command: it predicts the whole holdout in one process and
  **writes/refreshes the eval report every 10 games** (so you get a progress snapshot without
  stopping). It resumes too — finished games are skipped. (The old `eval` still exists for the
  local "10 games then stop" workflow.)

### Capacity knobs to raise on a bigger card
The defaults are tuned for a tight ~10 GB GPU. With 24 GB+ you can go faster / do more:

| Knob | File | Default | Raise to | Effect |
|---|---|---|---|---|
| `LARGE_OUTPUT_BATCH` | `models/pipeline.py` | 16 | 24–48 | Bigger train batch for the player-vocab heads → fewer steps. |
| `batch_size` | passed to `setup` (`--batch-size`) | 32 | 64–128 | Train batch for the small heads. |
| `ROLLOUT_BATCH_SIZE` | `config.py` | 16 | 32–64 | More concurrent game-sims pooled per GPU forward pass → faster eval. Pure throughput, no effect on results. |
| `FINAL_HOLDOUT_GAMES` | `config.py` | 100 | 200+ | Predict more holdout games (more robust eval). Re-run `setup` after changing. |
| `STAGE_SIMS` | `config.py` | 11 | 15–21 | More sims per game → tighter averaged box scores (and bigger error bars estimate). |

Watch the first training epoch's memory; if it OOMs, drop `LARGE_OUTPUT_BATCH`. If there's lots
of headroom, raise it (and `ROLLOUT_BATCH_SIZE` for eval).

---

## 6. Pull results back, then stop the pod

You want `reports/` (the HTML + Parquet eval + training reports) and, if you want to re-run
predictions later, `artifacts_full2/` (the weights).

```bash
# on the pod
tar -czf train2_out.tgz reports artifacts_full2
runpodctl send train2_out.tgz
# on your PC
runpodctl receive <code> && tar -xzf train2_out.tgz
```

Then **terminate the pod** so billing stops. If you kept `data/` (and optionally
`artifacts_full2/`) on a persistent network volume, the next run is just: spin up a pod, re-clone
the code, reuse the volume.

---

## Notes
- I never launch training/rollout myself — these commands are for you to run on the pod.
- Train 1's weights/reports under `./artifacts_full` are untouched; train 2 lives under
  `./artifacts_full2` / report tag `full_train_2`, so you can compare the two.
- Determinism: the train/val/holdout split is seeded (`SEED=42` in `config.py`), so the cloud run
  uses the same partitions as a local run would.
