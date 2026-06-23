# CourtVisionIQ

An NBA play-by-play sequence model. The core **Event/Time Transformer** reads a game as a
sequence of events (with on-court rosters encoded by a Set Transformer) and predicts the
next event and the time until it. Built on **TensorFlow 2.20 / Keras 3**.

---

## GPU training on Windows (via WSL2)

> **Why WSL2?** TensorFlow dropped **native-Windows GPU support after v2.10**. This repo
> runs TF 2.20 / Keras 3, whose only Windows wheels are **CPU-only** — the RTX 4070 is
> invisible to native-Windows TF. The supported way to train on the GPU from a Windows
> machine is **WSL2 + Ubuntu** with `tensorflow[and-cuda]` (CUDA + cuDNN bundled via pip).
> WSL2 uses the **Windows** NVIDIA driver through GPU passthrough — no driver is installed
> inside Linux.

### Prerequisites (Windows side)

- A working **Windows NVIDIA driver** (check with `nvidia-smi` in PowerShell). Do **not**
  install an NVIDIA driver inside WSL — it breaks passthrough.
- **WSL2 with Ubuntu-22.04**. If not installed:
  ```powershell
  wsl --install -d Ubuntu-22.04
  wsl -l -v        # Ubuntu-22.04 should show VERSION 2
  ```
- **Enough RAM for WSL.** Training needs more host RAM than the WSL2 default cap
  (~50% of host), which otherwise OOM-kills the run. Create `C:\Users\<you>\.wslconfig`:
  ```ini
  [wsl2]
  memory=24GB
  swap=8GB
  ```
  then `wsl --shutdown` and reopen. See [docs/gpu_wsl2_setup.md](docs/gpu_wsl2_setup.md).

### One-time setup (inside WSL)

Open the distro and verify the GPU is visible, then build a venv **in the Linux
filesystem** (not under `/mnt/c` — venvs created on the Windows mount get corrupted):

```bash
wsl -d Ubuntu-22.04

nvidia-smi                                   # should list the RTX 4070

python3 -m venv ~/cviq-venv                  # Linux-fs venv (NOT on /mnt/c)
source ~/cviq-venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

Confirm TensorFlow sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# -> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Per-session

```bash
wsl -d Ubuntu-22.04
source ~/cviq-venv/bin/activate
cd /mnt/c/Projects/CourtVisionIQ          # repo, shared with Windows via the mount
```

---

## Data

Cleaned, model-ready season files live in `./data/season<YYYY>.csv` (e.g.
`data/season2003.csv`). The preprocessor consumes **every** `*.csv` in `./data` that has
`game_id` + roster columns, so keep only the seasons you want to train on there (anything
parked in `data/_excluded/` is ignored). To regenerate cleaned data from the raw
play-by-play in `./RawData/MasterFiles/`, add `--clean` to the run command.

---

## Train the Event/Time model (e.g. on 2003 data)

```bash
# preprocess (build vocabs + tensors) and train on the GPU:
python main.py --rebuild-vocabs --epochs 50 --batch-size 64 --train
```

`train()` prints `Training on GPU x1` when the GPU is in use. Watch utilization from another
shell with `watch -n1 nvidia-smi`. If you hit OOM on 12 GB VRAM, drop `--batch-size` to 32
(`mixed_float16` is already enabled on GPU).

Useful flags (see `main.py`): `--clean` (re-clean raw data first), `--data-dir`,
`--skip-preprocess`, `--epochs`, `--batch-size`.

---

## Saved artifacts & reloading

Training writes a self-contained, reloadable artifact under `./artifacts/<key>/` — for the
Event/Time model, `./artifacts/event_time/`:

| File                       | Contents                                   |
|----------------------------|--------------------------------------------|
| `event_time.keras`         | Full single-file Keras model               |
| `event_time.weights.h5`    | Weights only (robust reload path)          |
| `norm_stats.json`          | Time-normalization stats                   |
| `encoder/vocabs/*.json`    | Shared, frozen token vocabularies          |

Reload a trained model to run inference or rerun tests against the saved state:

```python
# Single model — rebuilds the graph from the frozen vocabs and restores weights:
from models.event_time_model import EventTimeModel
inst, model = EventTimeModel.from_artifacts("./artifacts")

# All registered models together (forward-facing manager):
from models.model_bundle import ModelBundle
bundle = ModelBundle.load("./artifacts")
model = bundle.models["event_time"]
```

The persistence layout, the model registry, and `ModelBundle` are shared across models, so
new models save/load/test the same way (add one entry to `models/registry.py`).

---

## Tests

```bash
source ~/cviq-venv/bin/activate
cd /mnt/c/Projects/CourtVisionIQ
pytest -q
```

`tests/test_model_persistence.py` trains a tiny model, reloads it (`from_artifacts`, full
`.keras`, and `ModelBundle`), and runs inference — parametrized over every model in the
registry, so new models inherit IO coverage automatically.
