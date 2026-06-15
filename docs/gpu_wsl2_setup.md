# GPU Training Setup — WSL2 + Ubuntu (RTX 4070)

## Why this is needed

TensorFlow **dropped native-Windows GPU support after v2.10**. This repo runs TF
2.20 on Python 3.13, for which the only Windows wheel is **CPU-only** — so
`tf.config.list_physical_devices("GPU")` returns `[]` and the RTX 4070 is
invisible, even though the driver (CUDA 13.1) is fine.

The supported way to use the 4070 with modern TF on this machine is **WSL2 with a
real Ubuntu distro**, installing `tensorflow[and-cuda]` (bundles the matching
CUDA + cuDNN userspace libs via pip — no system CUDA toolkit needed). There is **no
native-Windows GPU path** for this TF version — don't chase one.

> **Critical pitfall (this is what broke the first attempt):** create the venv in
> the **Linux filesystem** (e.g. `~/cviq-venv`), **never under `/mnt/c`**. A venv
> built on the Windows mount comes up missing its `bin/activate` and packages — it
> looks created but is unusable. Keep the *source tree* on `/mnt/c` (shared with
> Windows); put the *venv* in `~`.

The training code is already GPU-ready: `EventTimeModel.configure_gpu()` enables
per-GPU memory growth and `mixed_float16`, and the output heads are forced to
float32. Once TF can see the GPU, `train()` uses it automatically.

---

## Prerequisites (Windows side — leave as-is)

- **Keep the existing Windows NVIDIA driver.** WSL2 CUDA uses the Windows driver
  through `/usr/lib/wsl/lib`. **Do NOT install an NVIDIA driver inside WSL** — it
  breaks the GPU passthrough.
- Windows 11 (already met) — WSL2 GPU compute works out of the box.

---

## 0. Give WSL enough RAM (required)

Training compiles a large graph and needs more host RAM than the WSL2 default cap
(~50% of host). On a 31 GB machine the default ~15.5 GB cap **OOM-kills training**
(`dmesg` shows `Out of memory: Killed process … python`). Create
`C:\Users\<you>\.wslconfig` (Windows side):

```ini
[wsl2]
memory=24GB
swap=8GB
```

Then apply it: `wsl --shutdown` in PowerShell, reopen the distro, and confirm with
`free -h` (Mem total should read ~23–24 GB). Tune `memory` to leave headroom for
Windows.

## 1. Install a real Ubuntu distro

Only the `docker-desktop` WSL distro exists today; that one is not for general
use. Install Ubuntu (PowerShell, as your normal user):

```powershell
wsl --install -d Ubuntu-22.04
```

First launch prompts for a UNIX username/password. (A reboot may be required the
first time WSL2 is enabled.)

Verify it's WSL **2**:

```powershell
wsl -l -v        # Ubuntu-22.04 should show VERSION 2
```

## 2. Verify GPU passthrough inside Ubuntu

```bash
nvidia-smi       # should list the RTX 4070 — driver comes from Windows
```

If this fails, stop here — fix the driver/passthrough before installing TF.

## 3. Python environment (venv in the LINUX filesystem)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/cviq-venv                 # in $HOME, NOT under /mnt/c
source ~/cviq-venv/bin/activate
pip install --upgrade pip
```

## 4. Install GPU TensorFlow + repo deps

```bash
pip install "tensorflow[and-cuda]==2.20.0"  # CUDA + cuDNN bundled via pip
pip install pandas numpy pytest             # repo runtime deps
```

> Pinned to 2.20.0 to match `keras==3.12.0`. If you let `[and-cuda]` float to a
> different 2.x minor, keep it on a Keras-3 release and re-run `pytest` to confirm
> parity.

## 5. Confirm the GPU is visible to TF

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# -> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 6. Run training on the GPU

```bash
python main.py --rebuild-vocabs --epochs 50 --batch-size 64 --train
```

Trained artifacts land in `./artifacts/event_time/` (`event_time.keras`,
`event_time.weights.h5`, `norm_stats.json`) and reload via
`EventTimeModel.from_artifacts("./artifacts")` or `ModelBundle.load("./artifacts")`.

`train()` will print `Training on GPU x1`. Watch utilization from Windows with
`nvidia-smi -l 1` (or `watch -n1 nvidia-smi` inside WSL).

---

## Gotchas / tuning

- **File I/O across `/mnt/c` is slow.** Reading the raw CSVs and writing
  `.npz`/artifacts over the Windows mount adds overhead. For the heavy cleaning
  step, consider copying `RawData/` into the Linux filesystem (e.g. `~/data`) and
  pointing `--data-dir` there; training itself is compute-bound so the mount is
  usually fine.
- **VRAM (12 GB).** `configure_gpu()` already sets memory growth. If you hit OOM
  at batch 64, drop to 32. `mixed_float16` (on by default on GPU) roughly halves
  activation memory and uses Tensor Cores.
- **XLA.** Try `model.train(..., jit_compile=True)` for a possible speedup once a
  baseline run is stable; it can be finicky with dynamic shapes, so validate
  numerics first.
- **venv in `~`, source on `/mnt/c`.** The Linux-fs `~/cviq-venv` is the GPU
  interpreter; the repo stays on the Windows mount so you can edit it from Windows.
  Do **not** put the venv on `/mnt/c` (see the pitfall callout above).
- **Don't `pip install` system CUDA.** The pip `[and-cuda]` extra is
  self-contained; mixing in a system toolkit causes version conflicts.

---

## One-time vs. per-session

- One-time: steps 1–4 (install distro, venv, TF).
- Per session: `wsl -d Ubuntu-22.04`, `source ~/cviq-venv/bin/activate`,
  `cd /mnt/c/Projects/CourtVisionIQ`, then run `main.py`.
