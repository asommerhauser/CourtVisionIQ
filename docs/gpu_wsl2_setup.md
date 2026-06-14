# GPU Training Setup — WSL2 + Ubuntu (RTX 4070)

## Why this is needed

TensorFlow **dropped native-Windows GPU support after v2.10**. This repo runs TF
2.20 on Python 3.13, for which the only Windows wheel is **CPU-only** — so
`tf.config.list_physical_devices("GPU")` returns `[]` and the RTX 4070 is
invisible, even though the driver (CUDA 13.1) is fine.

The supported way to use the 4070 with modern TF on this machine is **WSL2 with a
real Ubuntu distro**, installing `tensorflow[and-cuda]` (bundles the matching
CUDA + cuDNN userspace libs via pip — no system CUDA toolkit needed).

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

## 3. Python environment

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
cd /mnt/c/Projects/CourtVisionIQ          # the repo, mounted from Windows
python3 -m venv .venv-linux                # separate venv from the Windows one
source .venv-linux/bin/activate
pip install --upgrade pip
```

## 4. Install GPU TensorFlow + repo deps

```bash
pip install "tensorflow[and-cuda]"         # CUDA + cuDNN bundled via pip
pip install pandas numpy pytest            # repo runtime deps
```

> If `tensorflow[and-cuda]` resolves to a different minor than 2.20, that's fine —
> keep it on a 2.x with Keras 3. Re-run the tests after install to confirm parity.

## 5. Confirm the GPU is visible to TF

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# -> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 6. Run training on the GPU

```bash
python main.py --rebuild-vocabs --epochs 50 --batch-size 64
```

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
- **Two venvs.** Keep the Windows interpreter for editing/CPU smoke tests and the
  Linux `.venv-linux` for GPU runs. They share the same source tree via the mount.
- **Don't `pip install` system CUDA.** The pip `[and-cuda]` extra is
  self-contained; mixing in a system toolkit causes version conflicts.

---

## One-time vs. per-session

- One-time: steps 1–4 (install distro, venv, TF).
- Per session: `wsl -d Ubuntu-22.04`, `cd /mnt/c/Projects/CourtVisionIQ`,
  `source .venv-linux/bin/activate`, then run `main.py`.
