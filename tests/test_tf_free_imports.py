"""
The TF-free boundary: which modules may import TensorFlow, and which may not.

Two packages hold both kinds of module. ``simulation`` holds the rollout driver (TF) alongside
``box_score`` / ``stats`` / ``eval_metrics`` / ``probes``, which are pure pandas. ``reporting``
holds the training collector (a Keras callback) alongside the whole evaluation report stack.
Before 3.0 both package ``__init__``s imported the TF side eagerly, so *touching* either package
anywhere loaded TensorFlow.

That cost is not theoretical. A TF import in a process that never runs a model still creates a
CUDA context, and these modules run beside a live eval pool -- ``scripts/peek_sims.py`` watching a
run, ``harvest.py`` pruning it, ``update_eval_report`` re-aggregating it. The VRAM comes out of
the workers. On Windows it is worse than a cost: importing pandas before TF breaks TF's native DLL
init, so a pandas-first caller did not get a slow import, it got a traceback.

Each check runs in a subprocess because ``sys.modules`` is process-global and ``conftest.py``
imports TF before anything else -- an in-process assertion would always fail.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Imported pandas-first on purpose: that is the ordering that used to be a hard failure on Windows,
# and it is the ordering every one of these callers actually uses.
_PROBE = (
    "import pandas  # noqa: F401\n"
    "import sys\n"
    "{body}\n"
    "print('tensorflow' in sys.modules or 'keras' in sys.modules)\n"
)


def _loads_tf(body: str) -> bool:
    """Run ``body`` in a fresh interpreter; report whether TF or Keras ended up loaded."""
    r = subprocess.run([sys.executable, "-c", _PROBE.format(body=body)],
                       cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip() == "True"


def test_box_score_and_stats_import_without_tensorflow():
    assert not _loads_tf(
        "from simulation.box_score import generate_box_score\n"
        "from simulation.stats import team_totals, possessions, advanced_stats"
    )


def test_eval_metrics_imports_without_tensorflow():
    """``eval_metrics`` is deliberately data-free and TF-free -- its own docstring says so."""
    assert not _loads_tf("from simulation.eval_metrics import _aggregate, win_metrics")


def test_the_evaluation_report_stack_imports_without_tensorflow():
    """Re-aggregating a finished run reads report.json and writes HTML/Parquet. No model involved."""
    assert not _loads_tf(
        "import reporting.eval_report\n"
        "import reporting.update_eval_report\n"
        "import reporting.baseline_comparison\n"
        "import reporting.query"
    )


def test_touching_either_package_does_not_import_tensorflow():
    """The regression this file exists for: a bare package import used to pull TF."""
    assert not _loads_tf("import simulation\nimport reporting")


def test_lazy_names_still_resolve():
    """Laziness must not break the public API -- both packages still export what they advertise."""
    code = (
        "import tensorflow  # entry points import TF first; see simulation/__init__\n"
        "import simulation, reporting\n"
        "assert simulation.GameSimulator.__name__ == 'GameSimulator'\n"
        "assert simulation.generate_box_score.__name__ == 'generate_box_score'\n"
        "assert reporting.ReportCollector.__name__ == 'ReportCollector'\n"
        "assert 'GameSimulator' in dir(simulation) and 'ReportCollector' in dir(reporting)\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_an_unknown_package_attribute_still_raises_attribute_error():
    """``__getattr__`` must not turn a typo into an ImportError or a silent None."""
    code = (
        "import simulation, reporting\n"
        "for pkg in (simulation, reporting):\n"
        "    try:\n"
        "        pkg.NoSuchName\n"
        "    except AttributeError:\n"
        "        pass\n"
        "    else:\n"
        "        raise SystemExit('expected AttributeError')\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
