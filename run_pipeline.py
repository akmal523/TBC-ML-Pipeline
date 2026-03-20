"""
run_pipeline.py — TBC-ML-Pipeline entry point

Automatically detects whether ABAQUS is available.
  - ABAQUS found    → full pipeline: generate cards → simulate → train model
  - ABAQUS not found → skip to ML training using the pre-built dataset.json
"""

import os
import sys
import subprocess
import shutil


# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
SRC         = os.path.join(ROOT, "src")
DATA        = os.path.join(ROOT, "data")
RESULTS     = os.path.join(ROOT, "results")
OUT_DIR     = os.path.join(ROOT, "out_dir")
RESULTS_ML  = os.path.join(ROOT, "results_ml")
DATASET     = os.path.join(RESULTS_ML, "dataset.json")
PREBUILT    = os.path.join(RESULTS, "dataset.json")   # committed to the repo


def banner(text):
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def check_abaqus() -> bool:
    """Return True if the 'abaqus' executable is on PATH."""
    return shutil.which("abaqus") is not None


def run(cmd: list, cwd: str = ROOT):
    """Run a command, stream output live, raise on failure."""
    print(f"\n> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def ensure_dataset():
    """
    Make sure results_ml/dataset.json exists.
    If not, copy from results/dataset.json (pre-built).
    """
    os.makedirs(RESULTS_ML, exist_ok=True)
    if os.path.exists(DATASET):
        print(f"  Dataset already present: {DATASET}")
        return
    if os.path.exists(PREBUILT):
        shutil.copy(PREBUILT, DATASET)
        print(f"  Pre-built dataset copied to: {DATASET}")
    else:
        print("\nERROR: No dataset found.")
        print("  Expected one of:")
        print(f"    {DATASET}")
        print(f"    {PREBUILT}")
        print("\n  Options:")
        print("    1. Install ABAQUS and run the full pipeline.")
        print("    2. Download dataset.json from the GitHub repository")
        print("       and place it in the results/ folder.")
        sys.exit(1)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    banner("TBC-ML-Pipeline")

    abaqus_available = check_abaqus()

    if abaqus_available:
        print("  ABAQUS detected — running full pipeline.\n")
        print("  Stages:")
        print("    [1] Generate ABAQUS input cards")
        print("    [2] Run simulations and extract results")
        print("    [3] Train ML model")
    else:
        print("  ABAQUS not found — skipping simulation stages.")
        print("  Pre-built dataset will be used for ML training.\n")
        print("  Stages:")
        print("    [1] Generate ABAQUS input cards  →  SKIPPED")
        print("    [2] Run simulations               →  SKIPPED")
        print("    [3] Train ML model                →  will run")

    # ── Stage 1: Generate cards ──────────────────────────
    banner("Stage 1 — Generate ABAQUS Input Cards")

    if abaqus_available:
        run([
            sys.executable,
            os.path.join(SRC, "generate_cards.py"),
            os.path.join(DATA, "materials.json"),
            OUT_DIR
        ])
    else:
        print("  Skipped (ABAQUS not available).")

    # ── Stage 2: Run ABAQUS pipeline ─────────────────────
    banner("Stage 2 — Simulate and Extract Results")

    if abaqus_available:
        run([sys.executable, os.path.join(SRC, "abaqus_ml_pipeline.py")])
    else:
        print("  Skipped (ABAQUS not available).")
        print("  Loading pre-built dataset instead...")
        ensure_dataset()

    # ── Stage 3: Train ML model ───────────────────────────
    banner("Stage 3 — Train ML Model")
    run([sys.executable, os.path.join(SRC, "train_model.py")])

    # ── Done ──────────────────────────────────────────────
    banner("Pipeline Complete")
    print("  Output files:")
    print(f"    {os.path.join(RESULTS_ML, 'dataset.json')}")
    print(f"    {os.path.join(ROOT,       'ml_results.json')}")
    print(f"    {os.path.join(ROOT,       'ml_results.png')}")
    print()


if __name__ == "__main__":
    main()
