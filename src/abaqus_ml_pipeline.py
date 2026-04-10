import os
import json
import shutil
import subprocess
import time
from multiprocessing import Pool

# ── Path anchor ───────────────────────────────────────────────────────────────
# All directories are resolved relative to the repo root (one level above src/)
# so the script is safe to call from any working directory.
_HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.normpath(os.path.join(_HERE, ".."))

VARIANTS_DIR = os.path.join(ROOT, "out_dir")
ABAQUS_JOBS  = os.path.join(ROOT, "abaqus_jobs")
RESULTS_DIR  = os.path.join(ROOT, "results_ml")
# ─────────────────────────────────────────────────────────────────────────────

# Boundary conditions
T_HOT  = 1400.0
T_COLD = 600.0

os.makedirs(ABAQUS_JOBS, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_manifest():
    """Load the variants manifest from Stage 1."""
    manifest_path = os.path.join(VARIANTS_DIR, "variants_manifest.json")
    with open(manifest_path, "r") as f:
        return json.load(f)


def run_abaqus(job_name, inp_file):
    """
    Run ABAQUS simulation for one job.
    Returns True if successful, False otherwise.
    """
    odb_path = os.path.join(ABAQUS_JOBS, f"{job_name}.odb")

    # Skip if already completed
    if os.path.exists(odb_path):
        print(f"Skipping {job_name} (already completed)")
        return True

    # Copy input file to abaqus_jobs/
    src = os.path.join(VARIANTS_DIR, inp_file)
    dst = os.path.join(ABAQUS_JOBS, inp_file)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)

    print(f"Running ABAQUS job: {job_name}")

    result = subprocess.run(
        ["abaqus", f"job={job_name}", f"input={inp_file}", "cpus=1"],
        cwd=ABAQUS_JOBS,
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print(f"ERROR: ABAQUS failed for {job_name}")
        print(result.stderr)
        return False

    # Poll for ODB completion (max 10 min)
    max_wait     = 600
    wait_interval = 5
    elapsed      = 0

    while elapsed < max_wait:
        if os.path.exists(odb_path):
            initial_size = os.path.getsize(odb_path)
            time.sleep(2)
            final_size = os.path.getsize(odb_path)
            if initial_size == final_size:
                print(f"Completed: {job_name}")
                return True

        time.sleep(wait_interval)
        elapsed += wait_interval

    print(f"ERROR: Timeout waiting for {job_name}")
    return False


def create_odb_extraction_script():
    """
    Write the ABAQUS/Python ODB extraction script to abaqus_jobs/.
    This script runs inside the ABAQUS Python interpreter.
    """
    script = """
import sys
import json
from odbAccess import openOdb

def extract_thermal_data(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step  = odb.steps[odb.steps.keys()[0]]
        frame = step.frames[-1]

        temp_field = frame.fieldOutputs['NT11']
        temps = [float(v.data) for v in temp_field.values]

        flux_field = frame.fieldOutputs['HFL']
        fluxes = []
        for v in flux_field.values:
            if hasattr(v.data, '__len__') and len(v.data) > 1:
                flux_y = float(v.data[1])   # HFL2 = through-thickness
            else:
                flux_y = float(v.data)
            fluxes.append(abs(flux_y))

        return {
            'avg_heat_flux_W_per_mm2': sum(fluxes) / len(fluxes),
            'max_heat_flux_W_per_mm2': max(fluxes),
            'max_temperature_K':       max(temps),
            'min_temperature_K':       min(temps),
            'avg_temperature_K':       sum(temps) / len(temps),
            'num_nodes':               len(temps),
            'num_elements':            len(fluxes),
        }
    finally:
        odb.close()

if __name__ == '__main__':
    odb_file    = sys.argv[1]
    output_json = sys.argv[2]
    results = extract_thermal_data(odb_file)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
"""
    script_path = os.path.join(ABAQUS_JOBS, "extract_odb.py")
    with open(script_path, "w") as f:
        f.write(script)
    return script_path


def extract_results(job_name, ysz_data):
    """
    Extract thermal results from ODB and compute effective thermal conductivity.
    """
    odb_path    = os.path.join(ABAQUS_JOBS, f"{job_name}.odb")
    result_json = os.path.join(ABAQUS_JOBS, f"{job_name}_results.json")

    if not os.path.exists(odb_path):
        print(f"ERROR: ODB file missing for {job_name}")
        return None

    print(f"Extracting results from {job_name}.odb")
    result = subprocess.run(
        ["abaqus", "python", "extract_odb.py",
         f"{job_name}.odb", f"{job_name}_results.json"],
        cwd=ABAQUS_JOBS,
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print(f"ERROR: Result extraction failed for {job_name}")
        print(result.stderr)
        return None

    with open(result_json, "r") as f:
        results = json.load(f)

    # k_eff = (q * L_total) / ΔT
    L_total  = ysz_data["thickness_mm"] + 10.0   # YSZ + CMSX-4  [mm]
    delta_T  = T_HOT - T_COLD                     # [K]
    q        = results["avg_heat_flux_W_per_mm2"]  # [W mm⁻²]

    results["effective_thermal_conductivity"] = (q * L_total) / delta_T
    results["total_thickness_mm"]             = L_total
    results["temperature_difference_K"]       = delta_T
    results["job_name"]                       = job_name

    return results


def calculate_ysz_thermal_conductivity(ysz_data):
    """Average YSZ thermal conductivity across temperature points."""
    return sum(k for _, k in ysz_data["thermal_conductivity"]) / len(ysz_data["thermal_conductivity"])


def calculate_features(ysz_data):
    """Extract ML feature dict from one YSZ variant."""
    k_ysz_avg  = calculate_ysz_thermal_conductivity(ysz_data)
    cp_ysz_avg = sum(cp for _, cp in ysz_data["specific_heat"]) / len(ysz_data["specific_heat"])

    return {
        "ysz_thickness_mm": ysz_data["thickness_mm"],
        "ysz_density":       ysz_data["density"],
        "ysz_k_avg":         k_ysz_avg,
        "ysz_cp_avg":        cp_ysz_avg,
    }


def process_job(entry):
    """Run one ABAQUS job and return a complete ML sample dict, or None on failure."""
    job_name = entry["id"]
    inp_file = entry["file"]

    if not run_abaqus(job_name, inp_file):
        return None

    results = extract_results(job_name, entry["YSZ"])
    if results is None:
        return None

    features = calculate_features(entry["YSZ"])

    return {
        "job_name":           job_name,
        "features":           features,
        "target":             results["effective_thermal_conductivity"],
        "simulation_results": results,
        "odb_file":           os.path.join(ABAQUS_JOBS, f"{job_name}.odb"),
        "inp_file":           os.path.join(VARIANTS_DIR, inp_file),
    }


def main():
    create_odb_extraction_script()

    manifest   = load_manifest()
    total_jobs = len(manifest)
    print(f"Processing {total_jobs} jobs with 4 parallel processes...")

    with Pool(processes=4) as pool:
        results_list = pool.map(process_job, manifest)

    all_data    = []
    failed_jobs = []

    for i, result in enumerate(results_list):
        if result is not None:
            all_data.append(result)
        else:
            failed_jobs.append(manifest[i]["id"])

        # Checkpoint every 50 samples
        if (i + 1) % 50 == 0:
            intermediate_path = os.path.join(
                RESULTS_DIR, f"dataset_intermediate_{i + 1}.json"
            )
            with open(intermediate_path, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"Checkpoint: {len(all_data)} samples saved")

    # Final dataset
    dataset_path = os.path.join(RESULTS_DIR, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(all_data, f, indent=2)

    # Failed jobs log
    if failed_jobs:
        failed_path = os.path.join(RESULTS_DIR, "failed_jobs.json")
        with open(failed_path, "w") as f:
            json.dump({"failed_jobs": failed_jobs}, f, indent=2)

    print(f"\n{'=' * 60}")
    print("  ML Dataset Complete")
    print(f"{'=' * 60}")
    print(f"  Successful : {len(all_data)}")
    print(f"  Failed     : {len(failed_jobs)}")
    print(f"  Dataset    : {dataset_path}")

    if all_data:
        k_vals = [d["target"] for d in all_data]
        print(f"\n  k_eff statistics:")
        print(f"    Min  : {min(k_vals):.6f}  W mm⁻¹ K⁻¹")
        print(f"    Max  : {max(k_vals):.6f}  W mm⁻¹ K⁻¹")
        print(f"    Mean : {sum(k_vals)/len(k_vals):.6f}  W mm⁻¹ K⁻¹")

        by_thickness = {}
        for d in all_data:
            th = d["features"]["ysz_thickness_mm"]
            by_thickness.setdefault(th, []).append(d["target"])

        print(f"\n  By YSZ thickness:")
        for th in sorted(by_thickness):
            vals = by_thickness[th]
            print(f"    {th} mm : {len(vals)} samples,"
                  f"  mean k_eff = {sum(vals)/len(vals):.6f}  W mm⁻¹ K⁻¹")


if __name__ == "__main__":
    main()