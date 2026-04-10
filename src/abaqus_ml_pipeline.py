import os
import json
import shutil
import subprocess
import time
from multiprocessing import Pool


# ── Path anchor ───────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.normpath(os.path.join(_HERE, ".."))

VARIANTS_DIR = os.path.join(ROOT, "out_dir")
ABAQUS_JOBS  = os.path.join(ROOT, "abaqus_jobs")
RESULTS_DIR  = os.path.join(ROOT, "results_ml")
# ─────────────────────────────────────────────────────────────────────────────

T_HOT  = 1400.0
T_COLD =  600.0

os.makedirs(ABAQUS_JOBS, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_manifest():
    manifest_path = os.path.join(VARIANTS_DIR, "variants_manifest.json")
    with open(manifest_path, "r") as f:
        return json.load(f)


def run_abaqus(job_name, inp_file):
    odb_path = os.path.join(ABAQUS_JOBS, f"{job_name}.odb")

    if os.path.exists(odb_path):
        print(f"Skipping {job_name} (already completed)")
        return True

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

    max_wait      = 600
    wait_interval = 5
    elapsed       = 0

    while elapsed < max_wait:
        if os.path.exists(odb_path):
            initial_size = os.path.getsize(odb_path)
            time.sleep(2)
            final_size   = os.path.getsize(odb_path)
            if initial_size == final_size:
                print(f"Completed: {job_name}")
                return True

        time.sleep(wait_interval)
        elapsed += wait_interval

    print(f"ERROR: Timeout waiting for {job_name}")
    return False


def create_odb_extraction_script():
    """
    Write the ABAQUS/Python ODB extraction script.

    Flux extraction uses area-weighted Gauss-point averaging:
    ─────────────────────────────────────────────────────────
      q̄'' = Σ_e Σ_g ( w_g · HFL2_e,g · |J_e,g| )
             ────────────────────────────────────────
             Σ_e Σ_g ( w_g · |J_e,g| )

    For DC2D4 elements with 2×2 Gauss quadrature, w_g = 1 and
    |J_e,g| = element_area / 4 (constant for rectangular elements).
    The expression therefore reduces to:
      q̄'' = Σ_e ( A_e · mean_HFL2_e ) / Σ_e A_e

    where A_e is the element area computed from node coordinates
    read directly from the ODB.  This correctly weights the thin
    YSZ elements and the CMSX-4 elements by their physical area,
    eliminating the bias introduced by the previous simple count
    average (which over-weighted YSZ integration points by the
    ratio N_YSZ_pts / N_CMSX4_pts independent of element height).

    HFL2 is the through-thickness (y-direction) flux component,
    accessed as v.data[1].  HFL1 (lateral, x-direction) is v.data[0]
    and is identically zero in a 1D through-y problem.  Using HFL1
    or the vector magnitude sqrt(HFL1²+HFL2²) instead of HFL2 was
    the bug in the original code; the magnitude is numerically equal
    to HFL2 only in this strictly 1D case and must not be relied upon
    for any 2D or 3D extension.
    """
    script = r"""
import sys
import json
from collections import defaultdict
from odbAccess import openOdb


def extract_thermal_data(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step_name = list(odb.steps.keys())[0]
        step      = odb.steps[step_name]
        frame     = step.frames[-1]

        # ── Nodal temperatures ──────────────────────────────────────────────
        temp_field = frame.fieldOutputs['NT11']
        temps = [float(v.data) for v in temp_field.values]

        # ── Build node-coordinate map from the assembly ──────────────────────
        inst_name = list(odb.rootAssembly.instances.keys())[0]
        instance  = odb.rootAssembly.instances[inst_name]

        node_coords = {}
        for n in instance.nodes:
            node_coords[n.label] = n.coordinates   # (x, y, z)

        # ── Compute element areas from node coordinates ───────────────────────
        # DC2D4 connectivity order: n1(BL), n2(BR), n3(TR), n4(TL) (CCW).
        # For rectangular elements: area = width * height.
        element_area = {}
        for el in instance.elements:
            conn = el.connectivity      # tuple of 4 node labels (1-indexed)
            n1   = node_coords[conn[0]]  # bottom-left  (x, y, z)
            n2   = node_coords[conn[1]]  # bottom-right
            n4   = node_coords[conn[3]]  # top-left
            width  = abs(n2[0] - n1[0])
            height = abs(n4[1] - n1[1])
            element_area[el.label] = width * height

        # ── HFL2 (through-thickness component) at integration points ─────────
        # Group by element label, then form element-level mean HFL2.
        flux_field     = frame.fieldOutputs['HFL']
        element_hfl2   = defaultdict(list)

        for v in flux_field.values:
            # v.data is a sequence: (HFL1, HFL2) for 2D problems.
            # HFL2 = index 1 = through-thickness (y) component.
            if hasattr(v.data, '__len__') and len(v.data) > 1:
                hfl2 = float(v.data[1])
            else:
                # Scalar field output (should not occur for HFL, but handle
                # gracefully so the pipeline never silently uses the wrong
                # component).
                raise RuntimeError(
                    "HFL field output has unexpected scalar shape; "
                    "expected a 2-component vector.  Check the *Element Output "
                    "request in the .inp file (position=INTEGRATION POINT)."
                )
            element_hfl2[v.elementLabel].append(hfl2)

        # ── Area-weighted average of HFL2 ─────────────────────────────────────
        total_weighted_flux = 0.0
        total_area          = 0.0

        for el_label, hfl2_list in element_hfl2.items():
            a    = element_area.get(el_label, 0.0)
            mean = sum(hfl2_list) / len(hfl2_list)
            total_weighted_flux += mean * a
            total_area          += a

        if total_area == 0.0:
            raise RuntimeError("Total element area is zero; ODB may be empty.")

        avg_flux = total_weighted_flux / total_area

        return {
            'avg_heat_flux_W_per_mm2': avg_flux,
            'max_heat_flux_W_per_mm2': max(
                abs(v) for lst in element_hfl2.values() for v in lst
            ),
            'max_temperature_K':  max(temps),
            'min_temperature_K':  min(temps),
            'avg_temperature_K':  sum(temps) / len(temps),
            'num_nodes':          len(temps),
            'num_elements':       len(element_hfl2),
            'total_area_mm2':     total_area,
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

    # k_eff = (q̄'' · L_total) / ΔT
    # L_total = YSZ thickness + CMSX-4 thickness [mm]
    # q̄''    = area-weighted through-thickness heat flux [W mm⁻²]
    # ΔT     = T_hot − T_cold [K]
    L_total = ysz_data["thickness_mm"] + 10.0
    delta_T = T_HOT - T_COLD
    q       = results["avg_heat_flux_W_per_mm2"]

    results["effective_thermal_conductivity"] = (q * L_total) / delta_T
    results["total_thickness_mm"]             = L_total
    results["temperature_difference_K"]       = delta_T
    results["job_name"]                       = job_name

    return results


def calculate_ysz_thermal_conductivity(ysz_data):
    return (
        sum(k for _, k in ysz_data["thermal_conductivity"])
        / len(ysz_data["thermal_conductivity"])
    )


def calculate_features(ysz_data):
    k_ysz_avg  = calculate_ysz_thermal_conductivity(ysz_data)
    cp_ysz_avg = (
        sum(cp for _, cp in ysz_data["specific_heat"])
        / len(ysz_data["specific_heat"])
    )

    return {
        "ysz_thickness_mm": ysz_data["thickness_mm"],
        "ysz_density":      ysz_data["density"],
        "ysz_k_avg":        k_ysz_avg,
        "ysz_cp_avg":       cp_ysz_avg,
    }


def process_job(entry):
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

        if (i + 1) % 50 == 0:
            intermediate_path = os.path.join(
                RESULTS_DIR, f"dataset_intermediate_{i + 1}.json"
            )
            with open(intermediate_path, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"Checkpoint: {len(all_data)} samples saved")

    dataset_path = os.path.join(RESULTS_DIR, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(all_data, f, indent=2)

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
            print(
                f"    {th} mm : {len(vals)} samples, "
                f" mean k_eff = {sum(vals)/len(vals):.6f}  W mm⁻¹ K⁻¹"
            )


if __name__ == "__main__":
    main()