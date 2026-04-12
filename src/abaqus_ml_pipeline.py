import os
import json
import shutil
import subprocess
import time
from multiprocessing import Pool

# -- Path anchor ---------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.normpath(os.path.join(_HERE, ".."))

VARIANTS_DIR = os.path.join(ROOT, "out_dir")
ABAQUS_JOBS  = os.path.join(ROOT, "abaqus_jobs")
RESULTS_DIR  = os.path.join(ROOT, "results_ml")
# ------------------------------------------------------------------------------

T_HOT  = 1400.0
T_COLD =  600.0

os.makedirs(ABAQUS_JOBS, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_manifest():
    manifest_path = os.path.join(VARIANTS_DIR, "variants_manifest.json")
    with open(manifest_path, "r") as f:
        return json.load(f)


def run_abaqus(job_name, inp_file):
    odb_path = os.path.join(ABAQUS_JOBS, "%s.odb" % job_name)

    if os.path.exists(odb_path):
        print("Skipping %s (already completed)" % job_name)
        return True

    src = os.path.join(VARIANTS_DIR, inp_file)
    dst = os.path.join(ABAQUS_JOBS, inp_file)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)

    print("Running ABAQUS job: %s" % job_name)

    result = subprocess.run(
        ["abaqus", "job=%s" % job_name, "input=%s" % inp_file, "cpus=1"],
        cwd=ABAQUS_JOBS,
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print("ERROR: ABAQUS failed for %s" % job_name)
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
                print("Completed: %s" % job_name)
                return True

        time.sleep(wait_interval)
        elapsed += wait_interval

    print("ERROR: Timeout waiting for %s" % job_name)
    return False


def create_odb_extraction_script():
    """
    Write the ABAQUS/Python ODB extraction script to abaqus_jobs/.

    Flux extraction uses area-weighted Gauss-point averaging:
      q_avg = sum_e( A_e * mean_HFL2_e ) / sum_e( A_e )
    where A_e is computed from node coordinates read from the ODB.

    HFL2 (v.data[1]) is the through-thickness (y) component.
    HFL1 (v.data[0]) is the lateral component, identically zero in
    a 1D through-y problem and must not be used for k_eff extraction.

    The script string contains only ASCII characters to avoid
    UnicodeEncodeError when the ABAQUS Windows Python (cp1252) writes it.
    """
    lines = [
        "import sys",
        "import json",
        "from collections import defaultdict",
        "from odbAccess import openOdb",
        "",
        "",
        "def extract_thermal_data(odb_path):",
        "    odb = openOdb(path=odb_path, readOnly=True)",
        "    try:",
        "        step_name = list(odb.steps.keys())[0]",
        "        step      = odb.steps[step_name]",
        "        frame     = step.frames[-1]",
        "",
        "        # Nodal temperatures.",
        "        temp_field = frame.fieldOutputs['NT11']",
        "        temps = [float(v.data) for v in temp_field.values]",
        "",
        "        # Node coordinate map.",
        "        inst_name = list(odb.rootAssembly.instances.keys())[0]",
        "        instance  = odb.rootAssembly.instances[inst_name]",
        "",
        "        node_coords = {}",
        "        for n in instance.nodes:",
        "            node_coords[n.label] = n.coordinates",
        "",
        "        # Element areas from node coordinates.",
        "        # DC2D4 connectivity (CCW): n1=BL, n2=BR, n3=TR, n4=TL.",
        "        # area = width * height for rectangular elements.",
        "        element_area = {}",
        "        for el in instance.elements:",
        "            conn   = el.connectivity",
        "            n1     = node_coords[conn[0]]",
        "            n2     = node_coords[conn[1]]",
        "            n4     = node_coords[conn[3]]",
        "            width  = abs(n2[0] - n1[0])",
        "            height = abs(n4[1] - n1[1])",
        "            element_area[el.label] = width * height",
        "",
        "        # HFL2 = index 1 = through-thickness (y) flux component.",
        "        # Group integration-point values by element label.",
        "        flux_field   = frame.fieldOutputs['HFL']",
        "        element_hfl2 = defaultdict(list)",
        "",
        "        for v in flux_field.values:",
        "            if hasattr(v.data, '__len__') and len(v.data) > 1:",
        "                hfl2 = float(v.data[1])",
        "            else:",
        "                raise RuntimeError(",
        "                    'HFL has unexpected scalar shape. '",
        "                    'Check *Element Output position=CENTROIDAL.'",
        "                )",
        "            element_hfl2[v.elementLabel].append(hfl2)",
        "",
        "        # Area-weighted average of HFL2.",
        "        total_weighted = 0.0",
        "        total_area     = 0.0",
        "        for el_label, hfl2_list in element_hfl2.items():",
        "            a    = element_area.get(el_label, 0.0)",
        "            mean = sum(hfl2_list) / len(hfl2_list)",
        "            total_weighted += mean * a",
        "            total_area     += a",
        "",
        "        if total_area == 0.0:",
        "            raise RuntimeError('Total element area is zero.')",
        "",
        "        avg_flux = total_weighted / total_area",
        "        all_hfl2 = [v for lst in element_hfl2.values() for v in lst]",
        "",
        "        return {",
        "            'avg_heat_flux_W_per_mm2': avg_flux,",
        "            'max_heat_flux_W_per_mm2': max(abs(v) for v in all_hfl2),",
        "            'max_temperature_K':       max(temps),",
        "            'min_temperature_K':       min(temps),",
        "            'avg_temperature_K':       sum(temps) / len(temps),",
        "            'num_nodes':               len(temps),",
        "            'num_elements':            len(element_hfl2),",
        "            'total_area_mm2':          total_area,",
        "        }",
        "    finally:",
        "        odb.close()",
        "",
        "",
        "if __name__ == '__main__':",
        "    odb_file    = sys.argv[1]",
        "    output_json = sys.argv[2]",
        "    results = extract_thermal_data(odb_file)",
        "    with open(output_json, 'w') as f:",
        "        json.dump(results, f, indent=2)",
        "",
    ]

    script_path = os.path.join(ABAQUS_JOBS, "extract_odb.py")
    # Explicit ASCII encoding prevents UnicodeEncodeError on Windows
    # ABAQUS Python (cp1252 locale).
    with open(script_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
    return script_path


def extract_results(job_name, ysz_data):
    odb_path    = os.path.join(ABAQUS_JOBS, "%s.odb" % job_name)
    result_json = os.path.join(ABAQUS_JOBS, "%s_results.json" % job_name)

    if not os.path.exists(odb_path):
        print("ERROR: ODB file missing for %s" % job_name)
        return None

    print("Extracting results from %s.odb" % job_name)
    result = subprocess.run(
        ["abaqus", "python", "extract_odb.py",
         "%s.odb" % job_name, "%s_results.json" % job_name],
        cwd=ABAQUS_JOBS,
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print("ERROR: Result extraction failed for %s" % job_name)
        print(result.stderr)
        return None

    with open(result_json, "r") as f:
        results = json.load(f)

    # k_eff = (q_avg * L_total) / delta_T
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
        "odb_file":           os.path.join(ABAQUS_JOBS, "%s.odb" % job_name),
        "inp_file":           os.path.join(VARIANTS_DIR, inp_file),
    }


def main():
    create_odb_extraction_script()

    manifest   = load_manifest()
    total_jobs = len(manifest)
    print("Processing %d jobs with 4 parallel processes..." % total_jobs)

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
                RESULTS_DIR, "dataset_intermediate_%d.json" % (i + 1)
            )
            with open(intermediate_path, "w") as f:
                json.dump(all_data, f, indent=2)
            print("Checkpoint: %d samples saved" % len(all_data))

    dataset_path = os.path.join(RESULTS_DIR, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(all_data, f, indent=2)

    if failed_jobs:
        failed_path = os.path.join(RESULTS_DIR, "failed_jobs.json")
        with open(failed_path, "w") as f:
            json.dump({"failed_jobs": failed_jobs}, f, indent=2)

    print("\n" + "=" * 60)
    print("  ML Dataset Complete")
    print("=" * 60)
    print("  Successful : %d" % len(all_data))
    print("  Failed     : %d" % len(failed_jobs))
    print("  Dataset    : %s" % dataset_path)

    if all_data:
        k_vals = [d["target"] for d in all_data]
        print("\n  k_eff statistics:")
        print("    Min  : %.6f  W/mm/K" % min(k_vals))
        print("    Max  : %.6f  W/mm/K" % max(k_vals))
        print("    Mean : %.6f  W/mm/K" % (sum(k_vals) / len(k_vals)))

        by_thickness = {}
        for d in all_data:
            th = d["features"]["ysz_thickness_mm"]
            by_thickness.setdefault(th, []).append(d["target"])

        print("\n  By YSZ thickness:")
        for th in sorted(by_thickness):
            vals = by_thickness[th]
            print(
                "    %.1f mm : %d samples,  mean k_eff = %.6f  W/mm/K"
                % (th, len(vals), sum(vals) / len(vals))
            )


if __name__ == "__main__":
    main()