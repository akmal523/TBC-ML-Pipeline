import os
import json
import shutil
import subprocess
import time
from multiprocessing import Pool

# Directory configuration
VARIANTS_DIR = "out_dir"
ABAQUS_JOBS = "abaqus_jobs"
RESULTS_DIR = "results_ml"

# Boundary conditions 
T_HOT = 1400.0
T_COLD = 600.0

os.makedirs(ABAQUS_JOBS, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_manifest():
    """Load the variants manifest from Step 1."""
    manifest_path = os.path.join(VARIANTS_DIR, "variants_manifest.json")
    with open(manifest_path, "r") as f:
        return json.load(f)


def run_abaqus(job_name, inp_file):
    """
    Run ABAQUS simulation for one job.
    Returns True if successful, False otherwise.
    """
    # Skip if already completed
    odb_path = os.path.join(ABAQUS_JOBS, f"{job_name}.odb")
    if os.path.exists(odb_path):
        print(f"Skipping {job_name} (already completed)")
        return True
    
    # Copy input file to abaqus_jobs directory
    src = os.path.join(VARIANTS_DIR, inp_file)
    dst = os.path.join(ABAQUS_JOBS, inp_file)
    
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
    
    print(f"Running ABAQUS job: {job_name}")
    
    # Execute ABAQUS in background
    result = subprocess.run(
        ["abaqus", f"job={job_name}", f"input={inp_file}", "cpus=1"],
        cwd=ABAQUS_JOBS,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: ABAQUS failed for {job_name}")
        print(result.stderr)
        return False
    
    # Wait for job to complete (check for .odb file)
    max_wait = 600  # 10 minutes timeout
    wait_interval = 5  # check every 5 seconds
    elapsed = 0
    
    while elapsed < max_wait:
        if os.path.exists(odb_path):
            # Additional check: ensure file is not being written
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
    script for extracting thermal data from ODB files.
    script runs inside ABAQUS Python environment.
    """
    script = """
import sys
import json
from odbAccess import openOdb

def extract_thermal_data(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)
    
    try:
        # Get last frame (steady state solution)
        step = odb.steps[odb.steps.keys()[0]]
        frame = step.frames[-1]
        
        # Extract temperature field
        temp_field = frame.fieldOutputs['NT11']
        temps = [float(v.data) for v in temp_field.values]
        
        # Extract heat flux field
        flux_field = frame.fieldOutputs['HFL']
        fluxes = []
        for v in flux_field.values:
            # Calculate flux magnitude
            if hasattr(v.data, '__len__'):
                flux_mag = sum([float(c)**2 for c in v.data])**0.5
            else:
                flux_mag = abs(float(v.data))
            fluxes.append(flux_mag)
        
        results = {
            'avg_heat_flux_W_per_mm2': sum(fluxes) / len(fluxes),
            'max_heat_flux_W_per_mm2': max(fluxes),
            'max_temperature_K': max(temps),
            'min_temperature_K': min(temps),
            'avg_temperature_K': sum(temps) / len(temps),
            'num_nodes': len(temps),
            'num_elements': len(fluxes)
        }
        
        return results
        
    finally:
        odb.close()

if __name__ == '__main__':
    odb_file = sys.argv[1]
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
    Extract simulation results from ODB file and calculate effective thermal conductivity.
    """
    odb_path = os.path.join(ABAQUS_JOBS, f"{job_name}.odb")
    result_json = os.path.join(ABAQUS_JOBS, f"{job_name}_results.json")
    
    if not os.path.exists(odb_path):
        print(f"ERROR: ODB file missing for {job_name}")
        return None
    
    # Run extraction script with ABAQUS Python
    print(f"Extracting results from {job_name}.odb")
    result = subprocess.run(
        ["abaqus", "python", "extract_odb.py", f"{job_name}.odb", f"{job_name}_results.json"],
        cwd=ABAQUS_JOBS,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: Result extraction failed for {job_name}")
        print(result.stderr)
        return None
    
    # Load extracted results
    with open(result_json, "r") as f:
        results = json.load(f)
    
    # Calculate effective thermal conductivity
    # k_eff = (q * L) / ΔT
    # where: q = heat flux, L = total thickness, ΔT = temperature difference
    L_total = ysz_data["thickness_mm"] + 10.0  # YSZ + CMSX-4
    delta_T = T_HOT - T_COLD
    q = results["avg_heat_flux_W_per_mm2"]
    
    k_effective = (q * L_total) / delta_T
    
    results["effective_thermal_conductivity"] = k_effective
    results["total_thickness_mm"] = L_total
    results["temperature_difference_K"] = delta_T
    results["job_name"] = job_name
    
    return results


def calculate_ysz_thermal_conductivity(ysz_data):
    """
    Calculate average YSZ thermal conductivity from temperature-dependent data.
    This is a FEATURE for ML training.
    """
    k_values = [k for T, k in ysz_data["thermal_conductivity"]]
    return sum(k_values) / len(k_values)


def calculate_features(ysz_data):
    """
    Extract ML features from YSZ coating material data.
    
    Features (inputs for ML model):
    - ysz_thickness_mm: Thickness of YSZ coating layer
    - ysz_density: Density of YSZ coating (varied ±20%)
    - ysz_k_avg: Thermal conductivity of YSZ coating (varied ±20%) - KEY FEATURE
    - ysz_cp_avg: Specific heat of YSZ coating (varied ±20%)
    """
    # KEY FEATURE: YSZ coating thermal conductivity
    k_ysz_avg = calculate_ysz_thermal_conductivity(ysz_data)
    
    cp_values = [cp for T, cp in ysz_data["specific_heat"]]
    cp_ysz_avg = sum(cp_values) / len(cp_values)
    
    features = {
        "ysz_thickness_mm": ysz_data["thickness_mm"],
        "ysz_density": ysz_data["density"],
        "ysz_k_avg": k_ysz_avg,  # YSZ coating thermal conductivity
        "ysz_cp_avg": cp_ysz_avg
    }
    
    return features


def process_job(entry):
    """
    Process single job: run ABAQUS and extract results.
    Used for parallel processing.
    """
    job_name = entry["id"]
    inp_file = entry["file"]
    
    # Run ABAQUS simulation
    success = run_abaqus(job_name, inp_file)
    if not success:
        return None
    
    # Extract results
    results = extract_results(job_name, entry["YSZ"])
    if results is None:
        return None
    
    # Calculate features
    features = calculate_features(entry["YSZ"])
    
    # Create ML sample
    ml_sample = {
        "job_name": job_name,
        "features": features,
        "target": results["effective_thermal_conductivity"],
        "simulation_results": results,
        "odb_file": f"{job_name}.odb",
        "inp_file": inp_file
    }
    
    return ml_sample


def main():
    # Create ODB extraction script
    create_odb_extraction_script()
    
    # Load manifest from Step 1
    manifest = load_manifest()
    total_jobs = len(manifest)
    
    print(f"Processing {total_jobs} jobs with 8 parallel processes...")
    
    # Run jobs in parallel
    with Pool(processes=8) as pool:
        results_list = pool.map(process_job, manifest)
    
    # Filter out failed jobs
    all_data = []
    failed_jobs = []
    
    for i, result in enumerate(results_list):
        if result is not None:
            all_data.append(result)
        else:
            failed_jobs.append(manifest[i]["id"])
        
        # Save intermediate results every 50 samples
        if (i + 1) % 50 == 0:
            intermediate_path = os.path.join(RESULTS_DIR, f"dataset_intermediate_{i+1}.json")
            with open(intermediate_path, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"Saved intermediate: {len(all_data)} samples processed")
    
    # Save final ML dataset
    dataset_path = os.path.join(RESULTS_DIR, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(all_data, f, indent=2)
    
    # Save failed jobs list
    if failed_jobs:
        failed_path = os.path.join(RESULTS_DIR, "failed_jobs.json")
        with open(failed_path, "w") as f:
            json.dump({"failed_jobs": failed_jobs}, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ML Dataset Complete")
    print(f"{'='*60}")
    print(f"Successful samples: {len(all_data)}")
    print(f"Failed jobs: {len(failed_jobs)}")
    print(f"Dataset saved to: {dataset_path}")
    
    if all_data:
        # Summary statistics
        k_values = [d["target"] for d in all_data]
        print(f"\nEffective Thermal Conductivity Statistics:")
        print(f"  Min:  {min(k_values):.6f} W/mm·K")
        print(f"  Max:  {max(k_values):.6f} W/mm·K")
        print(f"  Mean: {sum(k_values)/len(k_values):.6f} W/mm·K")
        
        # Group by thickness
        by_thickness = {}
        for d in all_data:
            th = d["features"]["ysz_thickness_mm"]
            if th not in by_thickness:
                by_thickness[th] = []
            by_thickness[th].append(d["target"])
        
        print(f"\nResults by YSZ Thickness:")
        for th in sorted(by_thickness.keys()):
            vals = by_thickness[th]
            print(f"  {th} mm: {len(vals)} samples, mean k_eff = {sum(vals)/len(vals):.6f} W/mm·K")


if __name__ == "__main__":
    main()
