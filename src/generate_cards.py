import json
import random
import os
import sys


def vary(value, percent=0.20):
    """Apply ±20% random variation to a value."""
    return value * (1 + random.uniform(-percent, percent))


def pick_ysz_elastic(db):
    """Select random YSZ elastic orientation and apply variation."""
    elastic_data = db["materials"]["YSZ"]["elastic_properties"]
    orientation = random.choice([k for k in elastic_data.keys() if k.startswith("orientation_")])
    
    varied_elastic = []
    for entry in elastic_data[orientation]:
        varied_elastic.append({
            "T_K": entry["T_K"],
            "E_MPa": vary(entry["E_MPa"]),
            "nu": vary(entry["nu"])
        })
    
    return orientation, varied_elastic


def pick_ysz_thermal_conductivity(db):
    """Select random YSZ thermal conductivity curve (dense or porous)."""
    k_data = db["materials"]["YSZ"]["thermal_conductivity_W_per_mmK"]
    
    # Randomly choose dense or porous
    mode = random.choice(["dense", "porous"])
    curve = random.choice(list(k_data[mode].values()))
    
    # Apply variation to each temperature point
    return [[pt["T_K"], vary(pt["k"])] for pt in curve]


def pick_ysz_variant(db, thickness_mm):
    """Generate one randomized YSZ material variant."""
    base_density = db["materials"]["YSZ"]["density_tonne_per_mm3"]["data"]
    
    # Vary specific heat
    varied_cp = []
    for entry in db["materials"]["YSZ"]["specific_heat_mJ_per_tonneK"]["data"]:
        varied_cp.append([entry["T_K"], vary(entry["Cp"])])
    
    orientation, elastic = pick_ysz_elastic(db)
    
    return {
        "material_name": "YSZ",
        "thickness_mm": thickness_mm,
        "density": vary(base_density),
        "elastic_orientation": orientation,
        "elastic": elastic,
        "thermal_conductivity": pick_ysz_thermal_conductivity(db),
        "specific_heat": varied_cp
    }


def get_cmsx4_properties(db):
    """Extract CMSX-4 properties (unchanged)."""
    cmsx = db["materials"]["CMSX-4"]
    return {
        "material_name": "CMSX-4",
        "density": cmsx["density_tonne_per_mm3"]["data"],
        "k": cmsx["thermal_conductivity_W_per_mmK"]["data"],
        "elastic_001": cmsx["elastic_properties"]["orientation_001"]["data"],
        "elastic_101": cmsx["elastic_properties"]["orientation_101"]["data"],
        "elastic_111": cmsx["elastic_properties"]["orientation_111"]["data"],
        "Cp": cmsx["specific_heat_capacity_mJ_per_tonneK"]["data"]
    }


def generate_abaqus_input(ysz, cmsx, job_name, T_hot=1400.0, T_cold=600.0):
    """
    Generate ABAQUS input file for thermal barrier coating simulation.
    
    Single part with two material sections:
    - YSZ layer (bottom, hot side)
    - CMSX-4 layer (top, cold side)
    """
    lines = []
    
    # Header
    lines.append("*Heading")
    lines.append(f"** Job: {job_name}, YSZ thickness: {ysz['thickness_mm']} mm")
    lines.append("*Preprint, echo=NO, model=NO, history=NO")
    lines.append("**")
    
    # Geometry: YSZ on bottom, CMSX-4 on top
    ysz_thick = ysz['thickness_mm']
    cmsx_thick = 10.0
    y_interface = ysz_thick
    y_top = ysz_thick + cmsx_thick
    
    # Part definition
    lines.append("** PART: Composite")
    lines.append("*Part, name=Composite")
    lines.append("*Node")
    lines.append("      1,           0.,           0.")
    lines.append("      2,          10.,           0.")
    lines.append(f"      3,          10., {y_interface:12.6f}")
    lines.append(f"      4,           0., {y_interface:12.6f}")
    lines.append(f"      5,          10., {y_top:12.6f}")
    lines.append(f"      6,           0., {y_top:12.6f}")
    
    lines.append("*Element, type=DC2D4")
    lines.append("1, 1, 2, 3, 4")
    lines.append("2, 4, 3, 5, 6")
    
    lines.append("*Elset, elset=YSZ_Elements")
    lines.append(" 1,")
    lines.append("*Elset, elset=CMSX4_Elements")
    lines.append(" 2,")
    
    lines.append("*Nset, nset=HotSide")
    lines.append(" 1, 2")
    lines.append("*Nset, nset=ColdSide")
    lines.append(" 5, 6")
    lines.append("*Nset, nset=AllNodes")
    lines.append(" 1, 2, 3, 4, 5, 6")
    
    lines.append("** Section: YSZ_Section")
    lines.append("*Solid Section, elset=YSZ_Elements, material=YSZ")
    lines.append("1.,")
    lines.append("** Section: CMSX4_Section")
    lines.append("*Solid Section, elset=CMSX4_Elements, material=CMSX4")
    lines.append("1.,")
    lines.append("*End Part")
    lines.append("**")
    
    # Assembly
    lines.append("** ASSEMBLY")
    lines.append("*Assembly, name=Assembly")
    lines.append("*Instance, name=Composite-1, part=Composite")
    lines.append("*End Instance")
    lines.append("*Nset, nset=HotBC, instance=Composite-1")
    lines.append(" HotSide,")
    lines.append("*Nset, nset=ColdBC, instance=Composite-1")
    lines.append(" ColdSide,")
    lines.append("*End Assembly")
    lines.append("**")
    
    # Material: CMSX-4
    lines.append("** MATERIAL: CMSX4")
    lines.append("*Material, name=CMSX4")
    lines.append("*Density")
    lines.append(f"{cmsx['density'][0][1]:.6e},")
    
    # Correction: Removed dependencies=1 and fixed string formatting
    lines.append("*Conductivity")
    for T, k in cmsx["k"]:
        lines.append(f"{k:.6e}, {T:.2f}")
        
    lines.append("*Specific Heat")
    for T, cp in cmsx["Cp"]:
        lines.append(f"{cp:.6e}, {T:.2f}")
    lines.append("**")
    
    # Material: YSZ
    lines.append("** MATERIAL: YSZ")
    lines.append("*Material, name=YSZ")
    lines.append("*Density")
    lines.append(f"{ysz['density']:.6e},")
    
    # Correction: Removed dependencies=1 and fixed string formatting
    lines.append("*Conductivity")
    for T, k in sorted(ysz["thermal_conductivity"], key=lambda x: x[0]):
        lines.append(f"{k:.6e}, {T:.2f}")
        
    lines.append("*Specific Heat")
    for T, cp in sorted(ysz["specific_heat"], key=lambda x: x[0]):
        lines.append(f"{cp:.6e}, {T:.2f}")
    lines.append("**")
    
    # Initial conditions
    lines.append("** INITIAL CONDITIONS")
    lines.append("*Initial Conditions, type=TEMPERATURE")
    lines.append("Composite-1.AllNodes, 300.")
    lines.append("**")
    
    # Heat transfer step
    lines.append("** STEP: HeatTransfer")
    lines.append("*Step, name=HeatTransfer, nlgeom=NO")
    lines.append("*Heat Transfer, steady state")
    lines.append("1., 1., 1e-05, 1.")
    lines.append("**")
    lines.append("** BOUNDARY CONDITIONS")
    lines.append("*Boundary")
    lines.append(f"HotBC, 11, 11, {T_hot:.1f}")
    lines.append(f"ColdBC, 11, 11, {T_cold:.1f}")
    lines.append("**")
    lines.append("** OUTPUT REQUESTS")
    lines.append("*Output, field")
    lines.append("*Node Output")
    lines.append("NT,")
    lines.append("*Element Output")
    lines.append("HFL,")
    lines.append("*Output, history")
    lines.append("*Node Output, nset=ColdBC")
    lines.append("NT,")
    lines.append("*End Step")
    
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_cards.py materials.json out_dir")
        sys.exit(1)
    
    json_file = sys.argv[1]
    out_dir = sys.argv[2]
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load material database
    with open(json_file, "r") as f:
        db = json.load(f)
    
    # Get CMSX-4 properties (constant for all models)
    cmsx = get_cmsx4_properties(db)
    
    # Generate 512 variants: 4 thicknesses × 128 samples
    thicknesses = [0.5, 1.0, 1.5, 2.0]
    samples_per_thickness = 128
    
    manifest = []
    counter = 1
    
    for thickness in thicknesses:
        for _ in range(samples_per_thickness):
            # Generate randomized YSZ variant
            ysz = pick_ysz_variant(db, thickness)
            
            # Create job name and file path
            job_name = f"YSZ_var_{counter:03d}"
            inp_file = f"{job_name}.inp"
            inp_path = os.path.join(out_dir, inp_file)
            
            # Write ABAQUS input file
            with open(inp_path, "w") as f:
                # Change T_hot and T_cold here if needed
                f.write(generate_abaqus_input(ysz, cmsx, job_name, T_hot=1400.0, T_cold=600.0))
            
            # Add to manifest
            manifest.append({
                "id": job_name,
                "file": inp_file,
                "YSZ": ysz,
                "CMSX4": cmsx
            })
            
            counter += 1
    
    # Write manifest
    manifest_path = os.path.join(out_dir, "variants_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated {len(manifest)} ABAQUS input files")
    print(f"Output directory: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
