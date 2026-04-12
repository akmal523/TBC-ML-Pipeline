import json
import random
import os
import sys

# Number of elements through each layer.
# YSZ: 4 elements
# CMSX-4: 10 elements
N_YSZ  = 4
N_SUB  = 10


def vary(value, percent=0.20):
    """Apply +/-20% random variation to a value."""
    return value * (1 + random.uniform(-percent, percent))


def pick_ysz_elastic(db):
    """Select random YSZ elastic orientation and apply variation."""
    elastic_data = db["materials"]["YSZ"]["elastic_properties"]
    orientation = random.choice(
        [k for k in elastic_data.keys() if k.startswith("orientation_")]
    )

    varied_elastic = []
    for entry in elastic_data[orientation]:
        varied_elastic.append(
            {
                "T_K": entry["T_K"],
                "E_MPa": vary(entry["E_MPa"]),
                "nu": vary(entry["nu"]),
            }
        )

    return orientation, varied_elastic


def pick_ysz_thermal_conductivity(db):
    """Select random YSZ thermal conductivity curve (dense or porous)."""
    k_data = db["materials"]["YSZ"]["thermal_conductivity_W_per_mmK"]

    mode  = random.choice(["dense", "porous"])
    curve = random.choice(list(k_data[mode].values()))

    return [[pt["T_K"], vary(pt["k"])] for pt in curve]


def pick_ysz_variant(db, thickness_mm):
    """Generate one randomised YSZ material variant."""
    base_density = db["materials"]["YSZ"]["density_tonne_per_mm3"]["data"]

    varied_cp = []
    for entry in db["materials"]["YSZ"]["specific_heat_mJ_per_tonneK"]["data"]:
        varied_cp.append([entry["T_K"], vary(entry["Cp"])])

    orientation, elastic = pick_ysz_elastic(db)

    return {
        "material_name":         "YSZ",
        "thickness_mm":          thickness_mm,
        "density":               vary(base_density),
        "elastic_orientation":   orientation,
        "elastic":               elastic,
        "thermal_conductivity":  pick_ysz_thermal_conductivity(db),
        "specific_heat":         varied_cp,
    }


def get_cmsx4_properties(db):
    """Extract CMSX-4 properties (unchanged)."""
    cmsx = db["materials"]["CMSX-4"]
    return {
        "material_name": "CMSX-4",
        "density":       cmsx["density_tonne_per_mm3"]["data"],
        "k":             cmsx["thermal_conductivity_W_per_mmK"]["data"],
        "elastic_001":   cmsx["elastic_properties"]["orientation_001"]["data"],
        "elastic_101":   cmsx["elastic_properties"]["orientation_101"]["data"],
        "elastic_111":   cmsx["elastic_properties"]["orientation_111"]["data"],
        "Cp":            cmsx["specific_heat_capacity_mJ_per_tonneK"]["data"],
    }


def generate_abaqus_input(ysz, cmsx, job_name, T_hot=1400.0, T_cold=600.0):
    """
    Generate ABAQUS input file for the YSZ / CMSX-4 heat-transfer simulation.

    Mesh layout (N_YSZ = 4, N_SUB = 10):
    --------------------------------------------------------------
    Node rows  0 ... N_YSZ            : YSZ layer (hot face at row 0)
    Node rows  N_YSZ ... N_YSZ+N_SUB : CMSX-4 substrate (cold face at last row)
    --------------------------------------------------------------
    Total node rows  : N_YSZ + N_SUB + 1  =  15
    Total nodes      : 2 x 15             =  30
    Total elements   : N_YSZ + N_SUB      =  14  (DC2D4)
    --------------------------------------------------------------
    Element numbering (1-indexed):
      1 ... N_YSZ          -> YSZ_Elements  (YSZ layer)
      N_YSZ+1 ... N_YSZ+N_SUB -> CMSX4_Elements (substrate)
    --------------------------------------------------------------
    Hot face (Gamma_hot)  : nodes 1, 2      (y = 0)
    Cold face (Gamma_cold): nodes 29, 30    (y = ysz_thick + cmsx_thick)
    """
    lines = []

    # -- Header ----------------------------------------------------------------
    lines.append("*Heading")
    lines.append(f"** Job: {job_name}, YSZ thickness: {ysz['thickness_mm']} mm")
    lines.append("*Preprint, echo=NO, model=NO, history=NO")
    lines.append("**")

    # -- Geometry --------------------------------------------------------------
    ysz_thick  = ysz["thickness_mm"]
    cmsx_thick = 10.0
    y_top      = ysz_thick + cmsx_thick

    dy = ysz_thick  / N_YSZ   # element height inside YSZ layer
    ds = cmsx_thick / N_SUB   # element height inside CMSX-4 layer (1 mm)

    total_rows  = N_YSZ + N_SUB + 1   # 
    total_nodes = 2 * total_rows       # 
    n_elements  = N_YSZ + N_SUB        # 

    # -- Part definition -------------------------------------------------------
    lines.append("** PART: Composite")
    lines.append("*Part, name=Composite")

    # Nodes: two columns (x = 0 and x = 10), rows bottom (hot) to top (cold).
    lines.append("*Node")
    for row in range(total_rows):
        if row <= N_YSZ:
            y = row * dy
        else:
            y = ysz_thick + (row - N_YSZ) * ds
        node_l = 2 * row + 1
        node_r = 2 * row + 2
        lines.append(f"  {node_l:4d},           0., {y:12.6f}")
        lines.append(f"  {node_r:4d},          10., {y:12.6f}")

    # Elements (DC2D4, CCW: bottom-left -> bottom-right -> top-right -> top-left).
    lines.append("*Element, type=DC2D4")
    for el in range(1, n_elements + 1):
        row = el - 1          # bottom row of this element (0-indexed)
        bl  = 2 * row + 1
        br  = 2 * row + 2
        tl  = 2 * (row + 1) + 1
        tr  = 2 * (row + 1) + 2
        lines.append(f"{el}, {bl}, {br}, {tr}, {tl}")

    # Element sets.
    ysz_el_list  = ", ".join(str(i) for i in range(1, N_YSZ + 1))
    cmsx_el_list = ", ".join(str(i) for i in range(N_YSZ + 1, n_elements + 1))
    lines.append("*Elset, elset=YSZ_Elements")
    lines.append(f" {ysz_el_list}")
    lines.append("*Elset, elset=CMSX4_Elements")
    lines.append(f" {cmsx_el_list}")

    # Node sets.
    hot_l  = 1
    hot_r  = 2
    cold_l = 2 * (total_rows - 1) + 1   # 29
    cold_r = 2 * (total_rows - 1) + 2   # 30
    all_nodes_str = ", ".join(str(i) for i in range(1, total_nodes + 1))

    lines.append("*Nset, nset=HotSide")
    lines.append(f" {hot_l}, {hot_r}")
    lines.append("*Nset, nset=ColdSide")
    lines.append(f" {cold_l}, {cold_r}")
    lines.append("*Nset, nset=AllNodes")
    lines.append(f" {all_nodes_str}")

    # Section assignments.
    lines.append("** Section: YSZ_Section")
    lines.append("*Solid Section, elset=YSZ_Elements, material=YSZ")
    lines.append("1.,")
    lines.append("** Section: CMSX4_Section")
    lines.append("*Solid Section, elset=CMSX4_Elements, material=CMSX4")
    lines.append("1.,")
    lines.append("*End Part")
    lines.append("**")

    # -- Assembly --------------------------------------------------------------
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

    # -- Material: CMSX-4 ------------------------------------------------------
    # Full temperature-dependent table is written (not just the 300 K entry).
    lines.append("** MATERIAL: CMSX4")
    lines.append("*Material, name=CMSX4")
    lines.append("*Density")
    lines.append(f"{cmsx['density'][0][1]:.6e},")
    lines.append("*Conductivity")
    for T, k in cmsx["k"]:
        lines.append(f"{k:.6e}, {T:.2f}")
    lines.append("*Specific Heat")
    for T, cp in cmsx["Cp"]:
        lines.append(f"{cp:.6e}, {T:.2f}")
    lines.append("**")

    # -- Material: YSZ ---------------------------------------------------------
    lines.append("** MATERIAL: YSZ")
    lines.append("*Material, name=YSZ")
    lines.append("*Density")
    lines.append(f"{ysz['density']:.6e},")
    lines.append("*Conductivity")
    for T, k in sorted(ysz["thermal_conductivity"], key=lambda x: x[0]):
        lines.append(f"{k:.6e}, {T:.2f}")
    lines.append("*Specific Heat")
    for T, cp in sorted(ysz["specific_heat"], key=lambda x: x[0]):
        lines.append(f"{cp:.6e}, {T:.2f}")
    lines.append("**")

    # -- Initial conditions ----------------------------------------------------
    lines.append("** INITIAL CONDITIONS")
    lines.append("*Initial Conditions, type=TEMPERATURE")
    lines.append("Composite-1.AllNodes, 300.")
    lines.append("**")

    # -- Heat-transfer step ----------------------------------------------------
    lines.append("** STEP: HeatTransfer")
    lines.append("*Step, name=HeatTransfer, nlgeom=NO")
    lines.append("*Heat Transfer, steady state")
    lines.append("1., 1., 1e-05, 1.")
    lines.append("**")
    lines.append("** BOUNDARY CONDITIONS")
    lines.append("*Boundary")
    lines.append(f"HotBC,  11, 11, {T_hot:.1f}")
    lines.append(f"ColdBC, 11, 11, {T_cold:.1f}")
    lines.append("**")
    lines.append("** OUTPUT REQUESTS")
    lines.append("*Output, field")
    lines.append("*Node Output")
    lines.append("NT,")
    #lines.append("*Element Output, position=CENTROIDAL")
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
    out_dir   = sys.argv[2]

    os.makedirs(out_dir, exist_ok=True)

    with open(json_file, "r") as f:
        db = json.load(f)

    cmsx = get_cmsx4_properties(db)

    thicknesses           = [0.5, 1.0, 1.5, 2.0]
    samples_per_thickness = 128

    manifest = []
    counter  = 1

    for thickness in thicknesses:
        for _ in range(samples_per_thickness):
            ysz      = pick_ysz_variant(db, thickness)
            job_name = f"YSZ_var_{counter:03d}"
            inp_file = f"{job_name}.inp"
            inp_path = os.path.join(out_dir, inp_file)

            with open(inp_path, "w") as f:
                f.write(
                    generate_abaqus_input(ysz, cmsx, job_name,
                                          T_hot=1400.0, T_cold=600.0)
                )

            manifest.append(
                {"id": job_name, "file": inp_file, "YSZ": ysz, "CMSX4": cmsx}
            )
            counter += 1

    manifest_path = os.path.join(out_dir, "variants_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} ABAQUS input files")
    print(f"Output directory: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Mesh: {N_YSZ} YSZ elements + {N_SUB} CMSX-4 elements = {N_YSZ+N_SUB} total")


if __name__ == "__main__":
    main()