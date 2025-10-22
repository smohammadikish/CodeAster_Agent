# tools/comm_generator.py
import os
from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

COMM_TEMPLATE = r"""
# --- Code_Aster Beam Analysis (.comm) ---
DEBUT();

mesh = LIRE_MAILLAGE(FORMAT='MED', UNITE=20);

model = AFFE_MODELE(
  MAILLAGE=mesh,
  AFFE=_F(GROUP_MA=('BEAM',), PHENOMENE='MECANIQUE', MODELISATION='POU_D_E')
);

# --- Material definition ---
mat = DEFI_MATERIAU(ELAS=_F(E={E}, NU={NU}, RHO={RHO}));

chmat = AFFE_MATERIAU(
  MAILLAGE=mesh,
  AFFE=_F(GROUP_MA=('BEAM',), MATER=mat)
);

# --- Beam section geometry (RECTANGLE) ---
cara = AFFE_CARA_ELEM(
  MODELE=model,
  POUTRE=_F(GROUP_MA=('BEAM',),
            SECTION='RECTANGLE',
            CARA=('HY','HZ'),
            VALE=({HY}, {HZ}),
            ORIG_INER=(0.0, 0.0, 0.0),
            VECT_Y=(0.0, 1.0, 0.0))
);

# --- Boundary condition (BASE fixed) ---
bc = AFFE_CHAR_CINE(
  MODELE=model,
  MECA_IMPO=_F(GROUP_NO=('BASE',),
               DX=0.0, DY=0.0, DZ=0.0,
               DRX=0.0, DRY=0.0, DRZ=0.0)
);

# --- Load (Nodal Force on TIP) ---
load = AFFE_CHAR_MECA(
  MODELE=model,
  FORCE_NODALE=_F(GROUP_NO=('TIP',), FY={FY})
);

# --- Static analysis ---
res = MECA_STATIQUE(
  MODELE=model,
  CHAM_MATER=chmat,
  CARA_ELEM=cara,
  EXCIT=(_F(CHARGE=bc), _F(CHARGE=load))
);

# --- Print displacement field ---
IMPR_RESU(
  FORMAT='RESULTAT',
  UNITE=80,
  RESU=_F(RESULTAT=res, NOM_CHAM='DEPL')
);

FIN();
"""

@tool("generate_comm", return_direct=False)
def generate_comm_tool(
    med_name: str = "model.med",
    comm_name: str = "model.comm",
    E: float = 11e9,      # Pa
    NU: float = 0.3,
    RHO: float = 500.0,   # kg/m3
    HY: float = 0.10,     # m
    HZ: float = 0.10,     # m
    FY: float = -1000.0,  # N
) -> str:
    """
    Generate a .comm file for Code_Aster beam analysis.
    It uses a .med mesh (BEAM, BASE, TIP) and defines material, section, BC, and load.
    """
    med_path = os.path.join(WORKSPACE, med_name)
    comm_path = os.path.join(WORKSPACE, comm_name)

    if not os.path.exists(med_path):
        return f"❌ Error: mesh file '{med_path}' not found."

    # fill template
    content = COMM_TEMPLATE.format(E=E, NU=NU, RHO=RHO, HY=HY, HZ=HZ, FY=FY)

    # write to file
    with open(comm_path, "w") as f:
        f.write(content)

    return (
        f"✅ Generated {comm_name} in '{WORKSPACE}'\n"
        f"Material: E={E:.2e}, ν={NU}, ρ={RHO}\n"
        f"Section: HY={HY}, HZ={HZ}\n"
        f"Load: FY={FY} N (on TIP)\n"
        f"→ Ready for Code_Aster run"
    )