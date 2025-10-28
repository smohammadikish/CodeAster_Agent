# tools/comm_generator.py
import os

from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

COMM_TEMPLATE = r"""
# __________________________ Start __________________________
DEBUT(PAR_LOT='NON');

# ______________________ Read the mesh ______________________
mesh = LIRE_MAILLAGE(FORMAT='MED', UNITE=20);

# ____________ Define node groups (BASE, TIP, BEAM) __________
mesh = DEFI_GROUP(
  reuse=mesh,
  CREA_GROUP_NO=(
    _F(GROUP_MA=('BEAM',), CRIT_NOEUD='TOUS'),
    _F(GROUP_MA=('BASE',), CRIT_NOEUD='TOUS'),
    _F(GROUP_MA=('TIP',),  CRIT_NOEUD='TOUS'),
  ),
  MAILLAGE=mesh
);

# ___________________ Define mechanical model _________________
model = AFFE_MODELE(
  MAILLAGE=mesh,
  AFFE=_F(
    GROUP_MA=('BEAM',),
    PHENOMENE='MECANIQUE',
    MODELISATION='POU_D_T'
  )
);

# ____________________ Define material ________________________
mat = DEFI_MATERIAU(ELAS=_F(E={E}, NU={NU}, RHO={RHO}));

mater = AFFE_MATERIAU(
  MAILLAGE=mesh,
  AFFE=_F(GROUP_MA=('BEAM',), MATER=mat)
);

# __________ Define beam section geometry (rectangular) ________
cara = AFFE_CARA_ELEM(
  MODELE=model,
  POUTRE=_F(
    GROUP_MA=('BEAM',),
    SECTION='RECTANGLE',
    CARA=('HY','HZ'),
    VALE=({HY}, {HZ})
  )
);

# __________________ Boundary condition (BASE) _________________
constr = AFFE_CHAR_MECA(
  MODELE=model,
  DDL_IMPO=_F(
    GROUP_NO=('BASE',),
    DX=0.0, DY=0.0, DZ=0.0,
    DRX=0.0, DRY=0.0, DRZ=0.0
  )
);

# __________________ Applied load on TIP _______________________
load = AFFE_CHAR_MECA(
  MODELE=model,
  FORCE_NODALE=_F(
    GROUP_NO=('TIP',),
    FY={FY}
  )
);

# __________________ Static analysis __________________________
res_stat = MECA_STATIQUE(
  MODELE=model,
  CHAM_MATER=mater,
  CARA_ELEM=cara,
  EXCIT=(
    _F(CHARGE=constr),
    _F(CHARGE=load)
  ),
  SOLVEUR=_F(
    METHODE='MUMPS',
    STOP_SINGULIER='OUI',
    RENUM='METIS',
    NPREC=8,
    ELIM_LAGR='NON'
  )
);

# __________ Compute displacements, reactions, stresses ________
res_stat = CALC_CHAMP(
  reuse=res_stat,
  RESULTAT=res_stat,
  FORCE=('REAC_NODA',),
  CONTRAINTE=('EFGE_ELNO', 'SIGM_ELNO')
);

# __________ Create result tables (reactions, displacements) ___
tab_reac = CREA_TABLE(
  RESU=_F(
    GROUP_NO=('BASE',),
    NOM_CHAM='REAC_NODA',
    RESULTAT=res_stat,
    TOUT_CMP='OUI'
  ),
  TITRE='Nodal Reactions'
);

tab_disp = CREA_TABLE(
  RESU=_F(
    TOUT='OUI',
    NOM_CHAM='DEPL',
    RESULTAT=res_stat,
    TOUT_CMP='OUI'
  ),
  TITRE='Nodal Displacements'
);

# __________________ Print tables to text ______________________
IMPR_TABLE(
  TABLE=tab_reac,
  NOM_PARA=('NOEUD','DX','DY','DZ','DRX','DRY','DRZ','FX','FY','FZ','MX','MY','MZ'),
  UNITE=18
);

IMPR_TABLE(
  TABLE=tab_disp,
  NOM_PARA=('NOEUD','DX','DY','DZ','DRX','DRY','DRZ'),
  UNITE=19
);

# __________________ Print displacement field __________________
IMPR_RESU(
  FORMAT='RESULTAT',
  UNITE=80,
  RESU=_F(RESULTAT=res_stat, NOM_CHAM='DEPL')
);

# __________________ Print local coordinate systems ____________
IMPR_CONCEPT(
  CONCEPT=_F(
    CARA_ELEM=cara,
    MODELE=model,
    REPERE_LOCAL='ELEM'
  ),
  FORMAT='RESULTAT',
  UNITE=53
);

# __________________________ End __________________________
FIN();
"""


@tool("generate_comm", return_direct=False)
def generate_comm_tool(
    med_name: str = "model.med",
    comm_name: str = "model.comm",
    E: float = 11e9,  # Pa
    NU: float = 0.3,
    RHO: float = 480.0,  # kg/m3
    HY: float = 0.12,  # m
    HZ: float = 0.10,  # m
    FY: float = -1500.0,  # N
) -> str:
    """
    Generate the .comm file (command input for Code_Aster).
    Compatible with Code_Aster 14.4 Windows/Linux.
    Includes group reconstruction (BASE, TIP, BEAM) and full output tables.
    """

    med_path = os.path.join(WORKSPACE, med_name)
    comm_path = os.path.join(WORKSPACE, comm_name)

    if not os.path.exists(med_path):
        return f"❌ Error: mesh file '{med_path}' not found."

    # Fill the template with input parameters
    content = COMM_TEMPLATE.format(E=E, NU=NU, RHO=RHO, HY=HY, HZ=HZ, FY=FY)

    # Write .comm file
    with open(comm_path, "w", encoding="utf-8") as f:
        f.write(content)

    return (
        f"✅ Generated {comm_name} successfully.\n"
        f"Material: E={E:.2e}, ν={NU}, ρ={RHO}\n"
        f"Section: HY={HY}, HZ={HZ}\n"
        f"Load: FY={FY} N applied at TIP\n"
        f"→ Ready for Code_Aster run."
    )
