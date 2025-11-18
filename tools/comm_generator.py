# ----------------------------
#  tools/comm_generator.py  (UPDATED to match mm pipeline)
# ----------------------------

import os

from langchain_core.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


# ============================================================
#  Code_Aster .comm template (values are in millimeters)
# ============================================================

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

# __________ Define beam section geometry (rectangular, mm) ________
cara = AFFE_CARA_ELEM(
  MODELE=model,
  POUTRE=_F(
    GROUP_MA=('BEAM',),
    SECTION='RECTANGLE',
    CARA=('HY','HZ'),
    VALE=({HY_mm}, {HZ_mm})
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
    FY={FY_N}
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

# __________ Compute fields: reactions, displacements, stresses ________
res_stat = CALC_CHAMP(
  reuse=res_stat,
  RESULTAT=res_stat,
  FORCE=('REAC_NODA',),
  CONTRAINTE=('EFGE_ELNO', 'SIEF_ELNO')
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
IMPR_TABLE(TABLE=tab_reac, UNITE=18, SEPARATEUR=' ', FORMAT='TABLEAU', FORMAT_R='E12.5');
IMPR_TABLE(TABLE=tab_disp, UNITE=19, SEPARATEUR=' ', FORMAT='TABLEAU', FORMAT_R='E12.5');

# __________________ Print displacement field __________________
IMPR_RESU(
  FORMAT='RESULTAT',
  UNITE=80,
  RESU=_F(RESULTAT=res_stat, NOM_CHAM='DEPL')
);

# __________________ Print local coordinate systems ____________
IMPR_CONCEPT(
  CONCEPT=_F(CARA_ELEM=cara, MODELE=model, REPERE_LOCAL='ELEM'),
  FORMAT='RESULTAT',
  UNITE=53
);

# __________________ Internal forces table ______________________
SIEFTAB = POST_RELEVE_T(
  ACTION=_F(
    INTITULE='BeamForces',
    RESULTAT=res_stat,
    NOM_CHAM='SIEF_ELNO',
    GROUP_NO=('BEAM',),
    OPERATION='EXTRACTION',
    TOUT_ORDRE='OUI',
    TOUT_CMP='OUI'
  )
);

IMPR_TABLE(TABLE=SIEFTAB, UNITE=21, SEPARATEUR=';', FORMAT='TABLEAU', FORMAT_R='E12.5');

# __________________________ End __________________________
FIN();
"""


# ============================================================
#  Tool definition (arguments in **mm**)
# ============================================================


@tool("generate_comm", return_direct=False)
def generate_comm_tool(
    med_name: str = "model.med",
    comm_name: str = "model.comm",
    E: float = 210e9,
    NU: float = 0.3,
    RHO: float = 7850.0,
    HY_mm: float = 120.0,
    HZ_mm: float = 100.0,
    FY_N: float = -1500.0,
) -> str:
    """
    Generate a Code_Aster .comm file.

    IMPORTANT:
    - HY_mm, HZ_mm must be in millimeters (mm)
    - FY_N is the load in Newtons
    """

    med_path = os.path.join(WORKSPACE, med_name)
    comm_path = os.path.join(WORKSPACE, comm_name)

    if not os.path.exists(med_path):
        return f"❌ Error: mesh file '{med_path}' not found."

    # Fill template
    content = COMM_TEMPLATE.format(
        E=E,
        NU=NU,
        RHO=RHO,
        HY_mm=HY_mm,
        HZ_mm=HZ_mm,
        FY_N=FY_N,
    )

    # Write file
    with open(comm_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"✅ Generated {comm_name} successfully."
