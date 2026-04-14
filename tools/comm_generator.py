# ----------------------------
#  tools/comm_generator.py
# ----------------------------

import os

from langchain_core.tools import tool

from library.section_calculators import rectangular_section
from storage import resolve_workspace_file

COMM_TEMPLATE_CANTILEVER = r"""
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

# __________ Define beam section geometry (GENERALE, mm) ________
cara = AFFE_CARA_ELEM(
    MODELE=model,
    POUTRE=_F(
        GROUP_MA=('BEAM',),
        SECTION='GENERALE',
        CARA=('A','IY','IZ','JX','AY','AZ'),
        VALE=({A}, {IY}, {IZ}, {JX}, {AY}, {AZ}),
    ),
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


COMM_TEMPLATE_SIMPLY_SUPPORTED = r"""
# __________________________ Start __________________________
DEBUT(PAR_LOT='NON');

# ______________________ Read the mesh ______________________
mesh = LIRE_MAILLAGE(FORMAT='MED', UNITE=20);

# ____________ Define node groups (BASE, MID, TIP, BEAM) __________
mesh = DEFI_GROUP(
  reuse=mesh,
  CREA_GROUP_NO=(
    _F(GROUP_MA=('BEAM',), CRIT_NOEUD='TOUS'),
    _F(GROUP_MA=('BASE',), CRIT_NOEUD='TOUS'),
    _F(GROUP_MA=('MID',),  CRIT_NOEUD='TOUS'),
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

# __________ Define beam section geometry (GENERALE, mm) ________
cara = AFFE_CARA_ELEM(
    MODELE=model,
    POUTRE=_F(
        GROUP_MA=('BEAM',),
        SECTION='GENERALE',
        CARA=('A','IY','IZ','JX','AY','AZ'),
        VALE=({A}, {IY}, {IZ}, {JX}, {AY}, {AZ}),
    ),
);

# __________ Boundary conditions (pin at BASE, roller at TIP) _________
constr = AFFE_CHAR_MECA(
  MODELE=model,
  DDL_IMPO=(
    _F(
      GROUP_NO=('BASE',),
      DX=0.0, DY=0.0, DZ=0.0, DRX=0.0
    ),
    _F(
      GROUP_NO=('TIP',),
      DY=0.0, DZ=0.0
    )
  )
);

# __________________ Applied load at MID _______________________
load = AFFE_CHAR_MECA(
  MODELE=model,
  FORCE_NODALE=_F(
    GROUP_NO=('MID',),
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
    GROUP_NO=('BASE', 'TIP'),
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


@tool("generate_comm", return_direct=False)
def generate_comm_tool(
    scenario: str = "cantilever_tip_load",
    med_name: str = "model.med",
    comm_name: str = "model.comm",
    E: float = 210e9,
    NU: float = 0.3,
    RHO: float = 7850.0,
    A_mm2: float | None = None,
    IY_mm4: float | None = None,
    IZ_mm4: float | None = None,
    JX_mm4: float | None = None,
    AY_mm2: float | None = None,
    AZ_mm2: float | None = None,
    HY_mm: float = 120.0,
    HZ_mm: float = 100.0,
    FY_N: float = -1500.0,
    run_id: str = "",
) -> str:
    """
    Generate a Code_Aster .comm file.

    IMPORTANT:
    - Preferred section inputs are A_mm2, IY_mm4, IZ_mm4, JX_mm4, AY_mm2, AZ_mm2
    - HY_mm and HZ_mm are a compatibility fallback for a solid rectangle
    - FY_N is the load in Newtons
    """

    med_path = resolve_workspace_file(med_name, "inputs", run_id or None)
    comm_path = resolve_workspace_file(comm_name, "inputs", run_id or None)

    if not os.path.exists(med_path):
        return f"Error: mesh file '{med_path}' not found."

    if None in {A_mm2, IY_mm4, IZ_mm4, JX_mm4, AY_mm2, AZ_mm2}:
        section = rectangular_section(HZ_mm, HY_mm)
        A_mm2 = float(section["A"])
        IY_mm4 = float(section["Iy"])
        IZ_mm4 = float(section["Iz"])
        JX_mm4 = float(section["IT"])
        AY_mm2 = float(section["Avy"])
        AZ_mm2 = float(section["Avz"])

    template = (
        COMM_TEMPLATE_SIMPLY_SUPPORTED
        if scenario == "simply_supported_midpoint_load"
        else COMM_TEMPLATE_CANTILEVER
    )
    content = template.format(
        E=E,
        NU=NU,
        RHO=RHO,
        A=A_mm2,
        IY=IY_mm4,
        IZ=IZ_mm4,
        JX=JX_mm4,
        AY=AY_mm2,
        AZ=AZ_mm2,
        FY_N=FY_N,
    )

    with open(comm_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Generated {comm_name} successfully."
