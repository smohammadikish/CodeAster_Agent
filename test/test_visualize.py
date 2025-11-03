# ----------------------------
#  test/test_visualize.py
# ----------------------------

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)

# ============================================================
# 🧪 TOOL 1 — DEFLECTION & ROTATION
# ============================================================
print("=== 🧪 Running Deflection & Rotation Visualization ===")
resu_result = visualize_resu_tool.invoke({"resu_name": "beam_analysis.resu"})
print(resu_result)

# ============================================================
# 🧪 TOOL 2 — INTERNAL FORCES (SHEAR + MOMENT)
# ============================================================
print("\n=== 🧪 Running Internal Force Visualization (Shear & Moment) ===")
stress_result = visualize_stress_tool.invoke({"stress_name": "tab_sie.txt"})
print(stress_result)

# ============================================================
# 🧪 TOOL 3 — BENDING MOMENT CONTOUR (MFZ)
# ============================================================
print("\n=== 🧪 Running BENDING MOMENT MFZ CONTOUR ===")
contour_result = visualize_mfz_contour_tool.invoke({"stress_name": "tab_sie.txt"})
print(contour_result)

# ============================================================
# 🧪 TOOL 4 — FREE-BODY DIAGRAM
# ============================================================
if __name__ == "__main__":
    print("\n=== 🧪 Running Free-Body Visualization Test ===")

    workspace = "workspace"
    os.makedirs(workspace, exist_ok=True)
    reac_file = os.path.join(workspace, "tab_reac.txt")

    if not os.path.exists(reac_file):
        print(
            f"❌ Error: '{reac_file}' not found. Please copy your Code_Aster output file here."
        )
        sys.exit(1)

    # ✅ Invoke instead of direct call
    freebody_result = visualize_freebody_tool.invoke(
        {
            "reac_name": "tab_reac.txt",
            "support_type": "fixed",
            "geo_name": "model.geo",
            "comm_name": "model.comm",
            "unit": "m",
        }
    )
    print(freebody_result)

    # Verify diagram creation
    img_path = os.path.join(workspace, "freebody.png")
    if os.path.exists(img_path):
        print(f"✅ Diagram created successfully: {img_path}")
    else:
        print("⚠️ No image generated.")
