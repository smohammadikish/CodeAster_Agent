# test/test_visualize.py
import os
import sys

from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)

print("=== 🧪 Running Deflection & Rotation Visualization ===")
print(visualize_resu_tool.invoke({"resu_name": "beam_analysis.resu"}))

print("\n=== 🧪 Running Internal Force Visualization (Shear & Moment) ===")
print(visualize_stress_tool.invoke({"stress_name": "tab_sie.txt"}))

print("\n=== 🧪 Running BENDING MOMENT MFY ===")
print(visualize_mfz_contour_tool.invoke({"stress_name": "tab_sie.txt"}))


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    print("=== 🧪 Running Free-Body Visualization Test ===")

    workspace = "workspace"
    os.makedirs(workspace, exist_ok=True)
    reac_file = os.path.join(workspace, "tab_reac.txt")

    if not os.path.exists(reac_file):
        print(
            f"❌ Error: '{reac_file}' not found. Please copy your Code_Aster output file here."
        )
        sys.exit(1)

    # ✅ invoke instead of direct call
    result = visualize_freebody_tool.invoke(
        {"reac_name": "tab_reac.txt", "support_type": "fixed", "beam_length": 1.0}
    )

    print(result)

    img_path = os.path.join(workspace, "freebody.png")
    if os.path.exists(img_path):
        print(f"✅ Diagram created successfully: {img_path}")
    else:
        print("⚠️ No image generated.")
