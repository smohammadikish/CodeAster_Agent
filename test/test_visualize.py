# test/test_visualize.py
from tools.visualize_tool import (visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)

print("=== 🧪 Running Deflection & Rotation Visualization ===")
print(visualize_resu_tool.invoke({"resu_name": "beam_analysis.resu"}))

print("\n=== 🧪 Running Internal Force Visualization (Shear & Moment) ===")
print(visualize_stress_tool.invoke({"stress_name": "tab_sie.txt"}))

print("\n=== 🧪 Running BENDING MOMENT MFY ===")
print(visualize_mfz_contour_tool.invoke({"stress_name": "tab_sie.txt"}))
