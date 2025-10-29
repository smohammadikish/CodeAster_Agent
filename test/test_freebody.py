# test/test_freebody.py
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.freebody_tool import visualize_freebody_tool

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
