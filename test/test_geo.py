# ----------------------------
#  test/test_geo.py
# ----------------------------

from tools.geo_generator import generate_geo_tool

if __name__ == "__main__":
    result = generate_geo_tool.invoke(
        {"length_m": 4.0, "n_elems": 10, "geo_name": "beam.geo"}
    )
    print(result)
