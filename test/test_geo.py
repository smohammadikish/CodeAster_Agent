from tools.geo_generator import generate_geo_tool

print(generate_geo_tool.invoke({"length_m": 4.0, "geo_path": "beam.geo"}))
