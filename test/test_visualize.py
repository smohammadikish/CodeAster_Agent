# test/test_visualize.py
from tools.visualize_tool import visualize_resu_tool

print(visualize_resu_tool.invoke({"resu_name": "beam_job.resu", "n_points": 50}))
