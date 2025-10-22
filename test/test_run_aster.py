# test/test_run_aster.py
from tools.run_aster_tool import run_code_aster_tool

print(run_code_aster_tool.invoke({"export_name": "model.export"}))
