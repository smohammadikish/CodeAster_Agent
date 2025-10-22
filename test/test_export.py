# test/test_export.py
from tools.export_generator import generate_export_tool

print(
    generate_export_tool.invoke(
        {
            "job_name": "beam_analysis",
            "comm_name": "model.comm",
            "med_name": "model.med",
            "export_name": "model.export",
        }
    )
)
