# ----------------------------
#  test/test_comm.py
# ----------------------------

from tools.comm_generator import generate_comm_tool

if __name__ == "__main__":
    result = generate_comm_tool.invoke(
        {
            "med_name": "model.med",
            "comm_name": "model.comm",
            "E": 11e9,
            "NU": 0.3,
            "RHO": 480.0,
            "A_mm2": 12000.0,
            "IY_mm4": 14400000.0,
            "IZ_mm4": 10000000.0,
            "JX_mm4": 5000000.0,
            "AY_mm2": 10000.0,
            "AZ_mm2": 9000.0,
            "FY_N": -1500.0,
        }
    )
    print(result)
