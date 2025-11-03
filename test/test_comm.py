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
            "HY": 0.12,
            "HZ": 0.10,
            "FY": -1500.0,
        }
    )
    print(result)
