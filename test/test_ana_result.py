def cantilever_tip_response_rect(L, P, E, b, h):
    """
    Calculates tip deflection and rotation for a cantilever beam (rectangular section).

    Parameters:
        L (float): Beam length in meters.
        P (float): Tip load in Newtons.
        E (float): Young's modulus in Pascals.
        b (float): Section width (m)
        h (float): Section height (m)

    Returns:
        dict: { "I_m4": I, "deflection_m": δ, "rotation_rad": θ }
    """

    # Second moment of area for a rectangular cross-section
    I = (b * h**3) / 12.0

    # Tip response formulas
    delta = 1000 * P * L**3 / (3 * E * I)  # Tip deflection (mm)
    theta = P * L**2 / (2 * E * I)  # Tip rotation (radians)

    return {"I_m4": I, "deflection_mm": delta, "rotation_rad": theta}


# ---------------- Example ----------------
if __name__ == "__main__":
    L = 6.13  # meters
    P = 1350  # N
    E = 210e9  # Pa (steel)
    b = 0.21  # meters
    h = 0.13  # meters

    result = cantilever_tip_response_rect(L, P, E, b, h)

    print("Section I (m^4):", result["I_m4"])
    print("Tip Deflection (mm):", result["deflection_mm"])
    print("Tip Rotation (rad):", result["rotation_rad"])
