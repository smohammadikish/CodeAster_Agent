# tools/validators.py
import os

def file_exists(path: str) -> bool:
    """Check if a file exists and is not empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def check_mess_for_errors(mess_path: str) -> bool:
    """
    Check Code_Aster .mess file for errors or fatal messages.
    Returns True if everything looks OK, False otherwise.
    """
    if not os.path.exists(mess_path):
        return False
    with open(mess_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().lower()
    # 'error' or 'fatal' means something went wrong
    return "error" not in content and "fatal" not in content
