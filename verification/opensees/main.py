import sys
from pathlib import Path

# Add the src folder so the package can be imported when running this file directly.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ver_opensees.cli import main

# Start the command-line app when this file is executed.
if __name__ == "__main__":
    main()
