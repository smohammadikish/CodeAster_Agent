from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENSEES_SRC = REPO_ROOT / "verification" / "opensees" / "src"
RUNTIME_ROOT = REPO_ROOT / "workspace" / "test_runtime"

if str(OPENSEES_SRC) not in sys.path:
    sys.path.insert(0, str(OPENSEES_SRC))

from ver_opensees import config


class OpenSeesEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original = {
            key: os.environ.get(key) for key in config.OPENSEES_ENV_KEYS
        }
        for key in config.OPENSEES_ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in config.OPENSEES_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in self._original.items():
            if value is not None:
                os.environ[key] = value

    def test_load_dotenv_maps_legacy_opensees_key_to_canonical_name(self) -> None:
        temp_dir = RUNTIME_ROOT / "opensees_env_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        env_path = temp_dir / ".env"
        env_path.write_text(
            'opensees="C:\\STADATA\\OpenSees3.8.0\\bin\\OpenSees.exe"\n',
            encoding="utf-8",
        )

        config._load_dotenv(env_path)

        expected = r"C:\STADATA\OpenSees3.8.0\bin\OpenSees.exe"
        self.assertEqual(os.environ["OPENSEES_EXECUTABLE"], expected)
        self.assertEqual(os.environ["opensees"], expected)
        self.assertEqual(os.environ["OPENSEES"], expected)

    def test_resolve_opensees_executable_prefers_canonicalized_env_value(self) -> None:
        os.environ["opensees"] = r"C:\STADATA\OpenSees3.8.0\bin\OpenSees.exe"

        resolved = config._resolve_opensees_executable({})

        self.assertEqual(resolved, r"C:\STADATA\OpenSees3.8.0\bin\OpenSees.exe")
        self.assertEqual(
            os.environ["OPENSEES_EXECUTABLE"],
            r"C:\STADATA\OpenSees3.8.0\bin\OpenSees.exe",
        )


if __name__ == "__main__":
    unittest.main()
