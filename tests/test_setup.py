"""Run setup.sh with mocked commands; no GPU, network or installation required.

Usage: python3 -m unittest discover -s tests -p test_setup.py -v
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SETUP = Path(__file__).resolve().parents[1] / "scripts" / "setup.sh"


class SetupTests(unittest.TestCase):
    def run_setup(self, header, smi_exit=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "activate").write_text(":\n")
            commands = {
                "nvidia-smi": (
                    '#!/usr/bin/env bash\n'
                    'if [[ "$*" == *--query-gpu=* ]]; then\n'
                    '  printf "0, NVIDIA GeForce RTX 4090, 24564 MiB, 615.71.09\\n"\n'
                    'else\n'
                    '  printf "%s\\n" "$MOCK_SMI_HEADER"\n'
                    '  exit "$MOCK_SMI_EXIT"\n'
                    'fi\n'
                ),
                "uv": '#!/usr/bin/env bash\nprintf "MOCK uv %s\\n" "$*"\n',
                "python": '#!/usr/bin/env bash\nprintf "MOCK version check\\n"\n',
            }
            for name, content in commands.items():
                path = bin_dir / name
                path.write_text(content)
                path.chmod(0o755)
            env = dict(
                os.environ,
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                VENV_DIR=str(venv),
                VLLM_VERSION="0.30.0",
                MOCK_SMI_HEADER=header,
                MOCK_SMI_EXIT=str(smi_exit),
            )
            return subprocess.run(
                ["bash", str(SETUP)], env=env, text=True, capture_output=True,
                timeout=10,
            )

    def test_supported_headers(self):
        cases = [
            ("CUDA Version: 13.0", "cu130"),
            ("CUDA Version:   13.1", "cu130"),
            ("CUDA Version:\t13.0", "cu130"),
            ("CUDA Version:13.0", "cu130"),
            ("CUDA Version : 13.0", "cu130"),
            ("CUDA Version:  12.9", "cu129"),
            # Exact header supplied by the rented instance.
            ("| NVIDIA-SMI 615.71.09              KMD Version: 615.71.09"
             "     CUDA UMD Version: 13.4     |", "cu130"),
            ("CUDA  UMD  Version:   13.4", "cu130"),
            ("CUDA UMD Version: 12.9", "cu129"),
        ]
        for header, backend in cases:
            with self.subTest(header=header):
                result = self.run_setup(header)
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                self.assertIn(f"--torch-backend={backend}", output)
                self.assertIn("setup 完成", output)

    def test_invalid_headers_report_error(self):
        for header in ("CUDA Version: N/A", "CUDA UMD Version: N/A", "no CUDA field"):
            with self.subTest(header=header):
                result = self.run_setup(header)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ERROR: 读不到有效的驱动 CUDA 版本", result.stderr)
                self.assertNotIn("MOCK uv", result.stdout)

    def test_nvidia_smi_failure_reports_error(self):
        result = self.run_setup("", smi_exit=9)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR: nvidia-smi 执行失败", result.stderr)
        self.assertNotIn("MOCK uv", result.stdout)


if __name__ == "__main__":
    unittest.main()
