# Local Development

This project has been prepared for local development from the zip archive.

## Environment

- Project path: `D:\Samsung\WireVision\StereoFastening-main`
- Python runtime: CPython 3.13.13, installed with `uv`
- Virtual environment: `.venv`

Create or recreate the environment:

```powershell
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --index-strategy unsafe-best-match -r requirements.txt
```

The `--index-strategy unsafe-best-match` option is needed because
`requirements.txt` uses the PyTorch CUDA index and `sam2` also needs packages
from PyPI.

## Run

```powershell
.\.venv\Scripts\python.exe PythonUI.py
```

Or double-click/run:

```powershell
.\run_app.cmd
```

Using `.venv\Scripts\python.exe` directly avoids PowerShell execution-policy
issues with `Activate.ps1`.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
```

Or:

```powershell
.\run_tests.cmd
```

Current smoke checks performed on this machine:

- `python -m unittest discover -p "test_*.py"`: 70 tests passed
- `python -m compileall -q .`: passed
- `import PythonUI`: passed

## Notes

- The app can run without a RealSense camera for the non-hardware paths, but
  camera capture and production operation still require the target hardware.
- SAM2 checkpoints are not included in the zip. Put them under `models\sam2`
  when video tracking is needed.
- Runtime data folders such as `Settings`, `Calibration`, `Templates`,
  `Preset`, and `Logs` are intentionally ignored by Git.
