"""Start backend (uvicorn) and frontend (streamlit) together.

Run with: python run.py
Stop with: Ctrl+C (kills both servers)
"""
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Re-launch under the project venv so `python run.py` works from anywhere.
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    sys.exit(subprocess.call([str(VENV_PY), str(Path(__file__).resolve())]))

PY = sys.executable

procs = [
    subprocess.Popen([PY, "-m", "uvicorn", "main:app", "--reload",
                      "--port", "8000"], cwd=ROOT),
    subprocess.Popen([PY, "-m", "streamlit", "run", "frontend/app.py",
                      "--server.port", "8501"], cwd=ROOT),
]

print("backend:  http://localhost:8000  (docs at /docs)")
print("frontend: http://localhost:8501")
print("Press Ctrl+C to stop both.")


def _stop(*_):
    for p in procs:
        if p.poll() is None:
            if sys.platform == "win32":
                # /T kills the whole tree (uvicorn --reload spawns children)
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                               capture_output=True)
            else:
                p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

try:
    for p in procs:
        p.wait()
finally:
    _stop()
