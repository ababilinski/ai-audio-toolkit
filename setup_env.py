"""Setup script to create virtual environment and install dependencies."""
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    project_dir = Path(__file__).parent
    venv_dir = project_dir / ".venv"

    # Check for uv first, fall back to venv
    uv_path = shutil.which("uv")

    if uv_path:
        print("=== Using UV package manager ===")
        print("Creating virtual environment with UV...")
        subprocess.run([uv_path, "venv", str(venv_dir)], check=True, cwd=str(project_dir))

        print("Installing dependencies with UV (GPU)...")
        subprocess.run(
            [uv_path, "pip", "install", "-r", "requirements.txt",
             "--python", str(venv_dir / "Scripts" / "python.exe")],
            check=True, cwd=str(project_dir)
        )
    else:
        print("=== UV not found, using standard venv ===")
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

        pip_path = venv_dir / "Scripts" / "pip.exe"
        print("Installing dependencies (GPU)...")
        subprocess.run(
            [str(pip_path), "install", "-r", "requirements.txt"],
            check=True, cwd=str(project_dir)
        )

    print()
    print("=== Setup complete! ===")
    print(f"Virtual environment created at: {venv_dir}")
    print()
    print("To activate:")
    print(f"  .venv\\Scripts\\activate     (Windows CMD)")
    print(f"  .venv/Scripts/activate      (Git Bash)")
    print()
    print("To run the application:")
    print("  python -m audio_editor.main")
    print()
    print("For CPU-only installation, use requirements-cpu.txt instead.")


if __name__ == "__main__":
    main()
