"""Package the API for Lambda Python 3.12, Linux x86_64."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parent
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    destination = root / "dist" / "lambda-query-api.zip"
    destination.parent.mkdir(exist_ok=True)
    with TemporaryDirectory() as temporary:
        staging = Path(temporary)
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--platform", "manylinux2014_x86_64", "--implementation", "cp",
            "--python-version", "3.12", "--only-binary=:all:",
            "--no-compile", "--target", str(staging),
            *metadata["project"]["dependencies"],
        ], check=True)
        for name in ("main.py", "lambda_function.py"):
            (staging / name).write_bytes((root / name).read_bytes())
        with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(staging).as_posix())
    print(destination)


if __name__ == "__main__":
    main()

