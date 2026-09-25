"""Build the reviewable, source-only pip-installable ZIP; never include .venv."""

import hashlib
from pathlib import Path
import zipfile


def build():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in (
        "pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
        "VENDORED_SHA256.json", "build_archive.py",
    )]
    for folder in ("src", "examples", "tests"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            if path.suffix in (".py", ".go") or path.name in ("py.typed", "LICENSE", "go.mod", "go.sum"):
                paths.append(path)
    output = root / "dist" / "lebrel-encrypted-python.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(paths):
            name = "lebrel-encrypted-python/" + path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    print(str(output))
    print("sha256=" + hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
