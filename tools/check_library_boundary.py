"""Check that source code stays inside the project's library boundary.

The image-processing pipeline must be implemented by hand. OpenCV is allowed
for image file input and output only, and ready-made vision libraries are not
allowed at all. See the "Library boundary" section of README.md.

Usage:
    python tools/check_library_boundary.py [PATH ...]

With no arguments, checks src/, tools/ and experiments/ if they exist.
Exits with status 1 if any violation is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# OpenCV names that may be used. Everything else on cv2 is rejected.
ALLOWED_CV2_NAMES = {"imread", "imwrite", "imdecode", "imencode"}
ALLOWED_CV2_PREFIXES = ("IMREAD_", "IMWRITE_")

# Modules that provide ready-made image-processing routines.
FORBIDDEN_MODULES = ("skimage", "scipy.ndimage")

DEFAULT_PATHS = ("src", "tools", "experiments")


def _cv2_name_allowed(name: str) -> bool:
    return name in ALLOWED_CV2_NAMES or name.startswith(ALLOWED_CV2_PREFIXES)


def _is_forbidden_module(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in FORBIDDEN_MODULES)


def _dotted_name(node: ast.AST) -> str | None:
    """Return 'a.b.c' for a Name/Attribute chain, or None for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def check_source(source: str, filename: str = "<string>") -> list[tuple[int, str]]:
    """Return (line, message) for every boundary violation in the source."""
    tree = ast.parse(source, filename=filename)
    violations: list[tuple[int, str]] = []
    cv2_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cv2" or alias.name.startswith("cv2."):
                    cv2_aliases.add((alias.asname or alias.name).split(".")[0])
                if _is_forbidden_module(alias.name):
                    violations.append((node.lineno, f"import of {alias.name} is not allowed"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "cv2" or node.module.startswith("cv2."):
                for alias in node.names:
                    if not _cv2_name_allowed(alias.name):
                        violations.append(
                            (node.lineno, f"cv2.{alias.name} is outside the image I/O allowlist")
                        )
            if _is_forbidden_module(node.module):
                violations.append((node.lineno, f"import from {node.module} is not allowed"))
            for alias in node.names:
                if _is_forbidden_module(f"{node.module}.{alias.name}"):
                    violations.append(
                        (node.lineno, f"import of {node.module}.{alias.name} is not allowed")
                    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted_name(node)
        if dotted is None:
            continue
        parts = dotted.split(".")
        # Only inspect the first attribute hanging directly off the cv2 alias.
        if len(parts) == 2 and parts[0] in cv2_aliases and not _cv2_name_allowed(parts[1]):
            violations.append((node.lineno, f"cv2.{parts[1]} is outside the image I/O allowlist"))

    return sorted(set(violations))


def iter_python_files(paths: list[Path]):
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(p for p in path.rglob("*.py") if ".venv" not in p.parts)


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv] or [Path(p) for p in DEFAULT_PATHS if Path(p).exists()]
    count = 0
    checked = 0
    for file in iter_python_files(paths):
        checked += 1
        try:
            violations = check_source(file.read_text(encoding="utf-8"), str(file))
        except SyntaxError as exc:
            print(f"{file}:{exc.lineno}: could not parse: {exc.msg}")
            count += 1
            continue
        for line, message in violations:
            print(f"{file}:{line}: {message}")
            count += 1

    if count:
        print(f"\n{count} library boundary violation(s) in {checked} file(s) checked.")
        return 1
    print(f"Library boundary OK ({checked} file(s) checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
