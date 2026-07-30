"""Parsing and hashing for nanorun experiment-script frontmatter."""

from __future__ import annotations

import hashlib
import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


_FRONTMATTER_PATTERN = re.compile(
    r"\A[ \t\r\n]*(?P<quote>\"\"\"|''')(?P<body>.*?)(?P=quote)",
    re.DOTALL,
)
_FIELD_PATTERN = re.compile(r"^([a-z][a-z0-9_-]*):(.*)$")
_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ManifestError(ValueError):
    """A script's nanorun frontmatter is malformed."""


@dataclass(frozen=True)
class ScriptManifest:
    """The nanorun-specific fields declared in a script's first docstring."""

    parent: Optional[str] = None
    kernels: Optional[str] = None
    dependencies: tuple[tuple[str, str], ...] = ()

    @property
    def dependency_map(self) -> dict[str, str]:
        return dict(self.dependencies)


def _validate_python_path(path: str, field: str) -> str:
    path = path.strip()
    candidate = Path(path)
    if (
        not path
        or path.startswith("./")
        or "\\" in path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.suffix != ".py"
    ):
        raise ManifestError(
            f"Frontmatter {field} must be a repository-relative .py path: {path or '<empty>'}"
        )
    return candidate.as_posix()


def _validate_module_name(module: str) -> str:
    module = module.strip()
    if not _MODULE_PATTERN.fullmatch(module) or keyword.iskeyword(module):
        raise ManifestError(
            f"Dependency module must be a Python identifier: {module or '<empty>'}"
        )
    return module


def parse_script_manifest(script_content: str) -> ScriptManifest:
    """Parse nanorun fields from the first module docstring.

    Docstrings without a recognized nanorun field are ordinary prose and return
    an empty manifest. Nanorun fields must come before any descriptive prose.
    """

    match = _FRONTMATTER_PATTERN.match(script_content)
    if not match:
        return ScriptManifest()

    parent: Optional[str] = None
    kernels: Optional[str] = None
    dependencies: list[tuple[str, str]] = []
    seen_modules: set[str] = set()
    seen_fields: set[str] = set()
    recognized_fields: list[str] = []
    unknown_lines: list[tuple[int, str]] = []
    in_dependencies = False

    for line_number, raw_line in enumerate(match.group("body").splitlines(), start=1):
        if not raw_line.strip():
            continue

        if in_dependencies and raw_line[:1].isspace():
            entry = raw_line.strip()
            if ":" not in entry:
                raise ManifestError(
                    f"Invalid dependency entry on frontmatter line {line_number}: {entry}"
                )
            raw_module, raw_path = entry.split(":", 1)
            module = _validate_module_name(raw_module)
            path = _validate_python_path(raw_path, f"dependency {module}")
            if module in seen_modules:
                raise ManifestError(f"Duplicate dependency module: {module}")
            seen_modules.add(module)
            dependencies.append((module, path))
            continue

        in_dependencies = False
        field_match = _FIELD_PATTERN.fullmatch(raw_line)
        if not field_match:
            unknown_lines.append((line_number, raw_line.strip()))
            continue

        field_name, raw_value = field_match.groups()
        if field_name not in {"parent", "kernels", "dependencies"}:
            unknown_lines.append((line_number, raw_line.strip()))
            continue
        if unknown_lines:
            first_line_number, first_line = unknown_lines[0]
            raise ManifestError(
                "Nanorun fields must precede descriptive docstring content; "
                f"found content on frontmatter line {first_line_number}: {first_line}"
            )
        if field_name in seen_fields:
            raise ManifestError(f"Duplicate frontmatter field: {field_name}")
        if field_name == "parent" and recognized_fields:
            raise ManifestError("Frontmatter parent must be the first field")
        if field_name == "kernels" and "dependencies" in seen_fields:
            raise ManifestError(
                "Frontmatter kernels must appear before dependencies"
            )

        seen_fields.add(field_name)
        recognized_fields.append(field_name)

        if field_name == "parent":
            parent = _validate_python_path(raw_value, "parent")
        elif field_name == "kernels":
            kernels = _validate_python_path(raw_value, "kernels")
        else:
            if raw_value.strip():
                raise ManifestError(
                    "Frontmatter dependencies must be a nested module-to-path mapping"
                )
            in_dependencies = True

    if not recognized_fields:
        return ScriptManifest()
    if "dependencies" in seen_fields and not dependencies:
        raise ManifestError("Frontmatter dependencies mapping cannot be empty")
    if kernels and "triton_kernels" in seen_modules:
        raise ManifestError(
            "Declare either kernels or dependency module triton_kernels, not both"
        )

    return ScriptManifest(
        parent=parent,
        kernels=kernels,
        dependencies=tuple(dependencies),
    )


def resolve_repo_python_file(repo_root: Path, declared_path: str) -> str:
    """Resolve and validate a declared repository-relative Python file."""

    path = _validate_python_path(declared_path, "path")
    root = repo_root.resolve()
    resolved = (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ManifestError(
            f"Declared path must stay inside the repository: {declared_path}"
        ) from error
    if not resolved.is_file() or resolved.suffix != ".py":
        raise ManifestError(f"Declared Python file not found: {declared_path}")
    return relative.as_posix()


def compute_script_hash(
    repo_root: Path,
    script_path: str,
    kernels_path: Optional[str] = None,
    dependencies: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Hash an entrypoint and all files that define its code identity.

    The legacy script-plus-kernels byte layout is intentionally unchanged.
    Generic dependencies add canonical, module-sorted records after it.
    """

    script_full = repo_root / script_path
    if not script_full.is_file():
        return None
    combined = script_full.read_bytes()

    if kernels_path:
        kernels_full = repo_root / kernels_path
        if not kernels_full.is_file():
            return None
        combined += b"\n---KERNELS---\n" + kernels_full.read_bytes()

    for module, path in sorted((dependencies or {}).items()):
        dependency_full = repo_root / path
        if not dependency_full.is_file():
            return None
        descriptor = json.dumps(
            {"module": module, "path": path},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        combined += (
            b"\n---DEPENDENCY---\n"
            + descriptor
            + b"\n"
            + dependency_full.read_bytes()
        )

    return hashlib.sha256(combined).hexdigest()[:12]
