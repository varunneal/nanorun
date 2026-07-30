"""Experiment lineage tracking via script parentage and code diffs."""

import difflib
import hashlib
from pathlib import Path
from typing import Mapping, Optional, Tuple, TYPE_CHECKING

from .config import Config, get_repo_root
from .script_manifest import compute_script_hash, parse_script_manifest

if TYPE_CHECKING:
    from .remote_control import RemoteSession


def get_diffs_dir() -> Path:
    """Get path to diffs directory, creating if needed."""
    config_dir = Config.get_config_dir()
    diffs_dir = config_dir / "diffs"
    diffs_dir.mkdir(exist_ok=True)
    return diffs_dir


def parse_parent_path(script_content: str) -> Optional[str]:
    """Parse parent path from script frontmatter.

    Looks for a docstring at the start of the file containing:
        parent: /path/to/parent/script.py

    Args:
        script_content: The full content of the script file

    Returns:
        Parent path if found, None otherwise (root node)
    """
    return parse_script_manifest(script_content).parent


def parse_kernels_path(script_content: str) -> Optional[str]:
    """Parse kernels file path from script frontmatter.

    Looks for a docstring at the start of the file containing:
        kernels: path/to/kernels.py

    Args:
        script_content: The full content of the script file

    Returns:
        Kernels path if found, None otherwise
    """
    return parse_script_manifest(script_content).kernels


def parse_dependencies(script_content: str) -> dict[str, str]:
    """Parse module-to-path dependency declarations from frontmatter."""

    return parse_script_manifest(script_content).dependency_map


def read_local_file(path: str) -> Optional[str]:
    """Read file content from local repo.

    Args:
        path: Path relative to repo root

    Returns:
        File content or None if not found
    """
    full_path = get_repo_root() / path
    if full_path.exists():
        return full_path.read_text()
    return None


def compute_content_hash(content: bytes) -> str:
    """Compute SHA256 hash of content, returning first 12 chars."""
    full_hash = hashlib.sha256(content).hexdigest()
    return full_hash[:12]


def compute_file_hash(path: str) -> Optional[str]:
    """Compute SHA256 hash of a local file, returning first 12 chars.

    Uses read_bytes() to match remote daemon's hash computation exactly.

    Args:
        path: Path relative to repo root

    Returns:
        First 12 chars of SHA256 hash, or None if file not found
    """
    full_path = get_repo_root() / path
    if not full_path.exists():
        return None
    content = full_path.read_bytes()
    return compute_content_hash(content)


def compute_combined_hash(
    script_path: str,
    kernels_path: Optional[str] = None,
    dependencies: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Compute the combined hash of a script and its declared code files.

    The legacy script-plus-kernels hash format is preserved when no generic
    dependencies are declared.

    Args:
        script_path: Path to the script (relative to repo root)
        kernels_path: Optional path to the kernels file (relative to repo root)
        dependencies: Optional module-to-path dependency mapping

    Returns:
        First 12 chars of SHA256 hash, or None if script not found
    """
    return compute_script_hash(
        get_repo_root(),
        script_path,
        kernels_path=kernels_path,
        dependencies=dependencies,
    )


def get_parent_info(script_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Get parent path and hash for a script.

    Args:
        script_path: Path to the child script (relative to repo root)

    Returns:
        Tuple of (parent_path, parent_hash), both None if no parent
    """
    # Read child content
    script_content = read_local_file(script_path)
    if script_content is None:
        return None, None

    # Parse parent path from frontmatter
    parent_path = parse_parent_path(script_content)
    if parent_path is None:
        return None, None

    # Compute parent hash
    parent_hash = compute_file_hash(parent_path)
    if parent_hash is None:
        return parent_path, None  # Parent declared but file not found

    return parent_path, parent_hash


def generate_diff(child_path: str, parent_path: str) -> Optional[str]:
    """Generate unified diff between parent and child scripts.

    Args:
        child_path: Path to child script (relative to repo root)
        parent_path: Path to parent script (relative to repo root)

    Returns:
        Unified diff string, or None on error
    """
    parent_content = read_local_file(parent_path)
    child_content = read_local_file(child_path)

    if parent_content is None or child_content is None:
        return None

    # Generate unified diff
    parent_lines = parent_content.splitlines(keepends=True)
    child_lines = child_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        parent_lines,
        child_lines,
        fromfile=parent_path,
        tofile=child_path,
    )
    return "".join(diff)


def generate_combined_diff(
    child_path: str,
    parent_path: str,
    child_kernels: Optional[str],
    parent_kernels: Optional[str],
    child_dependencies: Optional[Mapping[str, str]] = None,
    parent_dependencies: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Generate a unified diff for an entrypoint and declared dependencies.

    Output format:
    --- parent_script
    +++ child_script
    [script diff]

    ================================================================================
    --- parent_kernels (or /dev/null)
    +++ child_kernels (or /dev/null)
    [kernels diff]

    Args:
        child_path: Path to child script (relative to repo root)
        parent_path: Path to parent script (relative to repo root)
        child_kernels: Optional path to child's kernels file
        parent_kernels: Optional path to parent's kernels file

    Returns:
        Combined unified diff string, or None on error
    """
    # Generate script diff
    script_diff = generate_diff(child_path, parent_path)
    if script_diff is None:
        return None

    child_files = dict(child_dependencies or {})
    parent_files = dict(parent_dependencies or {})
    if child_kernels:
        child_files["triton_kernels"] = child_kernels
    if parent_kernels:
        parent_files["triton_kernels"] = parent_kernels

    if not child_files and not parent_files:
        return script_diff

    parts = [script_diff]
    for module in sorted(set(parent_files) | set(child_files)):
        parent_dependency = parent_files.get(module)
        child_dependency = child_files.get(module)
        parent_content = (
            read_local_file(parent_dependency) if parent_dependency else ""
        ) or ""
        child_content = (
            read_local_file(child_dependency) if child_dependency else ""
        ) or ""
        dependency_diff = "".join(
            difflib.unified_diff(
                parent_content.splitlines(keepends=True),
                child_content.splitlines(keepends=True),
                fromfile=parent_dependency or "/dev/null",
                tofile=child_dependency or "/dev/null",
            )
        )
        if not dependency_diff:
            continue
        parts.append(
            "\n"
            + "=" * 80
            + f"\ndependency: {module}\n"
            + dependency_diff
        )

    return "".join(parts)


def store_diff(child_hash: str, diff_content: str) -> Path:
    """Store a diff file named by child hash.

    Args:
        child_hash: Hash of the child script (12 chars)
        diff_content: The unified diff content

    Returns:
        Path to the stored diff file
    """
    diffs_dir = get_diffs_dir()
    diff_path = diffs_dir / f"{child_hash}.diff"
    diff_path.write_text(diff_content)
    return diff_path


def get_diff(child_hash: str) -> Optional[str]:
    """Retrieve stored diff for a child hash.

    Args:
        child_hash: Hash of the child script

    Returns:
        Diff content if found, None otherwise
    """
    diffs_dir = get_diffs_dir()
    diff_path = diffs_dir / f"{child_hash}.diff"
    if diff_path.exists():
        return diff_path.read_text()
    return None


def process_lineage(
    script_path: str,
    child_hash: str,
) -> Tuple[Optional[str], Optional[str], Optional[Path]]:
    """Process lineage for a script: find parent, generate and store diff.

    Args:
        script_path: Path to the script being run (relative to repo root)
        child_hash: Pre-computed hash of the child script

    Returns:
        Tuple of (parent_path, parent_hash, diff_path)
        All None if script is a root node (no parent)
    """
    # Get parent info
    parent_path, parent_hash = get_parent_info(script_path)
    if parent_path is None:
        # Root node - no parent
        return None, None, None

    if parent_hash is None:
        # Parent declared but not found
        return parent_path, None, None

    child_content = read_local_file(script_path)
    parent_content = read_local_file(parent_path)
    if child_content is None or parent_content is None:
        return parent_path, parent_hash, None
    child_manifest = parse_script_manifest(child_content)
    parent_manifest = parse_script_manifest(parent_content)

    # Generate and store diff
    diff_content = generate_combined_diff(
        script_path,
        parent_path,
        child_manifest.kernels,
        parent_manifest.kernels,
        child_manifest.dependency_map,
        parent_manifest.dependency_map,
    )
    if diff_content:
        diff_path = store_diff(child_hash, diff_content)
        return parent_path, parent_hash, diff_path

    return parent_path, parent_hash, None
