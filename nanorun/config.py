"""Session and configuration management.

Supports multiple named sessions stored in .nanorun/sessions/{name}.json
with an active session pointer in .nanorun/active_session.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
from datetime import datetime, timezone

TRACK_FILE = ".track.json"


@dataclass
class SessionConfig:
    """Configuration for a remote session."""
    name: str  # Session name (e.g. "session-1", "my-h100")
    host: str
    user: str
    port: int = 22
    cuda_version: Optional[str] = None
    gpu_type: str = "H100"  # H100, H200, GH200, DGX_SPARK
    gpu_count: int = 1  # Number of GPUs on the machine
    has_sudo: bool = True
    repo_path: str = "~/nanorun"
    tmux_session: str = "nanorun"
    key_file: Optional[str] = None  # Path to SSH private key (-i flag)
    ssh_options: Optional[List[str]] = None  # Extra SSH -o options (e.g. ["IdentitiesOnly=yes"])
    use_pty: bool = False  # Request PTY for exec (needed for RunPod SSH proxy)


@dataclass
class Track:
    """An experiment track/workstream. Stored as .track.json in track directory."""
    name: str
    directory: str  # Relative to repo root
    description: str = ""
    created_at: str = ""

    @classmethod
    def load(cls, directory: Path) -> Optional["Track"]:
        """Load track from .track.json in directory."""
        track_file = directory / TRACK_FILE
        if not track_file.exists():
            return None
        data = json.loads(track_file.read_text())
        # Directory is relative to repo root
        repo_root = Path(__file__).parent.parent
        rel_dir = str(directory.relative_to(repo_root))
        return cls(
            name=data.get("name", directory.name),
            directory=rel_dir,
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
        )

    def save(self) -> None:
        """Save track to .track.json in directory."""
        repo_root = Path(__file__).parent.parent
        track_dir = repo_root / self.directory
        track_dir.mkdir(parents=True, exist_ok=True)
        track_file = track_dir / TRACK_FILE
        data = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
        }
        track_file.write_text(json.dumps(data, indent=2))

    def delete(self) -> None:
        """Delete the .track.json file."""
        repo_root = Path(__file__).parent.parent
        track_file = repo_root / self.directory / TRACK_FILE
        if track_file.exists():
            track_file.unlink()

    @classmethod
    def create(cls, name: str, directory: str, description: str = "") -> "Track":
        """Create a new track and save it."""
        track = cls(
            name=name,
            directory=directory,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        track.save()
        return track


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def discover_tracks() -> List[Track]:
    """Discover all tracks by scanning for .track.json files."""
    repo_root = get_repo_root()
    tracks = []
    for track_file in repo_root.rglob(TRACK_FILE):
        track = Track.load(track_file.parent)
        if track:
            tracks.append(track)
    return sorted(tracks, key=lambda t: t.name)


def get_track(name: str) -> Optional[Track]:
    """Get a track by name."""
    for track in discover_tracks():
        if track.name == name:
            return track
    return None


@dataclass
class Config:
    """Main configuration managing named sessions.

    Sessions are stored in .nanorun/sessions/{name}.json.
    The active session name is stored in .nanorun/active_session.
    """
    session: Optional[SessionConfig] = None

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get the .nanorun config directory, creating if needed."""
        config_dir = get_repo_root() / ".nanorun"
        config_dir.mkdir(exist_ok=True)
        return config_dir

    @classmethod
    def get_sessions_dir(cls) -> Path:
        """Get the .nanorun/sessions directory, creating if needed."""
        sessions_dir = cls.get_config_dir() / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        return sessions_dir

    @classmethod
    def get_active_session_file(cls) -> Path:
        return cls.get_config_dir() / "active_session"

    @classmethod
    def _get_session_file(cls, name: str) -> Path:
        return cls.get_sessions_dir() / f"{name}.json"

    @classmethod
    def next_session_name(cls) -> str:
        """Generate the next auto-incremented session name."""
        existing = {f.stem for f in cls.get_sessions_dir().glob("*.json")}
        n = 1
        while f"session-{n}" in existing:
            n += 1
        return f"session-{n}"

    @classmethod
    def get_active_session_name(cls) -> Optional[str]:
        """Get the name of the active session, or None."""
        active_file = cls.get_active_session_file()
        if active_file.exists():
            name = active_file.read_text().strip()
            if name and cls._get_session_file(name).exists():
                return name
        return None

    @classmethod
    def set_active_session(cls, name: str) -> None:
        """Set the active session by name."""
        if not cls._get_session_file(name).exists():
            raise ValueError(f"Session '{name}' does not exist")
        cls.get_active_session_file().write_text(name)

    @classmethod
    def list_sessions(cls) -> List[SessionConfig]:
        """List all saved sessions."""
        sessions = []
        for f in sorted(cls.get_sessions_dir().glob("*.json")):
            try:
                data = json.loads(f.read_text())
                sessions.append(SessionConfig(**data))
            except (json.JSONDecodeError, TypeError):
                pass
        return sessions

    @classmethod
    def load_session(cls, name: str) -> Optional[SessionConfig]:
        """Load a specific session by name."""
        session_file = cls._get_session_file(name)
        if not session_file.exists():
            return None
        data = json.loads(session_file.read_text())
        return SessionConfig(**data)

    @classmethod
    def load(cls) -> "Config":
        """Load config with the active session."""
        name = cls.get_active_session_name()
        if name:
            return cls(session=cls.load_session(name))
        return cls()

    def save(self) -> None:
        """Save the current session to disk and set it as active."""
        if self.session:
            session_file = self._get_session_file(self.session.name)
            session_file.write_text(json.dumps(asdict(self.session), indent=2))
            self.set_active_session(self.session.name)

    @classmethod
    def delete_session(cls, name: str) -> bool:
        """Delete a session by name. Returns True if deleted."""
        session_file = cls._get_session_file(name)
        if not session_file.exists():
            return False
        session_file.unlink()
        if cls.get_active_session_name() is None:
            # Active pointer was pointing to deleted session (now dangling)
            active_file = cls.get_active_session_file()
            if active_file.exists():
                active_file.unlink()
        return True

    @classmethod
    def get_session_state_dir(cls, name: str) -> Path:
        """Get per-session state directory (.nanorun/sessions/{name}/), creating if needed."""
        d = cls.get_sessions_dir() / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def clear_active(cls) -> None:
        """Clear the active session pointer (does not delete the session file)."""
        active_file = cls.get_active_session_file()
        if active_file.exists():
            active_file.unlink()


def infer_track_from_path(script_path: str) -> Optional[str]:
    """Infer track name from script path by checking parent directories for .track.json.

    For example, if script is 'experiments/records/train.py' and
    'experiments/records/.track.json' exists with name='records',
    returns 'records'.
    """
    repo_root = get_repo_root()
    path = Path(script_path)

    # Walk up from script directory looking for .track.json
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            return None

    # Check the directory containing the script
    script_dir = repo_root / path.parent
    track = Track.load(script_dir)
    if track:
        return track.name

    return None
