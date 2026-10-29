"""Threshold profiles.

A profile is a YAML file mapping metric keys to threshold values. ``null``
disables a metric (it is still computed and reported, just not used as a
gate). Profiles may inherit from another profile with ``extends:``, so a
project-specific file only has to state what it changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .metrics import METRICS_BY_KEY

__all__ = [
    "ThresholdProfile",
    "load_profile",
    "builtin_profiles",
    "PACKAGE_PROFILE_DIR",
    "search_dirs",
]

# Profiles shipped with the package (installed with the wheel).
PACKAGE_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"

_BUILTIN_SUFFIX = ".yaml"
_BUILTIN_PREFIX = "thresholds."


def search_dirs() -> list[Path]:
    """Directories searched for named profiles, highest priority first.

    A ``config/`` directory next to the repo root wins over the packaged
    copies, so a checkout can override a shipped profile by editing the file
    in place without reinstalling.
    """
    candidates = [Path.cwd() / "config"]
    repo_config = Path(__file__).resolve().parents[2] / "config"
    candidates.append(repo_config)
    candidates.append(PACKAGE_PROFILE_DIR)

    seen: list[Path] = []
    for path in candidates:
        if path.is_dir() and path not in seen:
            seen.append(path)
    return seen


@dataclass
class ThresholdProfile:
    name: str
    thresholds: dict[str, float | None]
    pae_cutoff: float = 10.0
    dist_cutoff: float = 10.0
    rmsd_align_on: str = "target"
    missing_policy: str = "fail"
    description: str = ""
    path: Path | None = None
    lineage: list[str] = field(default_factory=list)

    def active(self) -> dict[str, float]:
        return {k: v for k, v in self.thresholds.items() if v is not None}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "pae_cutoff": self.pae_cutoff,
            "dist_cutoff": self.dist_cutoff,
            "rmsd_align_on": self.rmsd_align_on,
            "missing_policy": self.missing_policy,
            "thresholds": self.thresholds,
        }


def builtin_profiles() -> dict[str, Path]:
    """Map profile name -> path, first directory in :func:`search_dirs` wins."""
    out: dict[str, Path] = {}
    for directory in search_dirs():
        for path in sorted(directory.glob(f"{_BUILTIN_PREFIX}*{_BUILTIN_SUFFIX}")):
            name = path.name[len(_BUILTIN_PREFIX) : -len(_BUILTIN_SUFFIX)]
            out.setdefault(name, path)
    return out


def _resolve(spec: str | Path) -> Path:
    path = Path(spec)
    if path.exists():
        return path
    built = builtin_profiles()
    if str(spec) in built:
        return built[str(spec)]
    known = ", ".join(sorted(built)) or "none found"
    raise FileNotFoundError(
        f"Threshold profile {spec!r} not found as a file or a built-in. "
        f"Built-in profiles: {known}"
    )


def load_profile(
    spec: str | Path = "default", _seen: tuple[str, ...] = ()
) -> ThresholdProfile:
    """Load a profile by name or path, resolving ``extends:`` chains."""
    path = _resolve(spec)
    resolved = str(path.resolve())
    if resolved in _seen:
        chain = " -> ".join(list(_seen) + [resolved])
        raise ValueError(f"Circular 'extends' in threshold profiles: {chain}")

    with path.open() as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    parent: ThresholdProfile | None = None
    if payload.get("extends"):
        parent = load_profile(payload["extends"], _seen + (resolved,))

    thresholds: dict[str, float | None] = (
        dict(parent.thresholds) if parent else {k: None for k in METRICS_BY_KEY}
    )

    raw = payload.get("thresholds") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"'thresholds' in {path} must be a mapping")

    unknown = sorted(set(raw) - set(METRICS_BY_KEY))
    if unknown:
        raise ValueError(
            f"{path} sets unknown metric(s): {', '.join(unknown)}. "
            f"Known metrics: {', '.join(sorted(METRICS_BY_KEY))}"
        )
    for key, value in raw.items():
        thresholds[key] = None if value is None else float(value)

    def pick(field_name: str, fallback):
        if field_name in payload and payload[field_name] is not None:
            return payload[field_name]
        return getattr(parent, field_name) if parent else fallback

    profile = ThresholdProfile(
        name=str(payload.get("name") or path.stem.replace(_BUILTIN_PREFIX, "")),
        thresholds=thresholds,
        pae_cutoff=float(pick("pae_cutoff", 10.0)),
        dist_cutoff=float(pick("dist_cutoff", 10.0)),
        rmsd_align_on=str(pick("rmsd_align_on", "target")),
        missing_policy=str(pick("missing_policy", "fail")),
        description=str(payload.get("description") or (parent.description if parent else "")),
        path=path,
        lineage=(parent.lineage + [parent.name]) if parent else [],
    )

    if profile.rmsd_align_on not in {"target", "binder"}:
        raise ValueError(
            f"{path}: rmsd_align_on must be 'target' or 'binder', "
            f"got {profile.rmsd_align_on!r}"
        )
    if profile.missing_policy not in {"fail", "skip"}:
        raise ValueError(
            f"{path}: missing_policy must be 'fail' or 'skip', "
            f"got {profile.missing_policy!r}"
        )
    return profile
