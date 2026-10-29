"""Run and parse Dunbrack's upstream ``ipsae.py``.

:mod:`foldexa_binder_eval.ipsae` reimplements the scoring natively, which is
what the pipeline uses. This adapter exists so the native numbers can be
checked against the reference implementation on your own models.

Note that ``tests/test_ipsae.py`` does not call this adapter: it compares
against the score files committed in the submodule, so the suite needs no
subprocess and no upstream dependencies. Running the script itself, through
:func:`run_ipsae`, is a manual cross-check.

The upstream script is pinned at ``external/ipsae``. Clone it with
``scripts/setup_upstream.sh``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["IpsaeResult", "run_ipsae", "parse_ipsae_output", "find_ipsae_script"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCRIPT = _REPO_ROOT / "external" / "ipsae" / "ipsae.py"

_COLUMNS = (
    "Chn1 Chn2 PAE Dist Type ipSAE ipSAE_d0chn ipSAE_d0dom ipTM_af ipTM_d0chn "
    "pDockQ pDockQ2 LIS n0res n0chn n0dom d0res d0chn d0dom nres1 nres2 "
    "dist1 dist2 Model"
).split()

_FLOAT_COLUMNS = {
    "ipSAE", "ipSAE_d0chn", "ipSAE_d0dom", "ipTM_af", "ipTM_d0chn",
    "pDockQ", "pDockQ2", "LIS", "d0res", "d0chn", "d0dom", "PAE", "Dist",
}
_INT_COLUMNS = {"n0res", "n0chn", "n0dom", "nres1", "nres2", "dist1", "dist2"}


@dataclass
class IpsaeResult:
    """One row of the upstream chain-pair score file."""

    chain1: str
    chain2: str
    type: str
    values: dict[str, float | int | str]

    @property
    def ipsae(self) -> float:
        return float(self.values["ipSAE"])

    @property
    def pdockq(self) -> float:
        return float(self.values["pDockQ"])

    @property
    def lis(self) -> float:
        return float(self.values["LIS"])


def find_ipsae_script(explicit: str | Path | None = None) -> Path:
    """Locate ``ipsae.py``: explicit path, then the submodule, then ``PATH``."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"ipsae.py not found at {path}")
        return path
    if _DEFAULT_SCRIPT.exists():
        return _DEFAULT_SCRIPT
    found = shutil.which("ipsae.py")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "ipsae.py not found. Run scripts/setup_upstream.sh to populate "
        f"{_DEFAULT_SCRIPT.parent}, or pass an explicit path."
    )


def parse_ipsae_output(path: str | Path) -> list[IpsaeResult]:
    """Parse the whitespace-delimited chain-pair file ipsae.py writes."""
    rows: list[IpsaeResult] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("Chn1") or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < len(_COLUMNS) - 1:
            continue
        record: dict[str, float | int | str] = {}
        for name, raw in zip(_COLUMNS, fields, strict=False):
            if name in _FLOAT_COLUMNS:
                record[name] = float(raw)
            elif name in _INT_COLUMNS:
                record[name] = int(float(raw))
            else:
                record[name] = raw
        rows.append(
            IpsaeResult(
                chain1=str(record["Chn1"]),
                chain2=str(record["Chn2"]),
                type=str(record["Type"]),
                values=record,
            )
        )
    return rows


def run_ipsae(
    pae_file: str | Path,
    structure_file: str | Path,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    script: str | Path | None = None,
    timeout: float = 900.0,
) -> list[IpsaeResult]:
    """Run upstream ``ipsae.py`` and return its chain-pair rows.

    The script writes its output files next to ``structure_file``, following
    the naming ``<stem>_<pae>_<dist>.txt``.
    """
    script_path = find_ipsae_script(script)
    structure_file = Path(structure_file)

    cmd = [
        sys.executable,
        str(script_path),
        str(pae_file),
        str(structure_file),
        str(int(pae_cutoff)),
        str(int(dist_cutoff)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ipsae.py failed (exit {proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr: {proc.stderr.strip()[:2000]}"
        )

    stem = structure_file.stem
    tag = f"{int(pae_cutoff)}_{int(dist_cutoff)}"
    candidates = sorted(structure_file.parent.glob(f"{stem}*{tag}.txt"))
    candidates = [p for p in candidates if "byres" not in p.name]
    if not candidates:
        raise FileNotFoundError(
            f"ipsae.py reported success but no score file matching "
            f"{stem}*{tag}.txt was found in {structure_file.parent}"
        )
    return parse_ipsae_output(candidates[0])
