#!/usr/bin/env python3
"""Measure the native metric scales on real complexes.

Two of the structure-derived metrics here do not sit on the same scale as the
Rosetta measurements BindCraft's thresholds were set against, so their
thresholds cannot be copied across. This script is the measurement behind the
numbers quoted in ``docs/thresholds.md``, kept in the repository so those
numbers can be checked rather than taken on trust.

    python scripts/calibrate_thresholds.py external/boltzgen/example

For each multi-chain structure found, the smallest chain is treated as the
binder and the largest as the target, the twelve structure-derived metrics are
computed, and the distribution plus the rejection rate of each shipped
threshold is printed.

Caveats that apply to any run of this, stated in the docs too: these are
experimentally determined complexes, not designed binders; there is no decoy
set, so the output says where real interfaces sit and not what false-positive
rate a threshold buys; and the example set is small and antibody-heavy.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from foldexa_binder_eval import (  # noqa: E402
    Complex,
    compute_interface_geometry,
    load_profile,
)

# metric key in a profile -> attribute on InterfaceGeometry
MEASURED = {
    "shape_complementarity": "shape_complementarity",
    "delta_sasa": "delta_sasa",
    "n_interface_residues": "n_interface_residues_binder",
    "n_interface_hbonds": "n_interface_hbonds",
    "n_unsat_hbonds": "n_unsat_hbonds",
    "surface_hydrophobicity": "surface_hydrophobicity",
}

# Metrics where a bigger number is better. The rest are upper bounds.
HIGHER_IS_BETTER = {
    "shape_complementarity",
    "delta_sasa",
    "n_interface_residues",
    "n_interface_hbonds",
}

MIN_CHAIN_RESIDUES = 20
MAX_COMPLEX_RESIDUES = 2500
MIN_BURIED_AREA = 400.0


def collect(root: Path, verbose: bool = True) -> list[dict]:
    """Score every usable binder:target pair under ``root``."""
    rows: list[dict] = []
    seen: set[str] = set()

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".cif", ".pdb"):
            continue
        # The example sets repeat some structures across subdirectories.
        if path.name in seen:
            continue

        try:
            probe = Complex.from_file(path)
        except Exception:
            continue

        chains = list(dict.fromkeys(probe.residue_chain_ids.tolist()))
        if len(chains) < 2 or probe.n_residues > MAX_COMPLEX_RESIDUES:
            continue

        counts = Counter(probe.residue_chain_ids.tolist())
        ordered = sorted(chains, key=lambda c: counts[c])
        binder, target = ordered[0], ordered[-1]
        if binder == target:
            continue
        if counts[binder] < MIN_CHAIN_RESIDUES or counts[target] < MIN_CHAIN_RESIDUES:
            continue

        try:
            cx = Complex.from_file(path, binder_chain=binder, target_chains=target)
            geometry = compute_interface_geometry(cx)
        except Exception as exc:
            if verbose:
                print(f"  skipped {path.name}: {exc}")
            continue

        if (
            not np.isfinite(geometry.shape_complementarity)
            or geometry.delta_sasa < MIN_BURIED_AREA
        ):
            continue

        seen.add(path.name)
        row = {key: getattr(geometry, attr) for key, attr in MEASURED.items()}
        row["name"] = path.name
        row["pair"] = f"{binder}:{target}"
        rows.append(row)

        if verbose:
            print(
                f"{path.name:32s} {binder}:{target}  "
                f"SC={geometry.shape_complementarity:.3f} "
                f"dSASA={geometry.delta_sasa:7.0f} "
                f"nres={geometry.n_interface_residues_binder:3d} "
                f"hb={geometry.n_interface_hbonds:3d} "
                f"unsat={geometry.n_unsat_hbonds:3d} "
                f"phob={geometry.surface_hydrophobicity:.2f}"
            )
    return rows


def report(rows: list[dict], profiles: tuple[str, ...]) -> None:
    print(f"\n=== {len(rows)} unique interfaces ===\n")

    print(f"{'metric':26s} {'n':>3s} {'min':>8s} {'p10':>8s} "
          f"{'median':>8s} {'p90':>8s} {'max':>8s}")
    for key in MEASURED:
        values = np.array([r[key] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        print(
            f"{key:26s} {values.size:3d} {values.min():8.2f} "
            f"{np.percentile(values, 10):8.2f} {np.median(values):8.2f} "
            f"{np.percentile(values, 90):8.2f} {values.max():8.2f}"
        )

    for name in profiles:
        profile = load_profile(name)
        print(f"\n--- rejection rate, profile '{name}' ---")
        for key in MEASURED:
            threshold = profile.thresholds.get(key)
            if threshold is None:
                print(f"  {key:26s} not gated")
                continue
            values = np.array([r[key] for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            higher = key in HIGHER_IS_BETTER
            rejected = int((values < threshold).sum() if higher else (values > threshold).sum())
            print(
                f"  {key:26s} {'>=' if higher else '<='} {threshold:<8g} "
                f"rejects {rejected:2d}/{values.size} "
                f"({100.0 * rejected / values.size:3.0f}%)"
            )

    print(
        "\nThese are experimentally determined complexes, not designed binders,"
        "\nand there is no decoy set here. A rejection rate says how often a"
        "\nthreshold would discard a real interface, not what it buys in"
        "\nfalse positives avoided."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory of multi-chain structures, searched recursively.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        help="Profile to report rejection rates for; repeatable.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary.")
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")

    rows = collect(args.directory, verbose=not args.quiet)
    if not rows:
        print(
            f"No usable binder:target pairs found under {args.directory}.\n"
            "Expected multi-chain .cif or .pdb files with at least "
            f"{MIN_CHAIN_RESIDUES} residues per chain.",
            file=sys.stderr,
        )
        return 1

    report(rows, tuple(args.profile or ("default", "strict")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
