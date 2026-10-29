"""Command line interface."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click

from . import __version__
from .metrics import METRICS, TIERS, evaluate_candidate
from .report import render_table, write_csv, write_html, write_json
from .thresholds import builtin_profiles, load_profile

CONFIDENCE_SUFFIXES = (".json", ".npz")
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".mmcif", ".bcif", ".ent")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="foldexa-binder-eval")
def main() -> None:
    """Score binder candidates against a target on twelve metrics.

    Collects the confidence output of a structure predictor (RoseTTAFold3,
    BoltzGen/Boltz, AlphaFold2/3), the interface-aware PAE scores of ipSAE,
    and structure-derived interface and developability metrics into one
    pass/fail decision.
    """


# ----------------------------------------------------------------------
# Affixes the predictors bolt onto a shared job name. Stripping them yields a
# key that is identical for a model and its confidence files.
#
#   AlphaFold3  fold_x_model_0.cif   fold_x_full_data_0.json
#                                    fold_x_summary_confidences_0.json
#   Boltz       x_model_0.cif        pae_x_model_0.npz
#                                    confidence_x_model_0.json
#   ColabFold   x_unrelaxed_rank_001_..._model_1.pdb
#                                    x_scores_rank_001_..._model_1.json
_STRIP_PATTERNS = (
    r"^(pae|plddt|confidence|scores|summary_confidences|full_data)_",
    r"_(full_data|summary_confidences|confidences|scores|model|sample|pae|plddt)",
    r"_(unrelaxed|relaxed)",
)


def _job_key(name: str) -> str:
    """Reduce a file stem to the identifier a model shares with its scores."""
    key = name.lower()
    for pattern in _STRIP_PATTERNS:
        key = re.sub(pattern, "_", key)
    # Trailing sample indices differ in position between conventions.
    key = re.sub(r"[_\-]*\d+$", "", key)
    return re.sub(r"[_\-]+", "_", key).strip("_")


def _auto_confidence(structure: Path, explicit: tuple[str, ...]) -> list[Path]:
    """Find the confidence files that belong to this structure.

    Matching is on the shared job key rather than a substring, because no
    predictor names a model and its scores identically: AlphaFold3 writes
    ``fold_x_model_0.cif`` beside ``fold_x_full_data_0.json``, where neither
    stem contains the other.
    """
    if explicit:
        return [Path(p) for p in explicit]

    target_key = _job_key(structure.stem)
    if not target_key:
        return []

    exact: list[Path] = []
    loose: list[Path] = []
    for path in sorted(structure.parent.iterdir()):
        if path.suffix.lower() not in CONFIDENCE_SUFFIXES:
            continue
        if structure.stem in path.stem or path.stem in structure.stem:
            exact.append(path)
        elif _job_key(path.stem) == target_key:
            loose.append(path)
    return exact or loose


@main.command()
@click.argument("structure", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--confidence",
    "-c",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Predictor confidence file (.json / .npz). Repeatable. "
    "Auto-detected from the structure's directory when omitted.",
)
@click.option(
    "--refold",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Independent refold of the binder sequence, for the RMSD metric.",
)
@click.option("--binder-chain", "-b", default=None, help="Binder chain id. Default: first chain.")
@click.option(
    "--target-chains",
    "-t",
    default=None,
    help="Comma-separated target chain ids. Default: every other chain.",
)
@click.option("--profile", "-p", default="default", show_default=True, help="Threshold profile name or path.")
@click.option("--json", "json_out", type=click.Path(path_type=Path), help="Write a JSON report here.")
@click.option("--csv", "csv_out", type=click.Path(path_type=Path), help="Write a CSV row here.")
@click.option("--html", "html_out", type=click.Path(path_type=Path), help="Write an HTML report here.")
@click.option("--no-sc", is_flag=True, help="Skip shape complementarity (the slowest metric).")
@click.option("--quiet", "-q", is_flag=True, help="Print only the verdict.")
def score(
    structure: Path,
    confidence: tuple[str, ...],
    refold: Path | None,
    binder_chain: str | None,
    target_chains: str | None,
    profile: str,
    json_out: Path | None,
    csv_out: Path | None,
    html_out: Path | None,
    no_sc: bool,
    quiet: bool,
) -> None:
    """Score a single binder:target complex.

    STRUCTURE is the predicted complex (.pdb / .cif).
    """
    prof = load_profile(profile)
    conf_paths = _auto_confidence(structure, confidence)

    evaluation = evaluate_candidate(
        structure=structure,
        confidence=conf_paths or None,
        refold=refold,
        thresholds=prof.thresholds,
        binder_chain=binder_chain,
        target_chains=target_chains,
        pae_cutoff=prof.pae_cutoff,
        dist_cutoff=prof.dist_cutoff,
        rmsd_align_on=prof.rmsd_align_on,
        compute_sc=not no_sc,
        missing_policy=prof.missing_policy,
        profile=prof.name,
    )

    if quiet:
        click.echo(f"{evaluation.design_id}\t{evaluation.verdict}")
    else:
        click.echo(render_table(evaluation, prof))
        if not conf_paths:
            click.secho(
                "  note: no confidence file found; the PAE-derived metrics "
                "could not be computed. Pass one with --confidence.",
                fg="yellow",
            )

    if json_out:
        click.echo(f"wrote {write_json([evaluation], json_out, prof)}")
    if csv_out:
        click.echo(f"wrote {write_csv([evaluation], csv_out)}")
    if html_out:
        click.echo(f"wrote {write_html([evaluation], html_out, prof)}")

    sys.exit(0 if evaluation.verdict == "PASS" else 1)


# ----------------------------------------------------------------------
@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--pattern", default="*.cif", show_default=True, help="Glob for candidate structures.")
@click.option("--binder-chain", "-b", default=None, help="Binder chain id.")
@click.option("--target-chains", "-t", default=None, help="Comma-separated target chain ids.")
@click.option("--profile", "-p", default="default", show_default=True)
@click.option("--refold-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory of refolds named after each candidate.")
@click.option("--out", "-o", type=click.Path(path_type=Path), default=Path("binder_eval"),
              show_default=True, help="Output prefix; writes .csv, .json and .html.")
@click.option("--no-sc", is_flag=True, help="Skip shape complementarity.")
@click.option("--fail-fast", is_flag=True, help="Stop on the first candidate that errors.")
def batch(
    directory: Path,
    pattern: str,
    binder_chain: str | None,
    target_chains: str | None,
    profile: str,
    refold_dir: Path | None,
    out: Path,
    no_sc: bool,
    fail_fast: bool,
) -> None:
    """Score every candidate in DIRECTORY and write a ranked report."""
    prof = load_profile(profile)
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise click.ClickException(
            f"No files matching {pattern!r} in {directory}. "
            f"Try --pattern '*.pdb'."
        )

    evaluations = []
    errors: list[tuple[Path, str]] = []
    with click.progressbar(candidates, label="scoring") as bar:
        for path in bar:
            try:
                refold = None
                if refold_dir:
                    matches = [
                        p
                        for p in refold_dir.iterdir()
                        if p.suffix.lower() in STRUCTURE_SUFFIXES and path.stem in p.stem
                    ]
                    refold = matches[0] if matches else None
                evaluations.append(
                    evaluate_candidate(
                        structure=path,
                        confidence=_auto_confidence(path, ()) or None,
                        refold=refold,
                        thresholds=prof.thresholds,
                        binder_chain=binder_chain,
                        target_chains=target_chains,
                        pae_cutoff=prof.pae_cutoff,
                        dist_cutoff=prof.dist_cutoff,
                        rmsd_align_on=prof.rmsd_align_on,
                        compute_sc=not no_sc,
                        missing_policy=prof.missing_policy,
                        profile=prof.name,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                if fail_fast:
                    raise click.ClickException(f"{path.name}: {exc}") from exc
                errors.append((path, str(exc)))

    if not evaluations:
        raise click.ClickException("No candidate could be scored.")

    passing = [e for e in evaluations if e.verdict == "PASS"]
    click.echo("")
    click.echo(f"  {len(passing)} / {len(evaluations)} candidates pass profile '{prof.name}'")
    for ev in sorted(evaluations, key=lambda e: (e.verdict != "PASS", e.design_id)):
        mark = "PASS" if ev.verdict == "PASS" else ev.verdict
        detail = ", ".join(v.spec.key for v in ev.failures)
        click.echo(f"    {mark:<10} {ev.design_id}" + (f"   failed: {detail}" if detail else ""))

    if errors:
        click.echo("")
        click.secho(f"  {len(errors)} candidate(s) could not be scored:", fg="yellow")
        for path, message in errors:
            click.echo(f"    {path.name}: {message}")

    click.echo("")
    click.echo(f"wrote {write_csv(evaluations, out.with_suffix('.csv'))}")
    click.echo(f"wrote {write_json(evaluations, out.with_suffix('.json'), prof)}")
    click.echo(f"wrote {write_html(evaluations, out.with_suffix('.html'), prof)}")


# ----------------------------------------------------------------------
@main.command("profiles")
def list_profiles() -> None:
    """List available threshold profiles."""
    found = builtin_profiles()
    if not found:
        click.echo("No profiles found.")
        return
    for name, path in found.items():
        prof = load_profile(path)
        active = len(prof.active())
        click.echo(f"  {name:<12} {active:>2}/12 gated   {path}")
        if prof.description:
            first = " ".join(prof.description.split())
            click.echo(f"               {first[:110]}")


@main.command("metrics")
@click.option("--profile", "-p", default="default", show_default=True)
def describe_metrics(profile: str) -> None:
    """Describe the twelve metrics and the active thresholds."""
    prof = load_profile(profile)
    for tier in TIERS:
        click.echo("")
        click.secho(f"  {tier.upper()}", bold=True)
        for spec in (m for m in METRICS if m.tier == tier):
            threshold = prof.thresholds.get(spec.key)
            arrow = ">=" if spec.direction == "higher" else "<="
            gate = f"{arrow} {threshold}" if threshold is not None else "not gated"
            click.echo(f"    {spec.label:<28} {gate:<14} [{spec.units}]")
            click.echo(f"      {' '.join(spec.description.split())}")
            click.echo(f"      source: {spec.source}")


if __name__ == "__main__":  # pragma: no cover
    main()
