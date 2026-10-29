"""Command line interface: file discovery, exit codes, subcommands."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from click.testing import CliRunner

from foldexa_binder_eval import Complex
from foldexa_binder_eval.cli import _auto_confidence, _job_key, main


# ----------------------------------------------------------------------
# confidence file discovery
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,scores",
    [
        # AlphaFold3: neither stem contains the other.
        ("fold_x_model_0", "fold_x_full_data_0"),
        ("fold_x_model_0", "fold_x_summary_confidences_0"),
        # Boltz.
        ("design_7_model_0", "pae_design_7_model_0"),
        ("design_7_model_0", "confidence_design_7_model_0"),
        ("design_7_model_0", "plddt_design_7_model_0"),
        # ColabFold.
        (
            "p_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000",
            "p_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000",
        ),
    ],
)
def test_job_keys_match_across_conventions(model, scores):
    assert _job_key(model) == _job_key(scores)


@pytest.mark.parametrize(
    "a,b",
    [
        ("design_7_model_0", "pae_design_9_model_0"),
        ("fold_abc_model_0", "fold_xyz_full_data_0"),
    ],
)
def test_job_keys_distinguish_different_jobs(a, b):
    assert _job_key(a) != _job_key(b)


def test_auto_confidence_finds_af3_sibling(tmp_path, two_chain_pdb):
    structure = tmp_path / "fold_job_model_0.pdb"
    shutil.copy(two_chain_pdb, structure)
    scores = tmp_path / "fold_job_full_data_0.json"
    scores.write_text("{}")
    (tmp_path / "unrelated_full_data_0.json").write_text("{}")

    found = _auto_confidence(structure, ())
    assert found == [scores]


def test_explicit_confidence_wins(tmp_path, two_chain_pdb):
    structure = tmp_path / "fold_job_model_0.pdb"
    shutil.copy(two_chain_pdb, structure)
    (tmp_path / "fold_job_full_data_0.json").write_text("{}")
    chosen = tmp_path / "elsewhere.json"
    chosen.write_text("{}")

    assert _auto_confidence(structure, (str(chosen),)) == [chosen]


def test_auto_confidence_returns_empty_when_nothing_matches(tmp_path, two_chain_pdb):
    structure = tmp_path / "fold_job_model_0.pdb"
    shutil.copy(two_chain_pdb, structure)
    assert _auto_confidence(structure, ()) == []


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------
@pytest.fixture
def runner():
    return CliRunner()


def _write_confidence(path, cx, pae_value=4.0, iptm=0.85):
    n = cx.n_residues
    pae = np.full((n, n), float(pae_value))
    np.fill_diagonal(pae, 0.4)
    path.write_text(
        json.dumps(
            {"pae": pae.tolist(), "plddt": [92.0] * n, "ptm": 0.8, "iptm": iptm}
        )
    )


def test_profiles_lists_builtins(runner):
    result = runner.invoke(main, ["profiles"])
    assert result.exit_code == 0
    for name in ("default", "strict", "nanobody", "screen"):
        assert name in result.output


def test_metrics_describes_all_twelve(runner):
    result = runner.invoke(main, ["metrics", "-p", "strict"])
    assert result.exit_code == 0
    assert "CONFIDENCE" in result.output
    assert "INTERFACE" in result.output
    assert "DEVELOPABILITY" in result.output
    assert result.output.count("source:") == 12


def test_score_exits_nonzero_on_failure(runner, tmp_path, two_chain_pdb):
    structure = tmp_path / "design_model_0.pdb"
    shutil.copy(two_chain_pdb, structure)
    cx = Complex.from_file(structure, binder_chain="B", target_chains="A")
    _write_confidence(tmp_path / "confidence_design_model_0.json", cx)

    result = runner.invoke(
        main, ["score", str(structure), "-b", "B", "-t", "A", "--no-sc"]
    )
    # No refold, so Refold RMSD is missing and the default profile fails it.
    assert result.exit_code == 1
    assert "VERDICT" in result.output


def test_score_quiet_prints_one_line(runner, tmp_path, two_chain_pdb):
    structure = tmp_path / "design_model_0.pdb"
    shutil.copy(two_chain_pdb, structure)
    cx = Complex.from_file(structure, binder_chain="B", target_chains="A")
    _write_confidence(tmp_path / "confidence_design_model_0.json", cx)

    result = runner.invoke(
        main, ["score", str(structure), "-b", "B", "-t", "A", "-q", "--no-sc"]
    )
    assert len(result.output.strip().splitlines()) == 1


def test_score_writes_requested_outputs(runner, tmp_path, two_chain_pdb):
    structure = tmp_path / "design_model_0.pdb"
    shutil.copy(two_chain_pdb, structure)
    cx = Complex.from_file(structure, binder_chain="B", target_chains="A")
    _write_confidence(tmp_path / "confidence_design_model_0.json", cx)

    out = tmp_path / "report"
    runner.invoke(
        main,
        [
            "score", str(structure), "-b", "B", "-t", "A", "--no-sc",
            "--json", str(out.with_suffix(".json")),
            "--csv", str(out.with_suffix(".csv")),
            "--html", str(out.with_suffix(".html")),
        ],
    )
    assert out.with_suffix(".json").exists()
    assert out.with_suffix(".csv").exists()
    assert out.with_suffix(".html").exists()
    json.loads(out.with_suffix(".json").read_text())


def test_batch_scores_a_directory(runner, tmp_path, two_chain_pdb):
    designs = tmp_path / "designs"
    designs.mkdir()
    for i in range(3):
        structure = designs / f"design_{i}_model_0.pdb"
        shutil.copy(two_chain_pdb, structure)
        cx = Complex.from_file(structure, binder_chain="B", target_chains="A")
        _write_confidence(designs / f"confidence_design_{i}_model_0.json", cx)

    out = tmp_path / "campaign"
    result = runner.invoke(
        main,
        [
            "batch", str(designs), "--pattern", "*.pdb",
            "-b", "B", "-t", "A", "--profile", "screen",
            "--no-sc", "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".csv").exists()
    assert out.with_suffix(".json").exists()
    assert out.with_suffix(".html").exists()

    payload = json.loads(out.with_suffix(".json").read_text())
    assert len(payload["designs"]) == 3


def test_batch_reports_when_nothing_matches(runner, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(main, ["batch", str(empty), "--pattern", "*.cif"])
    assert result.exit_code != 0
    assert "No files matching" in result.output


def test_unknown_profile_is_reported(runner, tmp_path, two_chain_pdb):
    structure = tmp_path / "design_model_0.pdb"
    shutil.copy(two_chain_pdb, structure)
    result = runner.invoke(
        main, ["score", str(structure), "-b", "B", "-t", "A", "-p", "nope"]
    )
    assert result.exit_code != 0
