"""Confidence parsing, token alignment, and report rendering."""

from __future__ import annotations

import json

import numpy as np
import pytest

from foldexa_binder_eval import Complex, evaluate_candidate, load_confidence, load_profile
from foldexa_binder_eval.metrics import METRICS_BY_KEY
from foldexa_binder_eval.report import (
    evaluations_to_frame,
    render_table,
    write_csv,
    write_html,
    write_json,
)


# ----------------------------------------------------------------------
# confidence parsing
# ----------------------------------------------------------------------
def test_reads_af2_style_json(tmp_path):
    payload = {"pae": np.full((4, 4), 7.0).tolist(), "plddt": [80, 85, 90, 95],
               "ptm": 0.7, "iptm": 0.6}
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload))
    conf = load_confidence(path)
    assert conf.pae.shape == (4, 4)
    assert conf.iptm == pytest.approx(0.6)
    assert conf.format == "af2-style"


def test_colabfold_list_wrapper_is_unwrapped(tmp_path):
    payload = [{"predicted_aligned_error": np.zeros((3, 3)).tolist(), "plddt": [90] * 3}]
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload))
    assert load_confidence(path).pae.shape == (3, 3)


def test_plddt_on_zero_to_one_scale_is_rescaled(tmp_path):
    payload = {"plddt": [0.9, 0.8, 0.95]}
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload))
    conf = load_confidence(path)
    assert conf.plddt.max() == pytest.approx(95.0)


def test_plddt_already_on_percent_scale_is_left_alone(tmp_path):
    payload = {"plddt": [90.0, 80.0, 95.0]}
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload))
    assert load_confidence(path).plddt.max() == pytest.approx(95.0)


def test_reads_npz(tmp_path):
    path = tmp_path / "pae.npz"
    np.savez(path, pae=np.full((5, 5), 9.0))
    conf = load_confidence(path)
    assert conf.pae.shape == (5, 5)
    assert conf.format == "boltz-npz"


def test_merging_fills_gaps_without_overwriting(tmp_path):
    npz = tmp_path / "pae.npz"
    np.savez(npz, pae=np.full((3, 3), 5.0))
    js = tmp_path / "confidence.json"
    js.write_text(json.dumps({"iptm": 0.77, "ptm": 0.8}))
    conf = load_confidence(npz, js)
    assert conf.pae.shape == (3, 3)
    assert conf.iptm == pytest.approx(0.77)
    assert len(conf.source) == 2


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_confidence(tmp_path / "nope.json")


def test_unsupported_extension_is_reported(tmp_path):
    path = tmp_path / "scores.txt"
    path.write_text("nothing")
    with pytest.raises(ValueError, match="Unsupported confidence format"):
        load_confidence(path)


def test_non_square_pae_is_rejected(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"pae": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}))
    with pytest.raises(ValueError, match="not square"):
        load_confidence(path)


def test_token_alignment_drops_ligand_tokens(tmp_path, two_chain_pdb):
    """A PAE matrix carrying extra ligand tokens maps onto polymer residues."""
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    n = cx.n_residues

    chain_ids, res_ids = [], []
    for residue in cx.residues:
        chain_ids.append(residue.chain_id)
        res_ids.append(residue.res_id)
    # Three extra ligand tokens on a chain that is not in the structure.
    chain_ids += ["L", "L", "L"]
    res_ids += [1, 1, 1]

    size = n + 3
    pae = np.full((size, size), 20.0)
    pae[:n, :n] = 6.0
    payload = {
        "pae": pae.tolist(),
        "token_chain_ids": chain_ids,
        "token_res_ids": res_ids,
    }
    path = tmp_path / "full_data.json"
    path.write_text(json.dumps(payload))

    aligned = load_confidence(path).align_to(cx)
    assert aligned.pae.shape == (n, n)
    assert np.allclose(aligned.pae, 6.0)


def test_token_alignment_reports_unmatched_residues(tmp_path, two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    payload = {
        "pae": np.zeros((2, 2)).tolist(),
        "token_chain_ids": ["A", "A"],
        "token_res_ids": [1, 2],
    }
    path = tmp_path / "full_data.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no matching token"):
        load_confidence(path).align_to(cx)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
@pytest.fixture
def evaluation(two_chain_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    pae = np.full((cx.n_residues, cx.n_residues), 4.0)
    np.fill_diagonal(pae, 0.4)
    path = tmp_path / "confidence.json"
    path.write_text(json.dumps({"pae": pae.tolist(), "plddt": [92.0] * cx.n_residues,
                                "iptm": 0.85, "ptm": 0.8}))
    return evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[path],
        binder_chain="B",
        target_chains="A",
        thresholds=load_profile("default").thresholds,
    )


def test_table_lists_every_metric(evaluation):
    text = render_table(evaluation, load_profile("default"))
    for spec in METRICS_BY_KEY.values():
        assert spec.label in text
    assert "VERDICT" in text


def test_frame_has_one_row_per_evaluation(evaluation):
    frame = evaluations_to_frame([evaluation, evaluation])
    assert len(frame) == 2
    assert "verdict" in frame.columns


def test_csv_round_trips(evaluation, tmp_path):
    import pandas as pd

    path = write_csv([evaluation], tmp_path / "out.csv")
    frame = pd.read_csv(path)
    assert len(frame) == 1
    assert frame.loc[0, "design_id"] == "complex"


def test_json_report_is_valid_and_complete(evaluation, tmp_path):
    path = write_json([evaluation], tmp_path / "out.json", load_profile("default"))
    payload = json.loads(path.read_text())
    assert payload["profile"]["name"] == "default"
    design = payload["designs"][0]
    assert set(design["metrics"]) == set(METRICS_BY_KEY)
    assert design["verdict"] in {"PASS", "FAIL", "INCOMPLETE"}


def test_json_has_no_nan_literals(evaluation, tmp_path):
    path = write_json([evaluation], tmp_path / "out.json")
    raw = path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)  # strict parse


def test_html_is_standalone_and_escaped(evaluation, tmp_path):
    path = write_html([evaluation], tmp_path / "out.html", load_profile("default"))
    html = path.read_text()
    assert html.startswith("<!doctype html>")
    assert "prefers-color-scheme" in html
    assert evaluation.design_id in html
    assert "<script" not in html.lower()
