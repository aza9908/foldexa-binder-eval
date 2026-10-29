"""The twelve-metric registry and end-to-end evaluation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from foldexa_binder_eval import (
    METRICS,
    METRICS_BY_KEY,
    Complex,
    evaluate_candidate,
    load_profile,
)
from foldexa_binder_eval.metrics import TIERS


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
def test_there_are_exactly_twelve_metrics():
    assert len(METRICS) == 12
    assert len(METRICS_BY_KEY) == 12


def test_every_metric_is_documented():
    for spec in METRICS:
        assert spec.label and spec.units and spec.source
        assert len(spec.description) > 40, f"{spec.key} needs a real description"


def test_tiers_are_valid_and_populated():
    assert {spec.tier for spec in METRICS} == set(TIERS)
    for tier in TIERS:
        assert any(spec.tier == tier for spec in METRICS)


def test_directions_are_valid():
    assert {spec.direction for spec in METRICS} == {"higher", "lower"}


@pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.key)
def test_pass_logic_respects_direction(spec):
    if spec.direction == "higher":
        assert spec.passes(10.0, 5.0)
        assert spec.passes(5.0, 5.0)  # inclusive
        assert not spec.passes(4.9, 5.0)
    else:
        assert spec.passes(1.0, 5.0)
        assert spec.passes(5.0, 5.0)  # inclusive
        assert not spec.passes(5.1, 5.0)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def _confidence_file(tmp_path, cx, pae_value=4.0, iptm=0.85):
    n = cx.n_residues
    pae = np.full((n, n), float(pae_value))
    np.fill_diagonal(pae, 0.4)
    payload = {
        "pae": pae.tolist(),
        "plddt": [92.0] * n,
        "ptm": 0.8,
        "iptm": iptm,
    }
    path = tmp_path / "confidence.json"
    path.write_text(json.dumps(payload))
    return path


def test_evaluation_covers_all_twelve(two_chain_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    conf = _confidence_file(tmp_path, cx)
    result = evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[conf],
        binder_chain="B",
        target_chains="A",
        thresholds=load_profile("default").thresholds,
    )
    assert set(result.values) == set(METRICS_BY_KEY)


def test_missing_refold_blocks_a_pass(two_chain_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    conf = _confidence_file(tmp_path, cx)
    result = evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[conf],
        binder_chain="B",
        target_chains="A",
        thresholds=load_profile("default").thresholds,
        missing_policy="fail",
    )
    assert result.values["binder_rmsd"].status == "missing"
    assert result.verdict != "PASS"


def test_skip_policy_reports_incomplete(two_chain_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    conf = _confidence_file(tmp_path, cx)
    thresholds = dict.fromkeys(METRICS_BY_KEY, None)
    thresholds["binder_rmsd"] = 2.5
    result = evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[conf],
        binder_chain="B",
        target_chains="A",
        thresholds=thresholds,
        missing_policy="skip",
    )
    assert result.verdict == "INCOMPLETE"


def test_refold_supplies_the_rmsd_metric(two_chain_pdb, refold_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    conf = _confidence_file(tmp_path, cx)
    result = evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[conf],
        refold=refold_pdb,
        binder_chain="B",
        target_chains="A",
        thresholds=load_profile("default").thresholds,
        rmsd_align_on="target",
    )
    assert result.values["binder_rmsd"].value == pytest.approx(1.0, abs=1e-3)
    assert result.values["binder_rmsd"].passed is True


def test_ungated_metric_is_reported_not_enforced(two_chain_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    conf = _confidence_file(tmp_path, cx)
    thresholds = dict.fromkeys(METRICS_BY_KEY, None)
    result = evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[conf],
        binder_chain="B",
        target_chains="A",
        thresholds=thresholds,
    )
    assert result.verdict == "PASS"
    assert result.values["delta_sasa"].status == "unfiltered"
    assert result.values["delta_sasa"].value is not None


def test_failure_is_attributed_to_the_right_metric(two_chain_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    conf = _confidence_file(tmp_path, cx)
    thresholds = dict.fromkeys(METRICS_BY_KEY, None)
    thresholds["complex_iptm"] = 0.99  # unreachable
    result = evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[conf],
        binder_chain="B",
        target_chains="A",
        thresholds=thresholds,
    )
    assert result.verdict == "FAIL"
    assert [m.spec.key for m in result.failures] == ["complex_iptm"]


def test_works_without_a_confidence_file(two_chain_pdb):
    result = evaluate_candidate(
        structure=two_chain_pdb,
        binder_chain="B",
        target_chains="A",
        thresholds=dict.fromkeys(METRICS_BY_KEY, None),
    )
    # pLDDT falls back to the B-factor column; PAE metrics cannot be computed.
    assert result.values["binder_plddt"].value == pytest.approx(88.0, abs=0.5)
    assert result.values["ipsae"].status == "missing"
    assert "no confidence file" in result.values["ipsae"].note


def test_mismatched_confidence_is_rejected(two_chain_pdb, tmp_path):
    payload = {"pae": np.zeros((7, 7)).tolist(), "plddt": [90.0] * 7}
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tokens"):
        evaluate_candidate(
            structure=two_chain_pdb,
            confidence=[path],
            binder_chain="B",
            target_chains="A",
            thresholds=dict.fromkeys(METRICS_BY_KEY, None),
        )


def test_row_export_is_flat(two_chain_pdb, tmp_path):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    conf = _confidence_file(tmp_path, cx)
    result = evaluate_candidate(
        structure=two_chain_pdb,
        confidence=[conf],
        binder_chain="B",
        target_chains="A",
        thresholds=load_profile("default").thresholds,
    )
    row = result.to_row()
    assert row["design_id"] == "complex"
    assert row["verdict"] in {"PASS", "FAIL", "INCOMPLETE"}
    for key in METRICS_BY_KEY:
        assert key in row and f"{key}__pass" in row
