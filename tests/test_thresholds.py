"""Threshold profiles: loading, inheritance, validation."""

from __future__ import annotations

import pytest
import yaml

from foldexa_binder_eval import builtin_profiles, load_profile
from foldexa_binder_eval.metrics import METRICS_BY_KEY


def test_all_builtin_profiles_load():
    profiles = builtin_profiles()
    assert {"default", "strict", "nanobody", "screen"} <= set(profiles)
    for name in profiles:
        assert load_profile(name).name


def test_default_gates_every_metric():
    profile = load_profile("default")
    assert set(profile.thresholds) == set(METRICS_BY_KEY)
    assert len(profile.active()) == 12


def test_profiles_only_reference_known_metrics():
    for name in builtin_profiles():
        profile = load_profile(name)
        assert set(profile.thresholds) <= set(METRICS_BY_KEY)


def test_strict_is_at_least_as_tight_as_default():
    default = load_profile("default")
    strict = load_profile("strict")
    for key, spec in METRICS_BY_KEY.items():
        lo, hi = default.thresholds[key], strict.thresholds[key]
        if lo is None or hi is None:
            continue
        if spec.direction == "higher":
            assert hi >= lo, f"{key}: strict {hi} is looser than default {lo}"
        else:
            assert hi <= lo, f"{key}: strict {hi} is looser than default {lo}"


def test_screen_is_no_tighter_than_default():
    default = load_profile("default")
    screen = load_profile("screen")
    for key, spec in METRICS_BY_KEY.items():
        lo, hi = default.thresholds[key], screen.thresholds[key]
        if lo is None or hi is None:
            continue
        if spec.direction == "higher":
            assert hi <= lo, f"{key}: screen {hi} is tighter than default {lo}"
        else:
            assert hi >= lo, f"{key}: screen {hi} is tighter than default {lo}"


def test_inheritance_fills_unstated_values():
    default = load_profile("default")
    nanobody = load_profile("nanobody")
    # Not restated by the nanobody profile, so inherited unchanged.
    assert nanobody.thresholds["binder_plddt"] == default.thresholds["binder_plddt"]
    # Restated.
    assert nanobody.thresholds["shape_complementarity"] != default.thresholds[
        "shape_complementarity"
    ]
    assert nanobody.lineage == ["default"]


def test_nanobody_aligns_rmsd_on_binder():
    assert load_profile("nanobody").rmsd_align_on == "binder"


def test_screen_skips_missing_metrics():
    assert load_profile("screen").missing_policy == "skip"


def test_null_disables_a_gate():
    screen = load_profile("screen")
    assert screen.thresholds["binder_rmsd"] is None
    assert "binder_rmsd" not in screen.active()


def test_unknown_metric_is_rejected(tmp_path):
    path = tmp_path / "thresholds.bad.yaml"
    path.write_text(yaml.safe_dump({"thresholds": {"not_a_metric": 1.0}}))
    with pytest.raises(ValueError, match="unknown metric"):
        load_profile(path)


def test_bad_align_mode_is_rejected(tmp_path):
    path = tmp_path / "thresholds.bad.yaml"
    path.write_text(yaml.safe_dump({"rmsd_align_on": "sideways", "thresholds": {}}))
    with pytest.raises(ValueError, match="rmsd_align_on"):
        load_profile(path)


def test_bad_missing_policy_is_rejected(tmp_path):
    path = tmp_path / "thresholds.bad.yaml"
    path.write_text(yaml.safe_dump({"missing_policy": "maybe", "thresholds": {}}))
    with pytest.raises(ValueError, match="missing_policy"):
        load_profile(path)


def test_missing_profile_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="Built-in profiles"):
        load_profile("does-not-exist")


def test_circular_extends_is_detected(tmp_path):
    a = tmp_path / "thresholds.a.yaml"
    b = tmp_path / "thresholds.b.yaml"
    a.write_text(yaml.safe_dump({"extends": str(b), "thresholds": {}}))
    b.write_text(yaml.safe_dump({"extends": str(a), "thresholds": {}}))
    with pytest.raises(ValueError, match="Circular"):
        load_profile(a)
