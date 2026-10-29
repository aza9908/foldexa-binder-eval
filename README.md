# foldexa-binder-eval

Twelve-metric acceptance testing for designed protein binders.

Binder design produces thousands of candidates and almost all of them are
wrong. The expensive part is not generating them, it is deciding which few to
express. This repository collects the tools that decision depends on
(RoseTTAFold3, BoltzGen, ipSAE, BindCraft), computes twelve metrics for a
binder against its target, applies thresholds, and returns a pass or a fail
with the reason attached.

```
PASS Binder pLDDT                         91.400    >= 80.000
PASS Complex ipTM                          0.850     >= 0.500
PASS ipSAE                                 0.712     >= 0.500
PASS Interface PAE                         4.210     <= 10.000
PASS pDockQ                                0.481     >= 0.230
PASS Buried surface area                    1640       >= 1000
PASS Interface residues                       14          >= 7
PASS Interface H-bonds                         6          >= 3
PASS Buried unsatisfied H-bonds                2          <= 4
PASS Shape complementarity                 0.512     >= 0.450
PASS Surface hydrophobicity                0.268     <= 0.350
FAIL Refold RMSD                           3.100      <= 2.500

VERDICT: FAIL   (11/12 gated metrics passed)
failed: Refold RMSD
```

## What it collects

| Tool | Role here | Pinned at |
|---|---|---|
| [RoseTTAFold3](https://github.com/RosettaCommons/foundry) (`models/rf3`) | Structure prediction; emits AF3-style PAE/pLDDT | `external/rf3-foundry` |
| [BoltzGen](https://github.com/HannesStark/boltzgen) | Binder generation and independent refolds | `external/boltzgen` |
| [ipSAE](https://github.com/DunbrackLab/IPSAE) | Interface-restricted PAE scoring (reference implementation) | `external/ipsae` |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | Source of the default filter thresholds | `external/bindcraft` |

The upstream tools are git submodules, not vendored copies, so they stay at a
pinned commit and update on your terms. Nothing here reimplements structure
prediction. What this package adds is the scoring and gating layer that sits
downstream of all of them, plus a native ipSAE implementation that runs
in-process over a whole campaign instead of one subprocess per model.

## The twelve metrics

Three tiers. A candidate passes only if every gated metric passes.

**Prediction confidence — does the model believe the complex?**

| Metric | Direction | Default | Source |
|---|---|---|---|
| Binder pLDDT | ≥ | 80 | BindCraft `Average_Binder_pLDDT` |
| Complex ipTM | ≥ | 0.50 | BindCraft `Average_i_pTM` |
| ipSAE | ≥ | 0.50 | [Dunbrack 2025](https://doi.org/10.1101/2025.02.10.637595) |
| Interface PAE | ≤ | 10.0 Å | BindCraft `Average_i_pAE` |
| pDockQ | ≥ | 0.23 | [Bryant 2022](https://www.nature.com/articles/s41467-022-28865-w) |

**Interface quality — is the interface physically real?**

| Metric | Direction | Default | Source |
|---|---|---|---|
| Buried surface area | ≥ | 1000 Å² | de novo binder literature |
| Interface residues | ≥ | 7 | BindCraft |
| Interface H-bonds | ≥ | 3 | BindCraft |
| Buried unsatisfied H-bonds | ≤ | 4 | BindCraft |
| Shape complementarity | ≥ | 0.45 | Lawrence & Colman 1993, recalibrated |

**Developability — will it survive the lab?**

| Metric | Direction | Default | Source |
|---|---|---|---|
| Surface hydrophobicity | ≤ | 0.35 | BindCraft |
| Refold RMSD | ≤ | 2.5 Å | BoltzGen `refolding_rmsd_threshold` |

`docs/metrics.md` explains what each one actually measures and why it earns a
slot. `docs/thresholds.md` shows where every number came from, including the
two that had to be recalibrated against real structures rather than copied.

## Install

```bash
git clone --recurse-submodules https://github.com/aza9908/foldexa-binder-eval.git
cd foldexa-binder-eval
pip install -e .
```

Already cloned without submodules, or want them later:

```bash
scripts/setup_upstream.sh          # ipsae + bindcraft, ~12 MB
scripts/setup_upstream.sh --all    # adds boltzgen and rf3-foundry, ~300 MB
```

Runtime dependencies are numpy, scipy, pandas, biotite, pyyaml and click.
There is no PyRosetta requirement: shape complementarity, buried surface,
hydrogen bonds and buried unsatisfied polars all have native implementations
here. That is a deliberate trade — see the calibration note in
`docs/thresholds.md`.

## Use

Score one candidate:

```bash
foldexa-binder-eval score designs/design_0.cif \
    --confidence designs/confidence_design_0.json \
    --refold refolds/design_0.cif \
    --binder-chain B --target-chains A
```

The confidence file is auto-detected when it sits beside the structure and
shares its name, so in a normal predictor output directory you can leave
`--confidence` off entirely.

Score a campaign and get a ranked report:

```bash
foldexa-binder-eval batch designs/ \
    --pattern '*.cif' --binder-chain B \
    --refold-dir refolds/ --profile strict \
    --out results/campaign
```

That writes `campaign.csv`, `campaign.json` and a standalone
`campaign.html`. The command exits non-zero when a candidate fails, so it
drops into CI or a Makefile without extra glue.

Inspect what is being applied:

```bash
foldexa-binder-eval profiles          # available threshold sets
foldexa-binder-eval metrics -p strict # the twelve, with active thresholds
```

From Python:

```python
from foldexa_binder_eval import evaluate_candidate, load_profile

profile = load_profile("nanobody")
result = evaluate_candidate(
    structure="designs/vhh_12.cif",
    confidence=["designs/confidence_vhh_12.json"],
    refold="refolds/vhh_12.cif",
    thresholds=profile.thresholds,
    binder_chain="B",
    rmsd_align_on=profile.rmsd_align_on,
)

print(result.verdict)
for failure in result.failures:
    print(failure.spec.label, failure.value, failure.threshold)
```

## Threshold profiles

| Profile | Gated | For |
|---|---|---|
| `screen` | 8 / 12 | First-pass triage before refolding exists |
| `default` | 12 / 12 | General de novo mini-binders |
| `nanobody` | 12 / 12 | VHH and antibody fragments |
| `strict` | 12 / 12 | Lead selection, the ones you will actually order |

Profiles are YAML and inherit with `extends:`, so a project-specific set only
states what it changes:

```yaml
extends: default
name: my-target
thresholds:
  ipsae: 0.60
  delta_sasa: 1400
```

Point at it with `--profile path/to/thresholds.my-target.yaml`, or drop it in
`config/` and call it by name.

## Input formats

Structures: `.pdb`, `.cif`, `.mmcif`, `.bcif`.

Confidence: AlphaFold2/ColabFold JSON, AlphaFold3 and RoseTTAFold3 JSON,
Boltz `.npz`. Two details that are easy to get wrong and are handled here:

- **Token alignment.** AF3, RF3 and Boltz tokenise ligands and ions one token
  per atom, so a PAE matrix is routinely larger than the residue count and
  not in the same order. When the file carries `token_chain_ids` the mapping
  is exact; otherwise a 1:1 correspondence is assumed and verified, and a
  mismatch is an error rather than a silently shifted matrix.
- **Modified residues.** MSE, TPO, SEP and PTR are flagged `hetero` in mmCIF
  but are part of the polymer. Filtering them out breaks the chain and shifts
  every downstream index. They are kept; waters, ions and free ligands are
  not.

## Validation

The native ipSAE implementation is checked against the score files
Dunbrack's `ipsae.py` produced for its own example models, in both AF2
(three chains, PDB, 15 Å cutoff) and AF3 (mmCIF with ATP/Mg/TPO, 10 Å cutoff)
form. Every structural quantity — `n0chn`, `n0dom`, `n0res`, `d0res`, the
token selected per residue, and the set of residue pairs clearing the cutoff
— matches exactly, and the scores agree to 10⁻⁶, which is the precision the
reference files print to.

```bash
pytest                 # 143 tests
pytest -k upstream     # just the reference regression
pytest tests/test_regressions.py
```

The reference tests skip if the ipSAE submodule has not been cloned, and CI
fails rather than passes if that happens.

`tests/test_regressions.py` pins a set of defects that were caught by
comparing against upstream and reviewing against the reference sources —
each one returned a plausible number rather than failing:

- a residue key without the chain id, which collided across a multi-chain
  target (every Fab has H and L both starting at residue 1) and reported a
  refold RMSD of 8 Å where the true value was 1 Å;
- confidence-file pairing that ignored the sample index, so in a normal AF3
  output directory `model_1` was scored with `model_0`'s PAE matrix;
- token alignment that dropped insertion codes, mapping 100, 100A and 100B
  onto a single PAE row, and that picked a modified residue's first atom
  rather than its CA;
- hydrogen bonds double-counted for hydroxyl pairs, where both atoms are
  donor and acceptor;
- a one-sided PAE triangle left half the matrix at 0.0 Å, i.e. maximal
  confidence, instead of mirroring or erroring;
- a constant B-factor column of 1.00 rescaled to a perfect pLDDT of 100.

That list is in the README rather than a changelog because each of them is a
thing this kind of tool gets wrong quietly, and knowing they were present is
useful when judging how much to trust the rest.

## Limitations

Worth knowing before trusting a number:

- **Shape complementarity is not Rosetta's `sc`.** It is an independent
  implementation on the van der Waals surface and reads roughly 0.17–0.20
  lower. The thresholds here are calibrated to that scale against 20
  experimentally determined complexes. Do not compare the value to a
  published Rosetta `sc`.
- **No interface ΔG.** Rosetta's `InterfaceAnalyzerMover` ΔG has no
  open-source equivalent of comparable quality, so rather than ship a weak
  proxy it is left out. BindCraft remains the tool for that number.
- **Hydrogen bonds are heavy-atom geometry.** Predictor output carries no
  hydrogens, so donor–acceptor distance is the criterion and angular terms
  are not evaluated. Counts run high against hydrogen-aware methods, which
  reject many distance-acceptable pairs on geometry.
- **The calibration set is 18 crystal interfaces, positives only.** No decoy
  set, antibody-heavy, and evolved rather than designed. It establishes the
  scale of the native metrics, not the false-positive rate of a threshold.
  Recalibrate against your own outcomes if you have them.
- **Refold RMSD needs a refold.** Without one that metric reports as missing,
  and under the default `missing_policy: fail` a candidate cannot pass on
  incomplete evidence. The `screen` profile sets `skip` instead.

## Licence

MIT, see `LICENSE`. The submodules under `external/` keep their own licences:
ipSAE is MIT, BindCraft and BoltzGen and the RosettaCommons foundry each
carry their own terms — check them before any commercial use.
