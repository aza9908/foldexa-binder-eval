# Where the thresholds come from

Every number in `src/foldexa_binder_eval/profiles/` traces to one of three
places: a published filter set, a source publication, or a calibration
measured here. This page says which, for each metric, and shows the
measurement for the two that could not simply be copied.

## Provenance

| Metric | Default | Origin |
|---|---|---|
| `binder_plddt` | ≥ 80 | BindCraft `Average_Binder_pLDDT ≥ 0.8` |
| `complex_iptm` | ≥ 0.50 | BindCraft `Average_i_pTM ≥ 0.5` |
| `ipsae` | ≥ 0.50 | ipSAE preprint |
| `pae_interaction` | ≤ 10.0 Å | BindCraft `Average_i_pAE ≤ 0.35`, denormalised |
| `pdockq` | ≥ 0.23 | Bryant et al. 2022 |
| `delta_sasa` | ≥ 1000 Å² | measured here (p25 of real interfaces) |
| `n_interface_residues` | ≥ 7 | BindCraft |
| `n_interface_hbonds` | ≥ 3 | BindCraft |
| `n_unsat_hbonds` | ≤ 4 | BindCraft |
| `shape_complementarity` | ≥ 0.45 | **recalibrated here** |
| `surface_hydrophobicity` | ≤ 0.35 | BindCraft value, **redefinition verified here** |
| `binder_rmsd` | ≤ 2.5 Å | BoltzGen `refolding_rmsd_threshold` |

### On the BindCraft numbers

BindCraft stores pLDDT and PAE on normalised scales. `Average_Binder_pLDDT`
is 0–1, so its 0.8 is 80 on the 0–100 scale used here.

`Average_i_pAE` is the one number whose conversion is **not** verifiable from
the pinned submodules. BindCraft's filter file gives the threshold as 0.35,
but the value it compares against comes from ColabDesign
(`prediction_model.aux["log"]["i_pae"]` in
`external/bindcraft/functions/colabdesign_utils.py`), and ColabDesign is not
a submodule here. ColabDesign normalises PAE by 31 Å, the top of AlphaFold2's
PAE bin range, which would put 0.35 at about 10.9 Å — but that factor cannot
be confirmed in this repository, so treat it as the reasoning behind the
default rather than established provenance.

The default is set to 10.0 Å regardless, because that is also the PAE cutoff
upstream ipSAE uses for AF3-class output, which makes the two consistent. If
you need a threshold you can defend precisely, set it from your own data.

BindCraft's own `Average_dSASA ≥ 1` is a presence check, not a size filter —
it rejects a complex with no interface at all. The real buried-area threshold
had to come from somewhere else.

## The calibration

Two native implementations sit on a different scale from the Rosetta
measurements BindCraft's thresholds were set against. Copying the numbers
across would have been wrong in opposite directions: too strict for shape
complementarity, too lax for hydrophobicity.

So both were measured. The measurement is `scripts/calibrate_thresholds.py`,
kept in the repository so these numbers can be checked rather than taken on
trust:

```bash
python scripts/calibrate_thresholds.py external/boltzgen/example
```

It finds 18 usable binder:target interfaces in the structures distributed
with BoltzGen — therapeutic Fab complexes (adalimumab, ustekinumab,
belimumab, guselkumab, secukinumab, tralokinumab), barnase–barstar, and a
spread of the harder targets in that set. For each, the smallest chain is
treated as the binder and the largest as the target. These are real,
experimentally determined interfaces, so a filter that rejects most of them
is mis-scaled by construction.

| Metric | n | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| Shape complementarity | 18 | 0.42 | 0.46 | 0.53 | 0.55 | 0.57 |
| Buried surface area (Å²) | 18 | 422 | 859 | 1545 | 3600 | 3820 |
| Interface residues | 18 | 5 | 8.1 | 15 | 36.9 | 40 |
| Interface H-bonds | 18 | 0 | 3.4 | 8.5 | 23.6 | 26 |
| Buried unsatisfied H-bonds | 18 | 0 | 0 | 1 | 5.3 | 9 |
| Surface hydrophobicity | 17 | 0.16 | 0.18 | 0.23 | 0.31 | 0.35 |

Surface hydrophobicity has n = 17 because one structure (`zf.cif`, a zinc
finger against DNA) has no residue meeting the surface-exposure criterion
and reports NaN.

What each default threshold does to this set:

| Metric | Default | Rejects |
|---|---|---|
| Shape complementarity | ≥ 0.45 | 2 / 18 (11%) |
| Buried surface area | ≥ 1000 Å² | 3 / 18 (17%) |
| Interface residues | ≥ 7 | 2 / 18 (11%) |
| Interface H-bonds | ≥ 3 | 2 / 18 (11%) |
| Buried unsatisfied H-bonds | ≤ 4 | 3 / 18 (17%) |
| Surface hydrophobicity | ≤ 0.35 | 0 / 17 (0%) |

Reading across:

- **Interface residues, H-bonds and unsatisfied polars** land on essentially
  the same scale as BindCraft's Rosetta-derived values. Its `≥ 7`, `≥ 3` and
  `≤ 4` each reject 11–17% of real interfaces here, which is where a filter
  of this kind belongs. Copied unchanged.
- **Shape complementarity** does not. Rosetta's `sc` on native interfaces
  clusters at 0.70–0.75; the native implementation here reads 0.42–0.57 on
  the same interfaces, a systematic offset of roughly 0.17–0.20 (see
  [metrics.md](metrics.md) for why — van der Waals rather than
  solvent-excluded surface). BindCraft's 0.60 would reject all 18. The
  default is **0.45**, just under the 10th percentile; `strict` uses 0.52,
  just under the median of 0.53, which rejects a third of them.
- **Surface hydrophobicity** needed the definition fixed rather than the
  threshold moved. With Rosetta's `is_apolar` set, which includes glycine and
  proline, these proteins run at a median of 0.40 and BindCraft's 0.35 would
  reject most of them. Excluding Gly and Pro brings the median to 0.23.
  Being straight about what that leaves: 0.35 now sits at the **maximum** of
  this set and rejects none of it, so on real interfaces it is a loose guard
  rather than a discriminating filter. It is kept because it is the published
  BindCraft value and because de novo designs are reliably greasier than
  evolved complexes, which is the population it has to catch. `strict`
  tightens it to 0.28, near the 70th–80th percentile, which rejects 24%.
- **Buried surface area** has no inherited value. The default of 1000 Å² sits
  just below the 25th percentile (1075 Å²). Note the minimum of 422 Å²: small
  interfaces do occur among real complexes, so this threshold does reject
  some genuine binders. That is the intended trade for a gate meant to cut a
  large generated set down to what is worth expressing. `strict` at 1600 Å²
  rejects 56% of real interfaces, which is appropriate for lead selection and
  would be indefensible for triage.

## Caveats on the calibration

Worth stating plainly:

- **n = 18** is small, and the set is antibody-heavy because that is what the
  pinned submodules happen to carry. The percentiles should be read as
  order-of-magnitude guidance, not as tight bounds.
- **Crystal interfaces are not designed interfaces.** De novo binders tend
  toward flatter, more hydrophobic interfaces than evolved ones. If anything
  that argues for tightening shape complementarity relative to these numbers,
  not loosening it.
- **These are positives only.** There is no decoy set here, so the numbers say
  where real interfaces sit, not what false-positive rate a threshold buys.
  A proper calibration would need both.

If you have an in-house set of designs with experimental outcomes, recalibrate
against it. The profile format exists for exactly that:

```yaml
extends: default
name: house
description: Calibrated on our own expressed designs, 2026 campaign.
thresholds:
  shape_complementarity: 0.49
  delta_sasa: 1250
```

## Profile differences

| Metric | screen | default | nanobody | strict |
|---|---|---|---|---|
| `binder_plddt` | 70 | 80 | 80 | 85 |
| `complex_iptm` | 0.40 | 0.50 | 0.50 | 0.70 |
| `ipsae` | 0.35 | 0.50 | 0.50 | 0.65 |
| `pae_interaction` | 15.0 | 10.0 | 10.0 | 7.0 |
| `pdockq` | 0.15 | 0.23 | 0.23 | 0.40 |
| `delta_sasa` | 600 | 1000 | 800 | 1600 |
| `n_interface_residues` | 5 | 7 | 6 | 10 |
| `n_interface_hbonds` | 1 | 3 | 3 | 5 |
| `n_unsat_hbonds` | — | 4 | 4 | 2 |
| `shape_complementarity` | — | 0.45 | 0.42 | 0.52 |
| `surface_hydrophobicity` | — | 0.35 | 0.32 | 0.28 |
| `binder_rmsd` | — | 2.5 | 3.0 | 1.5 |
| `rmsd_align_on` | target | target | **binder** | target |
| `missing_policy` | skip | fail | fail | fail |

`screen` leaves the expensive metrics ungated rather than unmeasured: they are
still computed and reported, so nothing is discarded on evidence that has not
been gathered yet. It also sets `missing_policy: skip`, because at that stage
the refold does not exist.

`nanobody` differs for three reasons, stated in the profile itself: CDR
paratopes bury less surface than designed interface faces, the framework
dominates whole-domain RMSD so the self-consistency question is asked
binder-on-binder, and VHH developability failures are overwhelmingly
aggregation-driven, so surface hydrophobicity is tightened rather than
relaxed.

The test suite enforces the orderings — `strict` is never looser than
`default`, `screen` never tighter — so a careless edit to one profile fails
CI rather than silently inverting a gate.
