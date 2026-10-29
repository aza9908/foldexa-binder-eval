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
is 0–1, so its 0.8 is 80 on the 0–100 scale used here. `Average_i_pAE` is
divided by 31 Å, so its 0.35 is `0.35 × 31 ≈ 10.9 Å`; the default here is
rounded down to 10.0 Å, which also matches the PAE cutoff ipSAE uses for
AF3-class output.

BindCraft's own `Average_dSASA ≥ 1` is a presence check, not a size filter —
it rejects a complex with no interface at all. The real buried-area threshold
had to come from somewhere else.

## The calibration

Two native implementations sit on a different scale from the Rosetta
measurements BindCraft's thresholds were set against. Copying the numbers
across would have been wrong in opposite directions: too strict for shape
complementarity, too lax for hydrophobicity.

So both were measured. Twenty binder:target interfaces were taken from
experimentally determined complexes distributed with BoltzGen — therapeutic
Fab complexes (adalimumab, ustekinumab, belimumab, guselkumab, secukinumab,
tralokinumab), barnase–barstar, and a spread of the harder targets in that
set. For each, the smaller chain was treated as the binder and the largest as
the target. These are real, experimentally validated interfaces, so a filter
that rejects most of them is mis-scaled by construction.

Distribution over those 20 interfaces:

| Metric | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| Shape complementarity | 0.42 | 0.47 | 0.53 | 0.55 | 0.57 |
| Buried surface area (Å²) | 422 | 895 | 1545 | 3594 | 3820 |
| Interface residues | 5 | 8.7 | 14 | 36 | 40 |
| Interface H-bonds | 0 | 3.8 | 9.5 | 25 | 29 |
| Buried unsatisfied H-bonds | 0 | 0 | 1 | 5.1 | 9 |
| Surface hydrophobicity | 0.16 | 0.18 | 0.24 | 0.34 | 0.42 |

Reading across:

- **Interface residues, H-bonds and unsatisfied polars** land on essentially
  the same scale as BindCraft's Rosetta-derived values. BindCraft's `≥ 7`,
  `≥ 3` and `≤ 4` sit near the 10th percentile of real interfaces in each
  case, which is where a filter of this kind belongs. Copied unchanged.
- **Shape complementarity** does not. Rosetta's `sc` on native interfaces
  clusters at 0.70–0.75; the native implementation here reads 0.42–0.57 on
  the same kind of interface, a systematic offset of roughly 0.17–0.20 (see
  `docs/metrics.md` for why — van der Waals rather than solvent-excluded
  surface). BindCraft's 0.60 would reject every one of the 20. The default
  is set to **0.45**, just under the 10th percentile; `strict` uses 0.52,
  the median.
- **Surface hydrophobicity** needed the definition fixed rather than the
  threshold moved. With Rosetta's `is_apolar` set, which includes glycine and
  proline, these 20 proteins ran at a median of 0.41 and BindCraft's 0.35
  would have rejected about three quarters of them. Excluding Gly and Pro
  brings the median to 0.24 and puts 0.35 at roughly the 90th percentile,
  which is the behaviour BindCraft's number was chosen to have. So the
  definition changed and the threshold stayed.
- **Buried surface area** has no inherited value. The default of 1000 Å² sits
  near the 25th percentile. Note the minimum of 422 Å²: small interfaces do
  occur among real complexes, so this threshold will reject some genuine
  binders. That is the intended trade for a gate meant to cut a large
  generated set down to what is worth expressing.

Reproduce it with any directory of multi-chain structures:

```python
from foldexa_binder_eval import Complex, compute_interface_geometry

cx = Complex.from_file(path, binder_chain="B", target_chains="A")
g = compute_interface_geometry(cx)
print(g.shape_complementarity, g.delta_sasa, g.surface_hydrophobicity)
```

## Caveats on the calibration

Worth stating plainly:

- **n = 20** is small, and the set is antibody-heavy because that is what was
  available offline. The percentiles should be read as order-of-magnitude
  guidance, not as tight bounds.
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
