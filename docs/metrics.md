# The twelve metrics

What each metric measures, how it is computed here, and why it earns a slot.
Thresholds are in [thresholds.md](thresholds.md).

A note on the shape of the set. Twelve is not an arbitrary number of knobs:
the metrics fall into three tiers that fail for different reasons, and a
candidate that clears one tier can still be nonsense by the standards of
another. A design can be predicted with total confidence and bury 300 Å². It
can bury 2000 Å² and leave eight polar atoms unsatisfied. It can be perfect
on paper and refold into something else. Each tier catches a failure mode the
others are blind to, which is why none of them can be dropped.

---

## Tier 1 — prediction confidence

Does the structure predictor actually believe this complex?

### 1. Binder pLDDT

**Key** `binder_plddt` · **Direction** higher is better · **Units** 0–100

Mean pLDDT over binder residues in the complex. Taken from the predictor's
confidence file when present, otherwise from the structure's B-factor column,
which is where AF2, AF3, RF3 and Boltz all write it. Values on a 0–1 scale
are rescaled to 0–100 automatically.

A binder the model cannot place confidently in its own right is not a
candidate, whatever the interface scores say. This is the cheapest possible
filter and it removes a surprising fraction of generated designs.

### 2. Complex ipTM

**Key** `complex_iptm` · **Direction** higher · **Units** 0–1

Interface pTM as reported by the predictor. If the confidence file does not
carry one, it is recomputed from the PAE matrix using the chain-length `d0`
(the `ipTM_d0chn` quantity in ipSAE's output) and the report records that it
was derived rather than read.

ipTM averages over *every* inter-chain residue pair, so a small, well-modelled
interface on a large target is diluted by hundreds of pairs that were never
going to be confident. That makes it a reasonable first gate and a poor final
one, which is exactly why ipSAE sits next to it.

### 3. ipSAE

**Key** `ipsae` · **Direction** higher · **Units** 0–1

The interface-restricted version of the same idea, from
[Dunbrack 2025](https://doi.org/10.1101/2025.02.10.637595). For an ordered
chain pair, consider only residue pairs whose PAE is below a cutoff
(10 Å by default, 15 Å for AF2). For each binder residue *i*:

```
n0res(i)  = number of target residues j with PAE(i,j) < cutoff
d0res(i)  = max(1.0, 1.24 · (max(n0res(i), 26) − 15)^(1/3) − 1.8)
ipSAE(i)  = mean over those j of  1 / (1 + (PAE(i,j) / d0res(i))²)
```

The score is the maximum over *i*, and the symmetric value is the larger of
the two directions. Rescaling `d0` by how many residues actually clear the
cutoff is the part that matters: it stops a genuine small interface from
being punished for the size of the target it sits on.

The floor at `d0 = 1.0` has a consequence worth internalising. When only a
few residue pairs clear the cutoff, the kernel is tiny no matter how low the
error is — a two-residue contact scores near zero even at 1 Å PAE. That is
the metric working as intended, and it is why ipSAE discriminates where ipTM
does not.

Implemented natively in `foldexa_binder_eval.ipsae` and validated against the
upstream script; see the validation section of the README.

### 4. Interface PAE

**Key** `pae_interaction` · **Direction** lower · **Units** Å

Mean predicted aligned error over the binder:target block of the PAE matrix,
averaged across both directions. No cutoff, no rescaling — the rawest reading
available of how confident the model is about where the two chains sit
relative to each other.

It is kept alongside ipSAE because the two fail differently: ipSAE can stay
respectable when a handful of residues are predicted very well, while the
mean stays poor. Requiring both closes that gap.

### 5. pDockQ

**Key** `pdockq` · **Direction** higher · **Units** 0–1

[Bryant et al. 2022](https://www.nature.com/articles/s41467-022-28865-w).
A logistic fit combining interface size and confidence:

```
x      = mean_plddt · log₁₀(n_contacts)
pDockQ = 0.724 / (1 + exp(−0.052 · (x − 152.611))) + 0.018
```

where contacts are CB–CB pairs (CA where there is no CB) within 8 Å and
`mean_plddt` is taken over the residues involved, at the representative atom
— not a per-residue average, since that is what the fit was built on.

0.23 is the published boundary between acceptable and incorrect models. It is
the one metric here that was fitted against experimental outcomes rather than
reasoned from structure, which makes it a useful independent check on the
others.

pDockQ2 and LIS are computed too and appear in the JSON report, but are not
gated: they correlate strongly enough with pDockQ that gating all three would
be one decision wearing three hats.

---

## Tier 2 — interface quality

The predictor can be confident about an interface that is not worth having.

### 6. Buried surface area

**Key** `delta_sasa` · **Direction** higher · **Units** Å²

```
ΔSASA = SASA(binder alone) + SASA(target alone) − SASA(complex)
```

Shrake–Rupley, 1.4 Å probe, via biotite. The total buried on both sides,
matching Rosetta's `dSASA` convention. Only the binder and target chains
participate; a bystander chain in the same model is excluded, since it would
otherwise absorb buried surface belonging to neither side.

Affinity scales with interface size over most of the useful range. A design
burying 400 Å² is a contact, not a binder.

### 7. Interface residues

**Key** `n_interface_residues` · **Direction** higher · **Units** count

Binder residues with at least one heavy atom within 4.0 Å of a target heavy
atom — the same cutoff BindCraft uses. The reported value counts binder-side
residues only; the total across both sides is in the JSON extras along with
the residue labels, which is what you want when inspecting a specific design.

Guards against the grazing contact: a single loop touching the target can
produce respectable buried area while providing nothing to optimise.

### 8. Interface H-bonds

**Key** `n_interface_hbonds` · **Direction** higher · **Units** count

Cross-interface donor–acceptor pairs at 2.4–3.5 Å. Donors and acceptors are
assigned per residue and atom name (backbone amide N donates, backbone
carbonyl O accepts, side chains by table).

Predictor output carries no hydrogens, so this is heavy-atom geometry with no
angular term. Counts therefore run somewhat high against hydrogen-aware
methods; the threshold is set with that in mind. Polar contacts are where
specificity lives, which is why a hydrophobic patch with zero hydrogen bonds
should not pass.

### 9. Buried unsatisfied H-bonds

**Key** `n_unsat_hbonds` · **Direction** lower · **Units** count

Polar heavy atoms that are solvent-accessible in the free chain, essentially
inaccessible in the complex, and have no hydrogen-bond partner anywhere in
the complex. This is the `ddG`-style definition, matching BindCraft's
`BuriedUnsatHbonds` filter configuration.

Each one is a desolvation penalty paid for nothing. In practice this is the
single most informative metric in the set for designs that score well
everywhere else and then fail to bind: burying a charged side chain against a
hydrophobic face looks fine to a confidence metric and costs several kcal/mol
in reality.

### 10. Shape complementarity

**Key** `shape_complementarity` · **Direction** higher · **Units** 0–1

[Lawrence & Colman 1993](https://doi.org/10.1006/jmbi.1993.1648). Dot
surfaces with outward normals are built for each side; for each interface dot
on one surface the nearest dot on the other is found, and

```
S = (n_A · −n_B) · exp(−w · d²),   w = 0.5
```

The score is the mean of the two medians. A dot counts as an interface dot
when a water probe could no longer reach it because of the partner — the same
burial criterion that defines ΔSASA, so the two interface definitions stay
consistent.

Two implementation choices, both of which affect the scale:

- Surfaces are built at the **van der Waals radius**, not as
  solvent-accessible surfaces. Two solvent-accessible surfaces in van der
  Waals contact are separated by twice the probe radius, about 3.4 Å, which
  drives `exp(−w d²)` to zero and collapses the score to near nothing. The
  original method uses the solvent-excluded surface, where the two sheets are
  in contact; the van der Waals envelope is the cheap approximation to it.
- Dot normals are **averaged over a 2.5 Å neighbourhood**. A raw van der
  Waals surface is bumpy at the atomic scale, and per-dot radial normals swing
  by tens of degrees between neighbouring atoms, depressing the `n_A · n_B`
  term for interfaces that are in fact well mated.

The result correlates with Rosetta's `sc` but sits on a different scale. It is
not interchangeable with a published `sc` value, and its threshold is
calibrated separately — see [thresholds.md](thresholds.md).

---

## Tier 3 — developability

A real binder that cannot be made is not a result.

### 11. Surface hydrophobicity

**Key** `surface_hydrophobicity` · **Direction** lower · **Units** fraction

Fraction of solvent-exposed binder residues whose side chain is apolar or
aromatic. A residue counts as exposed when its relative SASA in the free
binder — absolute SASA over the Gly-X-Gly maximum from
[Tien et al. 2013](https://doi.org/10.1371/journal.pone.0080635) — is at least
0.25. The hydrophobic set is `A C I L M F V W Y`.

Glycine and proline are deliberately excluded, which is a departure from
BindCraft: it uses Rosetta's `is_apolar` plus F/W/Y, and that sweeps both in.
Neither contributes a hydrophobic surface patch, and since this metric is a
proxy for aggregation and non-specific binding, including them adds noise.
The threshold is calibrated against this definition.

This is where otherwise good designs die. A binder with a greasy surface
aggregates in expression, sticks to columns, and gives non-specific signal in
every assay that follows.

### 12. Refold RMSD

**Key** `binder_rmsd` · **Direction** lower · **Units** Å

Backbone RMSD (N, CA, C, O) between the designed binder and an independent
refold of its sequence. Two modes:

- `align_on: target` superposes on the target backbone and then measures the
  binder — *does the binder land in the same pose?* BindCraft's `Binder_RMSD`.
- `align_on: binder` superposes the binder on itself — *does the sequence
  fold back to its own design?* BoltzGen's `filter_rmsd` self-consistency
  check. The `nanobody` profile uses this one, because for a fixed framework
  the pose question is dominated by framework agreement and says little about
  the CDRs.

The self-consistency check is the closest thing in this set to an independent
opinion: everything else is computed from one structure, and a design that
scores perfectly on all eleven other metrics but refolds into a different
shape has been evaluated on a structure its own sequence does not produce.

Without a refold this metric reports as missing. Under the default
`missing_policy: fail` a candidate then cannot pass — deliberately, since
passing would mean declaring success on evidence never gathered.
