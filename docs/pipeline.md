# How the pieces fit together

This package is the gate at the end of a binder design campaign. It does not
generate, fold or optimise anything. Everything upstream of the dashed line
below runs on its own; this repository reads what those tools wrote.

```
                 target structure + binding site
                              |
                              v
                     ┌─────────────────┐
                     │    BoltzGen     │   generate binder candidates
                     └────────┬────────┘
                              │  designs/*.cif
                              v
                     ┌─────────────────┐
                     │  RF3 / Boltz /  │   refold each sequence
                     │   AF2 / AF3     │   independently
                     └────────┬────────┘
                              │  refolds/*.cif + confidence
- - - - - - - - - - - - - - - │ - - - - - - - - - - - - - - - - - - -
                              v
                  ┌───────────────────────┐
                  │  foldexa-binder-eval  │
                  ├───────────────────────┤
                  │ confidence  PAE/pLDDT │  → pLDDT, ipTM
                  │ ipsae       native    │  → ipSAE, pDockQ, iPAE
                  │ interface   geometry  │  → ΔSASA, contacts, H-bonds,
                  │                       │    unsat, SC, hydrophobicity
                  │ structure   RMSD      │  → refold self-consistency
                  ├───────────────────────┤
                  │  thresholds → verdict │
                  └───────────┬───────────┘
                              v
                    csv · json · html · exit code
```

## Where each upstream tool enters

**BoltzGen** (`external/boltzgen`) generates candidates and, in the
inverse-folding path, the refolds used for the self-consistency check. Its
own `Analyze` step writes `aggregate_metrics_*.csv`;
`adapters/boltzgen.py` reads that file and carries its columns through into
the report as extras, so you can compare its gate against this one on the
same designs. The columns it emits — `design_ptm`,
`design_to_target_iptm`, `min_design_to_target_pae`, `filter_rmsd`,
`delta_sasa_*`, `plip_hbonds`, `plip_saltbridge` — overlap with the twelve
but are not the same measurements, so they are reported alongside rather than
mapped onto each other.

**RoseTTAFold3** (`external/rf3-foundry`, `models/rf3`) predicts complexes
and emits AF3-style confidence output: a `summary_confidences` block with
`ptm`, `iptm`, `has_clash` and `ranking_score`, and a `confidences` block with
the `pae` and `plddt` arrays. That means the AF3 parser handles it unchanged.
`adapters/rf3.py` pairs each predicted structure with its confidence files and
sorts by `ranking_score`.

Upstream flags the RF3 inference API, including confidence output naming, as
still stabilising. If a release renames files, `discover_rf3_outputs` is the
single place that needs adjusting.

**ipSAE** (`external/ipsae`) is the reference implementation of the interface
scoring. The scoring is reimplemented natively in `foldexa_binder_eval.ipsae`
so a campaign can be scored in-process instead of one subprocess per model,
and `adapters/ipsae_cli.py` runs and parses the upstream script so the two can
be compared. `tests/test_ipsae.py` asserts they agree against the score files
committed upstream.

**BindCraft** (`external/bindcraft`) is not called at runtime. It is pinned
because it is the provenance of most of the default thresholds, and having
`settings_filters/default_filters.json` next to the profiles makes it
possible to check a claimed threshold rather than take it on trust.

## A campaign, end to end

```bash
# 1. generate (BoltzGen, its own environment)
#    → workdir/intermediate_designs_inverse_folded/

# 2. refold each sequence independently
#    → workdir/refolds/

# 3. triage the full set before spending anything else on it
foldexa-binder-eval batch workdir/intermediate_designs_inverse_folded \
    --pattern '*.cif' --binder-chain B \
    --profile screen --out workdir/triage

# 4. full gate on what survived, now that refolds exist
foldexa-binder-eval batch workdir/intermediate_designs_inverse_folded \
    --pattern '*.cif' --binder-chain B \
    --refold-dir workdir/refolds \
    --profile default --out workdir/gated

# 5. lead selection
foldexa-binder-eval batch workdir/intermediate_designs_inverse_folded \
    --pattern '*.cif' --binder-chain B \
    --refold-dir workdir/refolds \
    --profile strict --out workdir/leads
```

Running `screen` first is not just about speed. Shape complementarity is the
slowest metric by a wide margin, and at the triage stage the refold does not
exist yet, so three of the twelve cannot be computed at all. `screen` gates on
what is available and reports the rest rather than failing designs for
evidence nobody has gathered.

## Chain conventions

The binder is one chain; the target is one or more. Defaults are the first
chain in the file as binder and everything else as target, which is rarely
what you want — name them:

```bash
--binder-chain B --target-chains A,C
```

Two things the loader does that are worth knowing, because getting either
wrong corrupts every metric downstream rather than producing an obvious
error:

**All chains stay loaded.** A PAE matrix from AF2 or Boltz carries no token
identifiers, so its only alignment to the structure is residue order over the
whole model. Dropping a bystander chain at load time would silently shift
that mapping. Chains are instead excluded at the point where it matters —
`Complex.pair_only()` restricts the geometry so a third chain cannot absorb
buried surface belonging to neither side of the interface.

**Modified residues are polymer.** MSE, TPO, SEP and PTR are flagged `hetero`
in mmCIF but are part of the chain. Filtering on that flag breaks the chain
and shifts every index against the PAE matrix. The loader keeps anything
biotite recognises as an amino acid or nucleotide and drops waters, ions and
free ligands. Pass `keep="all"` when the ligand is part of the interface
being scored.

## Output

`batch` writes three files from one run:

- **`.csv`** — one row per candidate, every metric value plus a `__pass`
  column for each, for sorting and joining against your own tables.
- **`.json`** — the same plus per-metric thresholds, status, provenance
  string, and the extras: pDockQ2, LIS, ipSAE variants, interface residue
  labels, per-side buried area, and which backend computed what. This is the
  one to keep.
- **`.html`** — a standalone report, dark-mode aware, no external assets,
  openable from a file:// URL. For sending to someone who will not run the
  tool.

`score` additionally exits non-zero on a failing candidate, so it composes
with `make`, CI, or a shell loop without extra glue.
