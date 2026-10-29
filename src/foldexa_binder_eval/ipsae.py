"""PAE-derived interface scores: ipSAE, pDockQ, pDockQ2, LIS.

This is a native re-implementation of the scoring in Dunbrack's ``ipsae.py``
(https://github.com/DunbrackLab/IPSAE, preprint
https://doi.org/10.1101/2025.02.10.637595), reduced to the binder:target case
and vectorised so a whole design campaign can be scored in-process without
shelling out per model.

The upstream script remains the reference. ``external/ipsae`` pins it as a
submodule, and ``tests/test_ipsae.py`` checks this implementation against the
score files committed there, which that script produced for its own example
models. The test parses those committed files rather than invoking the script;
:mod:`foldexa_binder_eval.adapters.ipsae_cli` can run it directly when you
want to compare on your own models.

Why ipSAE rather than raw ipTM: ipTM averages predicted alignment error over
*every* inter-chain residue pair, so a small, well-predicted interface on a
large target is diluted by hundreds of irrelevant pairs. ipSAE restricts the
average to pairs below a PAE cutoff and rescales ``d0`` by how many residues
actually clear that cutoff, which is what makes it discriminate real binders
from confident-looking nonsense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "InterfaceScores",
    "compute_interface_scores",
    "ptm_func",
    "calc_d0",
]

# Contact cutoff used by pDockQ / pDockQ2, in angstroms (Bryant et al. 2022).
PDOCKQ_DISTANCE_CUTOFF = 8.0

# LIS cutoff, in angstroms (Kim et al. 2024).
LIS_PAE_CUTOFF = 12.0

# d0 for the pDockQ2 pTM-style term, in angstroms (Zhu et al. 2023).
PDOCKQ2_D0 = 10.0


def ptm_func(pae: np.ndarray | float, d0: np.ndarray | float) -> np.ndarray:
    r"""TM-score kernel: :math:`1 / (1 + (\mathrm{PAE}/d_0)^2)`."""
    return 1.0 / (1.0 + (np.asarray(pae, dtype=float) / d0) ** 2)


def calc_d0(length, pair_type: str = "protein") -> np.ndarray:
    r"""Yang & Skolnick :math:`d_0`, floored as upstream does.

    .. math::
        d_0(L) = 1.24\,(L' - 15)^{1/3} - 1.8, \quad L' = \max(L, 26)

    clipped from below at 1.0 Å for protein pairs and 2.0 Å when a nucleic
    acid is involved. Yang & Skolnick, *Proteins* **57**:702-710 (2004).
    """
    arr = np.maximum(np.asarray(length, dtype=float), 26.0)
    floor = 2.0 if pair_type == "nucleic_acid" else 1.0
    return np.maximum(floor, 1.24 * (arr - 15.0) ** (1.0 / 3.0) - 1.8)


@dataclass
class InterfaceScores:
    """Scores for one ordered chain-group pair, plus the symmetric maxima."""

    ipsae: float
    ipsae_binder_to_target: float
    ipsae_target_to_binder: float
    ipsae_d0chn: float
    ipsae_d0dom: float
    iptm_d0chn: float
    pdockq: float
    pdockq2: float
    lis: float
    pae_interaction: float
    pae_interaction_binder_to_target: float
    pae_interaction_target_to_binder: float
    n_contacts: int
    n_interface_residues_pae: int
    pae_cutoff: float
    dist_cutoff: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ipsae": self.ipsae,
            "ipsae_b2t": self.ipsae_binder_to_target,
            "ipsae_t2b": self.ipsae_target_to_binder,
            "ipsae_d0chn": self.ipsae_d0chn,
            "ipsae_d0dom": self.ipsae_d0dom,
            "iptm_d0chn": self.iptm_d0chn,
            "pdockq": self.pdockq,
            "pdockq2": self.pdockq2,
            "lis": self.lis,
            "pae_interaction": self.pae_interaction,
            "n_contacts": self.n_contacts,
        }


def _asym_ipsae(
    pae: np.ndarray,
    mask_from: np.ndarray,
    mask_to: np.ndarray,
    pae_cutoff: float,
    pair_type: str,
) -> tuple[float, float, float, int]:
    """ipSAE in one direction.

    Returns ``(ipsae_d0res, ipsae_d0chn, ipsae_d0dom, n0dom)``.

    Mirrors the upstream loop: per residue of the source group, average the
    pTM kernel over target-group partners whose PAE clears the cutoff, then
    take the maximum over source residues.
    """
    n = pae.shape[0]
    valid = np.outer(mask_from, mask_to) & (pae < pae_cutoff)

    # --- d0chn: d0 from the summed length of both groups ----------------
    n0chn = int(mask_from.sum() + mask_to.sum())
    d0chn = float(calc_d0(n0chn, pair_type))
    kernel_chn = ptm_func(pae, d0chn)

    # --- d0dom: d0 from residues on either side that clear the cutoff ---
    rows_hit = valid.any(axis=1) & mask_from
    cols_hit = valid.any(axis=0) & mask_to
    n0dom = int(rows_hit.sum() + cols_hit.sum())
    d0dom = float(calc_d0(n0dom, pair_type)) if n0dom else float(calc_d0(0, pair_type))
    kernel_dom = ptm_func(pae, d0dom)

    # --- d0res: per-row d0 from that row's number of valid partners -----
    n0res = valid.sum(axis=1)
    d0res = calc_d0(n0res, pair_type)

    counts = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ipsae_chn_byres = np.where(
            counts > 0, (kernel_chn * valid).sum(axis=1) / np.maximum(counts, 1), 0.0
        )
        ipsae_dom_byres = np.where(
            counts > 0, (kernel_dom * valid).sum(axis=1) / np.maximum(counts, 1), 0.0
        )
        kernel_res = ptm_func(pae, d0res[:, None])
        ipsae_res_byres = np.where(
            counts > 0, (kernel_res * valid).sum(axis=1) / np.maximum(counts, 1), 0.0
        )

    # Only source-group rows may win the maximum.
    blocked = ~mask_from
    for arr in (ipsae_chn_byres, ipsae_dom_byres, ipsae_res_byres):
        arr[blocked] = 0.0

    if n == 0 or not valid.any():
        return 0.0, 0.0, 0.0, n0dom
    return (
        float(ipsae_res_byres.max()),
        float(ipsae_chn_byres.max()),
        float(ipsae_dom_byres.max()),
        n0dom,
    )


def _asym_iptm_d0chn(
    pae: np.ndarray, mask_from: np.ndarray, mask_to: np.ndarray, pair_type: str
) -> float:
    """ipTM recomputed from PAE with no cutoff -- the upstream ``ipTM_d0chn``."""
    n0chn = int(mask_from.sum() + mask_to.sum())
    d0chn = float(calc_d0(n0chn, pair_type))
    kernel = ptm_func(pae, d0chn)
    if not mask_to.any() or not mask_from.any():
        return 0.0
    byres = kernel[:, mask_to].mean(axis=1)
    byres = np.where(mask_from, byres, 0.0)
    return float(byres.max())


def compute_interface_scores(
    pae: np.ndarray,
    plddt: np.ndarray,
    distances: np.ndarray,
    binder_mask: np.ndarray,
    target_mask: np.ndarray,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    pair_type: str = "protein",
) -> InterfaceScores:
    """Compute every PAE-derived interface score for one binder:target pair.

    Parameters
    ----------
    pae:
        ``(n_res, n_res)`` predicted aligned error in angstroms.
    plddt:
        ``(n_res,)`` pLDDT on a 0-100 scale, taken at the representative
        (CB, else CA) atom -- this is what pDockQ was fit on.
    distances:
        ``(n_res, n_res)`` CB-CB (CA fallback) distances in angstroms.
    binder_mask, target_mask:
        Boolean residue masks. They must not overlap.
    pae_cutoff:
        PAE below which an inter-chain pair counts toward ipSAE. Upstream
        defaults: 15 Å for AF2, 10 Å for AF3/Boltz. We default to 10.
    dist_cutoff:
        Distance cutoff for the interface-residue count reported alongside.
    """
    pae = np.asarray(pae, dtype=float)
    distances = np.asarray(distances, dtype=float)
    binder_mask = np.asarray(binder_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)

    if binder_mask.shape != target_mask.shape:
        raise ValueError("binder_mask and target_mask must have the same shape")
    if (binder_mask & target_mask).any():
        raise ValueError("binder_mask and target_mask overlap")
    if pae.shape[0] != binder_mask.shape[0]:
        raise ValueError(
            f"PAE matrix covers {pae.shape[0]} residues but masks cover "
            f"{binder_mask.shape[0]}"
        )

    b2t = _asym_ipsae(pae, binder_mask, target_mask, pae_cutoff, pair_type)
    t2b = _asym_ipsae(pae, target_mask, binder_mask, pae_cutoff, pair_type)

    ipsae = max(b2t[0], t2b[0])
    ipsae_d0chn = max(b2t[1], t2b[1])
    ipsae_d0dom = max(b2t[2], t2b[2])
    iptm_d0chn = max(
        _asym_iptm_d0chn(pae, binder_mask, target_mask, pair_type),
        _asym_iptm_d0chn(pae, target_mask, binder_mask, pair_type),
    )

    # --- interface PAE -------------------------------------------------
    cross_bt = pae[np.ix_(binder_mask, target_mask)]
    cross_tb = pae[np.ix_(target_mask, binder_mask)]
    pae_b2t = float(cross_bt.mean()) if cross_bt.size else float("nan")
    pae_t2b = float(cross_tb.mean()) if cross_tb.size else float("nan")
    pae_interaction = float(np.nanmean([pae_b2t, pae_t2b]))

    # --- contacts ------------------------------------------------------
    contact = (
        np.outer(binder_mask, target_mask) & (distances <= PDOCKQ_DISTANCE_CUTOFF)
    )
    n_contacts = int(contact.sum())
    involved = contact.any(axis=1) | contact.any(axis=0)

    plddt = np.asarray(plddt, dtype=float)
    if n_contacts > 0 and np.isfinite(plddt[involved]).any():
        mean_plddt = float(np.nanmean(plddt[involved]))
        # Upstream uses log10 here, not the natural log.
        x = mean_plddt * math.log10(n_contacts)
        pdockq = 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018

        # pDockQ2 averages the pTM kernel over contacting pairs, and the PAE
        # matrix is not symmetric, so the two chain orders give different
        # values. Upstream reports the larger of the two for a chain pair;
        # doing the same keeps this comparable with its output instead of
        # silently depending on which chain was named the binder.
        def _directional_pdockq2(block_mask: np.ndarray) -> float:
            if not block_mask.any():
                return 0.0
            mean_ptm = float(ptm_func(pae[block_mask], PDOCKQ2_D0).mean())
            x2 = mean_plddt * mean_ptm
            return 1.31 / (1.0 + math.exp(-0.075 * (x2 - 84.733))) + 0.005

        pdockq2 = max(
            _directional_pdockq2(contact),
            _directional_pdockq2(contact.T),
        )
    else:
        pdockq = 0.0
        pdockq2 = 0.0

    # --- LIS -----------------------------------------------------------
    # Upstream computes LIS separately in each direction and the symmetric
    # value is the mean of the two, not a pooled average over all pairs.
    def _directional_lis(block: np.ndarray) -> float:
        if block.size == 0:
            return 0.0
        below = block[block < LIS_PAE_CUTOFF]
        if below.size == 0:
            return 0.0
        return float(((LIS_PAE_CUTOFF - below) / LIS_PAE_CUTOFF).mean())

    lis = float(
        (_directional_lis(cross_bt) + _directional_lis(cross_tb)) / 2.0
    )

    # --- interface residue count from PAE+distance ----------------------
    interface_pae = (
        np.outer(binder_mask, target_mask)
        & (pae < pae_cutoff)
        & (distances < dist_cutoff)
    )
    n_interface_residues_pae = int(
        (interface_pae.any(axis=1) | interface_pae.any(axis=0)).sum()
    )

    return InterfaceScores(
        ipsae=ipsae,
        ipsae_binder_to_target=b2t[0],
        ipsae_target_to_binder=t2b[0],
        ipsae_d0chn=ipsae_d0chn,
        ipsae_d0dom=ipsae_d0dom,
        iptm_d0chn=iptm_d0chn,
        pdockq=pdockq,
        pdockq2=pdockq2,
        lis=lis,
        pae_interaction=pae_interaction,
        pae_interaction_binder_to_target=pae_b2t,
        pae_interaction_target_to_binder=pae_t2b,
        n_contacts=n_contacts,
        n_interface_residues_pae=n_interface_residues_pae,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
    )
