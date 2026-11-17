# Geometry and the Emergence of Molecular Order in Early Life

**When can the spatial arrangement of chiral adsorption sites affect enantiomeric excess?**

Leon Sandler, Independent Researcher, Northbrook, IL, USA (sandler.leon@gmail.com)

Companion code, results, figures and manuscript for a Perspective on mineral-surface homochirality. Version 2.0.0 is the revised
manuscript (revision 2, prepared for *Discover Life*). Archived on Zenodo: [10.5281/zenodo.21120967](https://doi.org/10.5281/zenodo.21120967)
(concept DOI, resolves to the latest version).

## What the revised version does

The Perspective separates four ingredients that are often merged in discussions of surface-induced homochirality: a source of
global chiral bias, the local enantiomeric discrimination energy, the spatial organization of selective sites together with
molecular transport across them, and nonlinear chemical amplification. The script produces four illustrative calculations:

| Figure | Content |
| --- | --- |
| Fig. 1 | Independent-site equilibrium baseline: the population-weighted average of signed local preferences, `ee_ads = <tanh(ΔE_i / 2k_BT)>`. Energetic disorder lowers it smoothly and only at second order in its variance, so a gradual response to disorder follows from the averaging itself. |
| Fig. 2 | Two lattices with identical site statistics (4096 selective sites of 16384) and different arrangements, before and after finite-range sampling. |
| Fig. 3 | Contrast survival: the fraction of sites whose sampled preference stays above a threshold, versus transport length, for several domain correlation lengths; an identical continuous energy multiset; and a signed (racemic) surface where local survival coexists with zero global mean. |
| Fig. 4 | Recovered output: uniform recovery leaves the global mean unchanged (null case); recovery co-located with selective sites or a nonlinear local step makes the arrangement matter, and the direction of the effect follows the curvature of that step. |

All calculations are phenomenological illustrations with stated assumptions and order-of-magnitude parameters. They are not
first-principles or molecular-dynamics predictions for any specific mineral, and the response functions in Fig. 4 are
hypothetical.

## Reproduce

```
pip install -r requirements.txt
python code/chiral_geometry_r2.py figures
```

Runs in about 30 s and writes `Fig1`–`Fig4` (PNG and TIFF) and `results_r2.json` into the given folder. The run is seeded and
deterministic; `results/results_r2.json` holds the values quoted in the manuscript.

## Layout

| Path | Content |
| --- | --- |
| `code/chiral_geometry_r2.py` | The script (Online Resource 1 of the manuscript) |
| `results/results_r2.json` | Numerical results quoted in the text |
| `figures/` | Figs. 1–4 |
| `manuscript/` | Revised manuscript (Word and PDF) |
| `v1/` | The first version (July 2026): robustness script, summary and figures of the earlier multiplicative-suppression model |

## Changes from version 1

Version 2 replaces the earlier model. The multiplicative suppression `ee_eff = ee_eq (1 − β f_race)` and the description of weakly
selective sites as "racemization centers" were removed: weakly selective sites dilute the enantiomeric excess because they adsorb
both enantiomers almost equally, not because they interconvert them. The "peak ee" reported in version 1 was the value at the upper
end of the swept discrimination-energy range (30 meV), and the lower edge of its "Goldilocks window" was where the multiplicative
expression was clipped at zero; both were artefacts and are not used in version 2. The terminology "topology-controlled robustness"
was replaced by geometry-dependent language, the channel-width law of version 1 was withdrawn, and new sections address spatial
arrangement, the origin and amplification of handedness, and the distinction between steric adsorption and chirality-induced spin
selectivity. The version 1 files are kept unchanged in `v1/` for reference.

## License

Code: MIT (see `LICENSE`). Manuscript, figures and results: CC BY 4.0.
