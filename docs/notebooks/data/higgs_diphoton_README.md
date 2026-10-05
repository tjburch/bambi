# CMS discovery diphoton spectrum

These files support the third example in `../nonlinear_expressions.ipynb`.
They contain real observed counts, not simulated events or digitized plot points.

## Sources and attribution

- **CMS Collaboration, CMS Higgs boson observation statistical model, v1.0**
  (2024), <https://doi.org/10.17181/c2948-e8875>.
  Archive: <https://repository.cern/records/c2948-e8875/files/cms-h-observation-public-v1.0.tar.gz>.
  The archive's `LICENSE` explicitly licenses the material under
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
  The counts CSV is an adaptation: category histograms were rebinned across
  the full 100–180 GeV range. No events were added, rescaled, or assigned signal/background labels.
- **CMS Collaboration, Observation of a new boson at a mass of 125 GeV with the
  CMS experiment at the LHC**, Physics Letters B 716 (2012), 30–61,
  <https://doi.org/10.1016/j.physletb.2012.08.021>, Table 2.
  [Official table](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-12-028/CMS-HIG-12-028_Table_002.pdf).
  The categories CSV transcribes numerical facts from this table, omitting
  production-mode percentages and background uncertainties. It does not contain
  the table image or reproduce its layout.

Retrieved 2026-10-05. Archive SHA-256:
`9878c11273779fb60129d0198c0eba7b6addb2c5e40625086de58209f95b2ccd`.
`125.5/comb_hgg.input.root` SHA-256:
`c58ec471024ea57d9f6354df884d0da96be5ad6ad12a75b83c94c3dd3cd94f59`.

## Columns and category mapping

`higgs_diphoton_counts.csv` contains `category`, `mass` (bin center, GeV), and
`counts` (integer events in `[mass - 0.5, mass + 0.5)`). It contains 880 rows,
including zero-count bins, and 50,776 events. Rows are category-major, then
in increasing mass order. The notebook fits all 880 rows over 100–180 GeV and zooms the fitted plot
to 110–150 GeV. All 880 counts were also checked against independent
rebinning of the matching unit-weight, unbinned RooDataSets.

`higgs_diphoton_categories.csv` contains the category name, `signal_events`
(expected Standard Model signal at 125 GeV), `sigma_eff` (effective resolution
in GeV), and `background_per_gev` (estimated background density at 125 GeV).
Here `sigma_eff` is half the narrowest interval containing 68% of the signal;
using its category ratios as Gaussian standard-deviation ratios is an approximation.
The category order is the same in both files:

| CSV category | Published category | Workspace | Original binned dataset |
| --- | --- | --- | --- |
| `7TeV_cat0`–`7TeV_cat3` | 7 TeV BDT 0–3 | `wbkg` | `databinned_cat0_7TeV`–`databinned_cat3_7TeV` |
| `7TeV_cat4` | 7 TeV dijet tag | `wbkg` | `databinned_cat4_7TeV` |
| `8TeV_cat0`–`8TeV_cat3` | 8 TeV BDT 0–3 | `wbkg1` | `databinned_mvacat0_8TeV`–`databinned_mvacat3_8TeV` |
| `8TeV_cat4` | 8 TeV dijet tight | `wbkg1` | `databinned_mvacat4_8TeV` |
| `8TeV_cat5` | 8 TeV dijet loose | `wbkg1` | `databinned_mvacat5_8TeV` |

## Reproducing the count extraction

After downloading and unpacking the source archive, the following reproduces
the counts CSV. ROOT is needed **only for this one-time extraction**, not to
execute the notebook. The original RooDataHist `weight()` is the integer number
of observed events in a 0.25 GeV bin, not a category's S/(S+B) display weight.
The matching unbinned RooDataSets have unit event weights.

```python
import ROOT
import numpy as np
import pandas as pd

source = ROOT.TFile.Open("cms-h-observation-public-v1.0/125.5/comb_hgg.input.root")
rows = []
edges = np.arange(100, 181, 1)
for energy, ncat, workspace, prefix in [
    (7, 5, "wbkg", "cat"), (8, 6, "wbkg1", "mvacat")
]:
    for category in range(ncat):
        original = source.Get(workspace).data(
            f"databinned_{prefix}{category}_{energy}TeV"
        )
        mass, counts = [], []
        for i in range(original.numEntries()):
            mass.append(original.get(i).getRealValue("CMS_hgg_mass"))
            counts.append(original.weight())
        mass, counts = np.asarray(mass), np.asarray(counts)
        assert np.all(counts == counts.astype(int))
        rebinned, _ = np.histogram(mass, bins=edges, weights=counts)
        assert rebinned.sum() == counts[(mass >= 100) & (mass < 180)].sum()
        for center, count in zip((edges[:-1] + edges[1:]) / 2, rebinned):
            rows.append((f"{energy}TeV_cat{category}", center, int(count)))
pd.DataFrame(rows, columns=["category", "mass", "counts"]).to_csv(
    "higgs_diphoton_counts.csv", index=False
)
```

## Statistical use

The notebook fits independent Poisson counts by category, preserving the
information responsible for the categories' different sensitivities. It uses a
Gaussian signal, fixed relative signal yields/resolutions from Table 2, and
separate backgrounds with quadratic log-rates. It applies S/(S+B) weights only to the display,
computed from this simplified fit using the prescription described around
Figure 3 of the paper (a 2-sigma_eff-wide window centered at 125 GeV, with
normalization preserving total signal yield). These are not the exact numerical
weights from the original CMS fit. Weighted statistical error bars use the sum
of squared weights times counts.

The archive is the subsequently released discovery statistical model, not a
claim of a byte-for-byte reconstruction of the original Figure 3 inputs. The
notebook does not reproduce CMS's full signal shapes, background functions,
systematic uncertainties, or discovery significance.
The released CMS signal model uses calibrated Gaussian mixtures, including
right- and wrong-vertex components. The paper's background fits use category-dependent
degree 3–5 polynomials over 100–180 GeV with signal-bias studies; the notebook's
quadratic log-rates over 100–180 GeV have not undergone that validation. Agreement in
peak location does not establish agreement in extracted signal strength.

The notebook also compares predictive performance with PSIS-LOO and computes illustrative
Bayes factors by bridge sampling. These comparisons use the original category counts, not
the display-weighted spectrum. Background intercept priors are broad and common across
categories; the published background densities are used only for sampler initialization.
Signal-prior sensitivity is shown explicitly. The evidence comparisons are retrospective and
conditional on the stated simplified models, not a reconstruction of discovery significance.
