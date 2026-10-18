# ArviZ wide summary can mislabel multidimensional parameters

## Discovery

An independent review of the diagonal covariance calibration artifacts found that
the saved `summary.csv` row labelled `g000, x` had a mean near the next group's
intercept. Direct inspection of the labelled NetCDF array confirmed the mismatch:

| Quantity, replication 1 | Mean |
| --- | ---: |
| Raw `effect[g000, x]` | -0.0187553631 |
| CSV row labelled `effect[g000, x]` | -0.7454549566 |
| Raw `effect[g001, Intercept]` | -0.7454549566 |

The raw array has dimensions `(chain, draw, group, coefficient)`. Its values and
coordinate labels are correct. The wide summary placed coefficient-major values
under group-major labels. This is a reporting defect, separate from the PyMC LKJ
issues described in `pymc-lkj-bug-report.md`.

## Environment and reproduction

Confirmed locally with `arviz_stats==1.3.3`, `arviz_base==1.3.1`,
`xarray==2026.9.0`, and `pandas==2.3.3`. The archived CI summaries show the same
mislabelled rows. This reproducer requires neither Bambi nor MCMC:

```python
import arviz_stats as azs
import numpy as np
import xarray as xr

rng = np.random.default_rng(sum(map(ord, "structured-summary-order-reproducer")))
values = np.array([[0, 1], [10, 11], [20, 21]])
values = values + rng.normal(0, 0.01, (4, 100, 3, 2))
posterior = xr.Dataset(
    {
        "sd": (("chain", "draw", "coefficient"), rng.normal(size=(4, 100, 2))),
        "effect": (("chain", "draw", "group", "coefficient"), values),
    },
    coords={"group": ["g0", "g1", "g2"], "coefficient": ["Intercept", "x"]},
)
wide = azs.summary(posterior, ci_prob=0.94, ci_kind="hdi")
print(float(posterior.effect.sel(group="g0", coefficient="x").mean()))
print(wide.loc["effect[g0, x]", "mean"])
```

The first result is near 1; the second is near 10. The issue depends on differing
Dataset-wide and per-variable dimension orders. The `sd` variable introduces the
coefficient dimension before the effect's group dimension.

## Cause and affected evidence

`arviz_stats.summary` computes labelled summary arrays, then uses
`arviz_base.reorg.dataset_to_dataframe` for wide output. Its
`dataset_to_dataarray` helper obtains values with `Dataset.to_stacked_array` and
builds labels separately with `xarray_sel_iter`. In this case, their traversal
orders differ. The `fmt="xarray"` summary retains the correct coordinate/value
association.

The affected coefficient CSV rows must not be used to identify a group's mean or
interval. The calibration evaluator reads raw labelled posterior arrays, not these
CSVs. All 100 diagonal fits were reassessed from their saved posteriors: ranks,
coverage counts, and pass/fail decisions are unchanged. Diagnostic extrema are
also invariant to this row permutation. The original files remain preserved.

## Focused workaround

The validation script requests `fmt="xarray"` with unrounded values. For each
variable, it transposes the summary to that variable's original coordinate order,
then builds the CSV through a labelled pandas Series. Scalars and vectors are
handled separately from multidimensional variables. This changes reporting only;
it does not change fitting, recovery, prediction, or statistical gates.

The regression uses a Dataset whose coefficient dimension precedes its group
dimension, then checks every CSV mean and HDI against direct coordinate-selected
draws. Corrected summaries are saved beside archived originals as
`summary-coordinate-corrected.csv`. The reproducer results and resource logs are
retained under `.cache/structured-ci/`.

No upstream package source was modified and no upstream issue was published.
