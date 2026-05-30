# Add `Model.compute_log_prior`

Status: ready-for-agent

## Parent

`.scratch/prd-compute-log-prior.md` (PRD for bambinos/bambi#972)

## What to build

A new public method `Model.compute_log_prior(self, idata, inplace=True)` in
`bambi/models.py`, placed directly after `compute_log_likelihood` so the two
sibling methods are adjacent. It is a thin delegation wrapper around PyMC's
`pm.compute_log_prior` that adds a `log_prior` group to the `InferenceData` —
one variable per free random variable in the model, named to match the
variables in `idata.posterior`. The resulting group is exactly what
`psense(idata, group="prior")` consumes, mirroring the existing
log-likelihood workflow.

End-to-end behavior:

- Guards with `self._check_built()` first, so calling before build/fit yields a
  clear "call `.build()` or `.fit()`" error rather than an opaque
  `AttributeError` from dereferencing `self.backend.model`.
- Deletes any pre-existing `log_prior` group before delegating (matching the
  `del`-if-exists pattern in `compute_log_likelihood`), so re-runs are safe —
  PyMC otherwise raises `['log_prior'] group(s) already exists`.
- Delegates to `pm.compute_log_prior(idata, model=self.backend.model,
  extend_inferencedata=True, progressbar=False)`. PyMC names each `log_prior`
  variable after its free RV; Bambi already bakes term aliases into the PyMC RV
  names, so `log_prior`, `posterior`, and `psense`-selected names line up with
  no special alias handling.
- After delegating, stamps the group with `modeling_interface="bambi"` and
  `modeling_interface_version=__version__` (matching `compute_log_likelihood`).
  PyMC's own attrs are left in place alongside Bambi's.
- `inplace=False` deepcopies `idata` first, adds the group to the copy, and
  returns it (original untouched); `inplace=True` mutates in place and returns
  `None`. Mirrors `compute_log_likelihood`.
- NumPy-style docstring mirroring `compute_log_likelihood`'s structure
  (summary, Parameters, Returns) minus the `data` parameter. Drops the
  sibling's "new feature... may not work in all cases" caveat (delegation to
  well-tested PyMC makes it misleading). Adds a one-line note that the method
  is intended for prior sensitivity analysis.

No new imports (`pm`, `deepcopy`, `__version__` already imported). No `data=`
or `var_names=` parameter. No changes to `compute_log_likelihood`, changelogs,
or quartodoc (the `Model` class is auto-documented).

## Acceptance criteria

- [ ] `Model.compute_log_prior(self, idata, inplace=True)` exists in
  `bambi/models.py`, directly after `compute_log_likelihood`.
- [ ] After calling on a fitted model, `"log_prior"` is present in `idata`, with
  variables equal to the model's free RVs (e.g. `{"Intercept", "x", "sigma"}`
  for `y ~ x`).
- [ ] The `log_prior` group carries `modeling_interface="bambi"` and
  `modeling_interface_version` attrs.
- [ ] `inplace=False` returns a copy with the group added; the original `idata`
  is unmodified. `inplace=True` mutates in place and returns `None`.
- [ ] Calling on an unbuilt model raises the `_check_built()` error, not an
  `AttributeError`.
- [ ] Calling twice on the same `idata` does not raise (idempotent).
- [ ] A dedicated `test_compute_log_prior` test (simple Gaussian model,
  `mock_pymc_sample` fixture) asserts: group added; variable naming matches free
  RVs; non-mutation on `inplace=False`; idempotency on a second call.
- [ ] CI gates pass: `pixi run -e dev pre-commit run --all`,
  `pixi run -e dev pylint bambi`, `pixi run -e dev pytest tests`.

## Blocked by

None - can start immediately.
