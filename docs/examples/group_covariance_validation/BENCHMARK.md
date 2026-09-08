# Resource benchmark

`benchmark.py` measures AR1 model construction and graph building. It does not
sample, estimate model accuracy, or run a benchmark matrix. Compilation is off
unless `--compile` is supplied. Defaults are four subjects, four times, one
replicate, and interaction depth one. Two conditions and two contexts are used
only when the selected depth needs them.

Run from the repository root in the existing project environment. The output
directory must exist; the script refuses to overwrite any output file.

```sh
uv run --no-sync python docs/examples/group_covariance_validation/benchmark.py --output /Volumes/T9/Projects/bambi-correlation-structures/.scratch/covariance-checks/benchmark-small.json
```

Use `--groups`, `--times`, `--replicates`, `--conditions`, `--contexts`, and
`--interaction-depth` to vary one dimension at a time. All counts must be positive;
times must be at least two. This is a small construction fixture with alternating
binary outcomes, not a data-generating model for statistical recovery.

| Depth | Additive covariance blocks | Rows |
|---|---|---|
| 1 | subject | groups × times × replicates |
| 2 | subject; subject:condition | groups × conditions × times × replicates |
| 3 | subject; subject:condition; subject:context; subject:condition:context | groups × conditions × contexts × times × replicates |

All blocks have separate learned SD and AR1 correlation parameters. The fixture
uses a complete crossed design and shared integer time support. Replication adds
observations, not coefficient cells. `coefficient_count` counts latent group-time
coefficients across all blocks; it excludes fixed effects and covariance
hyperparameters. `--sparse` selects Bambi's sparse group-specific design path; it
does not promise sparse covariance factorization.

## Laptop limits

- Run one command at a time. Do not run alongside tests, reference fits, or other
  numerical work. Start with defaults; inspect memory before increasing sizes.
- The script fixes OMP, OpenBLAS, MKL, Accelerate, Numba, and BLIS thread environment
  limits to one before importing numerical libraries. It starts no worker pool.
  These settings do not provide an operating-system memory limit or control every
  external compiler process.
- The script must run as a separate process, not imported into a notebook that
  already initialized numerical thread pools.
- Use the existing environment. `uv run --no-sync` avoids dependency syncing;
  ensure the environment already exists before running it. Do not create or
  rebuild an environment just for this benchmark.
- Keep compilation caches on the external drive using the established task
  `PYTENSOR_FLAGS` settings. A build can perform internal graph preparation even
  without `--compile`; disabling explicit compilation does not remove all cost.
- Reserve larger interaction/time grids and `--compile` for suitable hardware.
  Compilation can consume substantial memory. No automatic repetitions or
  concurrency are enabled.

## Output and interpretation

JSON records wall time separately for imports, data generation, model
construction, graph build, and optional log-probability compilation. Compilation
time is `null` when skipped. The compiled function is not evaluated. Existing
compiler caches affect timing; record whether a comparison uses warm or cold
caches. Do not delete shared caches to manufacture a cold run.

Peak RSS is the current Python process's lifetime high-water mark, including
imports, not incremental model allocation. macOS reports raw bytes; Linux reports
raw KiB. Both are normalized to bytes, with the raw value and unit retained.
Intermediate snapshots are cumulative high-water marks, not separate phase
allocations. Child compiler memory and total system memory are not included.
Other platforms are rejected because their RSS units are not defined here.

The report also includes dimensions, formula, per-block coefficient counts,
package versions, Bambi source path, platform, thread settings, and PyTensor flags.
Use distinct output paths for each sparse/dense or size comparison. Verify that
the recorded source path points to this clone. Save failure logs separately;
failed runs must not be recorded as successful timing results.

**Status:** benchmark source supplied. No benchmark, numerical-library import,
compilation, or sampling was run while adding this script. Performance gates
remain pending until measurements are collected on the intended hardware.
