# Paired Formulae development dependency

`formulae.json` is a temporary CI override. Its `commit` must be the full 40-character
commit SHA containing the paired Formulae implementation. Both Bambi's normal
regression matrix and the structured acceptance jobs use the same checked-out SHA.
The install uses `--no-deps` after installing Bambi's runtime dependencies.

The pin is `2fbeb11d89a67db2700557872896dea0b5c5d9ab`, published on the
paired Formulae feature branch and reviewed in draft PR
https://github.com/tjburch/formulae/pull/1. The checkout action rejects a missing
pin or branch name. It must not quietly test an old released package or a moving
branch. Formulae was committed and pushed before setting this pin. Remove the
override when released Formulae support and Bambi's released dependency minimum
are coordinated; no version is invented in `pyproject.toml`.

The structured workflow runs bounded posterior and regression checks on PRs.
Adding `<!-- structured-full-acceptance -->` to the PR description also runs the
100-replication calibration, prior-sensitivity refits, and large benchmarks.
The marker starts extended jobs when the PR opens or its description changes;
ordinary code pushes rerun bounded checks without restarting the long jobs.
This PR trigger works before the workflow is available for `workflow_dispatch`
on the default branch. The workflows keep logs and raw draws as artifacts on
success or failure. A green compatibility smoke job is not statistical acceptance.

For fork-local draft CI, Bambi's `main` matches the intended base. Its fork default
is the abandoned `structured-group-covariance` branch and must not be used or
changed. Formulae's fork `master` is behind the intended upstream base `90b41e3`;
a focused validation base branch at that commit avoids mixing unrelated history
into the Formulae draft diff. Neither default branch needs to change.

Artifact uploads include the hidden `.cache` directories explicitly. Names contain
`github.run_attempt` so reruns retain earlier failure evidence. Environment and
installation records are written before sampling; sampler results are saved
before cleanup, recovery, prediction, or acceptance evaluation.
