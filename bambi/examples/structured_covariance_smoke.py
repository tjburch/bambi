"""Small, sequential execution checks for structured covariance backends.

Run one invocation per process, for example:
    uv run --no-sync bambi/examples/structured_covariance_smoke.py ar1 marginal nutpie
These short runs verify execution and shapes, not convergence or calibration.
"""

# pylint: disable=invalid-name

import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import bambi as bmb

wrapper, method, sampler = sys.argv[1:4]
seed = sum(map(ord, "structured-covariance-backend-smoke"))
data = pd.DataFrame(
    {
        "y": [0.3, 1.1, -0.2, 0.5, 0.9, -0.4],
        "x": [-1.0, 0.3, 0.8, -0.5, 0.1, 1.0],
        "time": [2, 0, 2, 1, 3, 0],
        "g": ["a", "a", "a", "b", "b", "b"],
        "h": ["z", "w", "z", "w", "z", "w"],
    }
)
variant = sys.argv[4] if len(sys.argv) > 4 else ""
inner = "time" if wrapper in ("ar1", "ou") else "1+x"
if variant == "three":
    inner = "1+x+I(x**2)"
options = ", homogeneous=True" if variant == "homogeneous" else ""
term = f"{wrapper}({inner}, by=g{options})"
priors = (
    {term: {"rho": bmb.Prior("Normal", mu=-0.6, sigma=0.05)}} if variant == "negative" else None
)
mean = f"y ~ x + (1|h) + {term}"
if variant == "hsgp":
    mean = f"y ~ hsgp(x, m=3, c=1.5) + (1|h) + {term}"
model = bmb.Model(mean, data, priors=priors)
start = time.perf_counter()
model.build()
original_coords = model.backend.model.coords.copy()
prior = model.prior_predictive(draws=2, random_seed=seed)
built = time.perf_counter()
result = model.fit(
    draws=4,
    tune=4,
    chains=1,
    cores=1,
    random_seed=seed,
    inference_method=sampler,
    structured_method=method,
    progressbar=False,
    compute_convergence_checks=False,
)
fitted = time.perf_counter()
output = Path(".cache/structured-covariance")
output.mkdir(parents=True, exist_ok=True)
name = "-".join([wrapper, method, sampler] + ([variant] if variant else []))
result.to_netcdf(output / f"{name}-raw.nc")
effect = next(iter(model.parameters["mu"].structured_terms.values()))
assert effect.label in result.posterior
assert np.isfinite(result.posterior[effect.label]).all()
assert model.backend.model.coords == original_coords
if variant == "hsgp":
    weights = result.posterior["hsgp(x, m=3, c=1.5)_weights"]
    assert "hsgp(x, m=3, c=1.5)_weights_dim" in weights.dims
if variant == "negative":
    assert (result.posterior[effect.label + "_rho"] < 0).all()
model.predict(result, random_seed=seed)
model.compute_log_likelihood(result)
target = data.iloc[[0, 0, 3]].copy()
target.loc[:, "g"] = "new"
target.loc[:, "time"] = [1, 1, 4]
model.predict(result, data=target, sample_new_groups=True, random_seed=seed)
assert np.isfinite(result.posterior["mu"]).all()
# Build the prior graph after fitting, then verify the selected backend is restored.
backend = model.backend
model.prior_predictive(draws=2, random_seed=seed)
assert model.backend is backend
record = {
    "wrapper": wrapper,
    "homogeneous": bool(options),
    "variant": variant,
    "method": method,
    "sampler": sampler,
    "seed": seed,
    "build_and_prior_seconds": built - start,
    "fit_and_recovery_seconds": fitted - built,
    "prediction_seconds": time.perf_counter() - fitted,
    "divergences": int(result.sample_stats["diverging"].sum()),
    "note": "Four draws in one chain establish execution only.",
    "versions": {
        name: version(name)
        for name in (
            "bambi",
            "formulae",
            "pymc",
            "pytensor",
            "nutpie",
            "jax",
            "numpyro",
            "blackjax",
        )
    },
    "platform": platform.platform(),
}
(output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
result.to_netcdf(output / f"{name}.nc")
print(json.dumps(record, indent=2))
