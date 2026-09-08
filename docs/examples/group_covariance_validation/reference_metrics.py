"""Common continuous posterior summaries for reference comparisons."""


def diagnose_scalar_metrics(inference, dataset):
    """Fail closed when the sampler does not expose required HMC diagnostics."""
    import arviz_stats as azs
    import numpy as np
    import xarray as xr

    posterior = inference["posterior"].to_dataset()
    variables = {
        name: value
        for name, value in dataset.data_vars.items()
        if not (
            name.startswith(("probability.", "log_likelihood.", "predictive_"))
            and bool((value == value.isel(chain=0, draw=0)).all())
        )
    }
    variables.update({name: value for name, value in posterior.items() if name.endswith("_offset")})
    table = azs.summary(xr.Dataset(variables), kind="diagnostics", round_to="none")
    samples = inference["sample_stats"].to_dataset()
    energy = np.asarray(samples["energy"].transpose("chain", "draw"))
    bfmi = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(energy, axis=1, ddof=1)
    if "reached_max_treedepth" not in samples:
        raise ValueError("Missing reached_max_treedepth; diagnostics cannot be certified")
    return {
        "chains": posterior.sizes["chain"],
        "rhat_max": float(table["r_hat"].max(skipna=False)),
        "ess_bulk_min": float(table["ess_bulk"].min(skipna=False)),
        "ess_tail_min": float(table["ess_tail"].min(skipna=False)),
        "divergences": int(samples["diverging"].sum()),
        "bfmi_min": float(np.min(bfmi)),
        "treedepth_hits": int(samples["reached_max_treedepth"].sum()),
    }


def summarize_metrics(dataset):
    """Keep chains separate when estimating mean and quantile Monte Carlo error."""
    import arviz_stats as azs

    probabilities = (0.03, 0.5, 0.97)
    means = dataset.mean(dim=("chain", "draw"))
    mean_errors = azs.mcse(dataset, method="mean")
    quantiles = dataset.quantile(probabilities, dim=("chain", "draw"))
    quantile_errors = {
        probability: azs.mcse(dataset, method="quantile", prob=probability)
        for probability in probabilities
    }
    constant = {
        name: bool((value == value.isel(chain=0, draw=0)).all())
        for name, value in dataset.data_vars.items()
    }
    return {
        name: {
            "mean": float(means[name]),
            "mcse_mean": 0.0 if constant[name] else float(mean_errors[name]),
            "quantiles": {
                str(probability): {
                    "value": float(quantiles[name].sel(quantile=probability)),
                    "mcse": 0.0 if constant[name] else float(quantile_errors[probability][name]),
                }
                for probability in probabilities
            },
        }
        for name in dataset.data_vars
    }


def add_binomial_metrics(metrics, probability, trials, log_likelihood):
    """Compare continuous conditional moments, not discrete predictive quantiles."""
    observation_dims = [dim for dim in probability.dims if dim not in ("chain", "draw")]
    likelihood_dims = [dim for dim in log_likelihood.dims if dim not in ("chain", "draw")]
    if len(observation_dims) != 1 or len(likelihood_dims) != 1:
        raise ValueError("Expected one observation dimension")
    if probability.sizes[observation_dims[0]] != len(trials):
        raise ValueError("Trial counts and probabilities differ in length")
    if log_likelihood.sizes[likelihood_dims[0]] != len(trials):
        raise ValueError("Log likelihood and probabilities differ in length")
    for index, count in enumerate(trials):
        p = probability.isel({observation_dims[0]: index}, drop=True)
        mean = count * p
        metrics[f"probability.{index + 1}"] = p
        metrics[f"log_likelihood.{index + 1}"] = log_likelihood.isel(
            {likelihood_dims[0]: index}, drop=True
        )
        metrics[f"predictive_mean.{index + 1}"] = mean
        metrics[f"predictive_second_moment.{index + 1}"] = count * p * (1 - p) + mean**2
        metrics[f"predictive_zero_probability.{index + 1}"] = (1 - p) ** count
    return metrics
