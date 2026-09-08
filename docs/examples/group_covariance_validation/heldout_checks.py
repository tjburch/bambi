"""Refit one blocked four-block AR1 split on a remote validation runner."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ[variable] = "1"


def split_mask(data, split):
    """Choose a block using predictors only; never use outcomes to select a split."""
    if split == "subject":
        mask = data["subject"].astype(str) == sorted(data["subject"].astype(str).unique())[-1]
    elif split == "future":
        mask = data["year"] == data["year"].max()
    elif split == "three-way":
        columns = ["subject", "condition", "context"]
        mask = None
        for values in sorted(set(map(tuple, data[columns].to_numpy()))):
            candidate = (data[columns] == values).all(axis=1)
            training = data.loc[~candidate]
            subject, condition, context = values
            if ((training["subject"] == subject) & (training["condition"] == condition)).any() and (
                (training["subject"] == subject) & (training["context"] == context)
            ).any():
                mask = candidate
                break
        if mask is None:
            raise ValueError("No three-way split preserves both lower-order groups")
    else:
        raise ValueError("Unknown held-out split")
    if not mask.any() or mask.all():
        raise ValueError("Split requires nonempty training and held-out rows")
    if data.loc[~mask, "year"].nunique() < 2:
        raise ValueError("Training data need at least two time coordinates")
    return mask.to_numpy()


def predictive_scores(probability, prediction, observed, trials, seed):
    """Score the posterior mixture, preserving joint draws across held-out rows."""
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import binom

    probability, prediction = np.asarray(probability), np.asarray(prediction)
    observed, trials = np.asarray(observed), np.asarray(trials)
    if probability.ndim != 3 or prediction.shape != probability.shape:
        raise ValueError("Expected matching chain, draw, observation arrays")
    if probability.shape[-1] != len(observed) or observed.shape != trials.shape:
        raise ValueError("Held-out observation dimensions differ")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid predictive probability")
    if (
        not np.isfinite(observed).all()
        or not np.isfinite(trials).all()
        or np.any(trials < 0)
        or np.any(trials != np.floor(trials))
        or np.any(observed != np.floor(observed))
        or np.any(observed < 0)
        or np.any(observed > trials)
    ):
        raise ValueError("Invalid binomial observations")
    if (
        not np.isfinite(prediction).all()
        or np.any(prediction != np.floor(prediction))
        or np.any(prediction < 0)
        or np.any(prediction > trials)
    ):
        raise ValueError("Invalid predictive binomial counts")
    samples = probability.reshape(-1, len(observed))
    log_mass = binom.logpmf(observed, trials, samples)
    log_score = logsumexp(log_mass, axis=0) - np.log(len(samples))
    below = binom.cdf(observed - 1, trials, samples).mean(axis=0)
    mass = np.exp(log_mass).mean(axis=0)
    pit = below + np.random.default_rng(seed).uniform(size=len(observed)) * mass
    lower, upper = np.quantile(
        prediction.reshape(-1, len(observed)), [0.03, 0.97], axis=0, method="inverted_cdf"
    )
    return {
        "probability_mean": samples.mean(axis=0),
        "log_score": log_score,
        "randomized_pit": pit,
        "lower_94": lower,
        "upper_94": upper,
        "covered_94": (observed >= lower) & (observed <= upper),
    }, float(logsumexp(log_mass.sum(axis=1)) - np.log(len(samples)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=["subject", "future", "three-way"], required=True)
    parser.add_argument("--family", choices=["bernoulli", "binomial"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed < 1 or args.output.exists():
        parser.error("Use a positive seed and a fresh output directory")

    import arviz_stats as azs
    import bambi as bmb
    import numpy as np
    import pandas as pd
    from posterior_checks import predictive_statistics
    from reference_metrics import diagnose_scalar_metrics
    from validation_identity import create_identity

    data = pd.read_csv(args.data, float_precision="round_trip")
    if args.family == "bernoulli" and not (data["trials"] == 1).all():
        raise ValueError("Bernoulli fixture must have one trial per observation")
    mask = split_mask(data, args.split)
    train, heldout = data.loc[~mask].reset_index(drop=True), data.loc[mask].reset_index(drop=True)
    groups = ["subject", "subject:condition", "subject:context", "subject:condition:context"]
    wrappers = [f"ar1(0 + year | {group})" for group in groups]
    priors = {
        name: bmb.Prior("Normal", mu=0, sigma=1.5, auto_scale=False)
        for name in ("Intercept", "x1", "x2")
    }
    priors.update(
        {
            wrapper: {
                "sd": bmb.Prior("HalfNormal", sigma=2.5, auto_scale=False),
                "rho": bmb.Prior("Normal", mu=0, sigma=0.5, auto_scale=False),
            }
            for wrapper in wrappers
        }
    )
    response = "y" if args.family == "bernoulli" else "proportion(y, trials)"
    model = bmb.Model(
        response + " ~ 1 + x1 + x2 + " + " + ".join(wrappers),
        train,
        family=args.family,
        priors=priors,
        center_predictors=False,
        noncentered=True,
    )
    args.output.mkdir(parents=True)
    train.to_csv(args.output / "training.csv", index=False)
    heldout.to_csv(args.output / "heldout.csv", index=False)
    metadata = {
        "identity": create_identity(args.data, "ar1"),
        "split": args.split,
        "family": args.family,
        "seed": args.seed,
        "chains": 4,
        "warmup": 1000,
        "draws": 1000,
        "heldout_rows": np.flatnonzero(mask).tolist(),
        "training_md5": hashlib.md5((args.output / "training.csv").read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("bambi", "pymc", "nutpie", "arviz-stats")
        },
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    model.build()
    prior = model.prior_predictive(draws=200, random_seed=args.seed, omit_offsets=False)
    prior.to_netcdf(args.output / "prior.nc")
    inference = model.fit(
        draws=1000,
        tune=1000,
        chains=4,
        cores=1,
        inference_method="nutpie",
        target_accept=0.95,
        random_seed=args.seed + 1,
        omit_offsets=False,
        include_response_params=True,
    )
    inference.to_netcdf(args.output / "inference.nc")
    posterior = inference["posterior"].to_dataset()
    names = ["Intercept", "x1", "x2"] + [
        f"{wrapper}_{parameter}" for wrapper in wrappers for parameter in ("sd", "rho")
    ]
    diagnostics = diagnose_scalar_metrics(inference, posterior[names])
    (args.output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False)
    )
    if (
        diagnostics["rhat_max"] > 1.01
        or diagnostics["ess_bulk_min"] < 400
        or diagnostics["ess_tail_min"] < 400
        or diagnostics["divergences"] != 0
        or diagnostics["bfmi_min"] < 0.3
        or diagnostics["treedepth_hits"] != 0
    ):
        raise ValueError("Held-out fit diagnostics fail; saved draws retained")
    azs.summary(posterior[names], ci_prob=0.94, ci_kind="hdi", round_to="none").to_csv(
        args.output / "posterior-summary.csv"
    )
    # New groups are detected automatically; sample_new_groups is a deprecated no-op.
    model.predict(inference, data=heldout, kind="response", random_seed=args.seed + 2)
    inference.to_netcdf(args.output / "inference-heldout.nc")
    predictions = inference["predictions"].to_dataset()
    probability = predictions[model.family.likelihood.parent].transpose("chain", "draw", ...)
    response_names = [name for name in predictions if name != model.family.likelihood.parent]
    if len(response_names) != 1:
        raise ValueError("Expected one held-out response variable")
    prediction = predictions[response_names[0]].transpose("chain", "draw", ...)
    scores, joint = predictive_scores(
        probability.values, prediction.values, heldout["y"], heldout["trials"], args.seed + 3
    )
    rows = heldout.assign(**scores)
    rows.to_csv(args.output / "scores.csv", index=False)
    for column in ("subject", "year"):
        rows.groupby(column)[
            ["probability_mean", "log_score", "covered_94", "randomized_pit"]
        ].mean().to_csv(args.output / f"scores-by-{column}.csv")
    statistics, observed = predictive_statistics(prediction, heldout)
    table = azs.summary(statistics, kind="stats", ci_prob=0.94, ci_kind="hdi", round_to="none")
    table["observed"] = pd.Series(observed)
    table.to_csv(args.output / "heldout-predictive.csv")
    (args.output / "report.md").write_text(
        f"# Held-out {args.split} validation\n\nSource: {metadata['identity']['source_commit']}.\n\n"
        f"Training rows: {len(train)}. Held-out rows: {len(heldout)}. Seed: {args.seed}.\n\n"
        f"Joint held-out log predictive density estimate: {joint:.6g}. "
        "This finite-draw joint importance estimate can be unstable in high dimensions.\n\n"
        "scores.csv reports posterior mean probabilities, marginal mixture log scores, randomized "
        "discrete PIT and empirical 94% predictive-count coverage. Count intervals are discrete and "
        "need not attain exactly 94% coverage. Shared latent effects make rows dependent; PIT values "
        "are not an independent uniform sample. No row-wise LOO or iid uniformity test is used.\n\n"
        "All held-out rows are predicted together to preserve latent covariance. Subject and year "
        "tables summarize dependent observations, not independent replications. Review predictive "
        "discrepancies and repeat predefined splits before any general calibration claim. "
        "Independent joint-reference comparison remains a separate validation gate.\n"
    )


if __name__ == "__main__":
    main()
