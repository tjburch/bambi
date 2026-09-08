"""Opt-in hosted sampler integration; short chains are not posterior validation."""

import os

import bambi as bmb
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BAMBI_COVARIANCE_SAMPLER_CHECK") != "1",
    reason="Hosted opt-in check; no local posterior sampling",
)


@pytest.fixture(params=[False, True])
def sparse_mode(request):
    previous = bmb.config["SPARSE_DOT"]
    bmb.config["SPARSE_DOT"] = request.param
    try:
        yield request.param
    finally:
        bmb.config["SPARSE_DOT"] = previous


def test_four_block_sampler_prediction(sparse_mode):
    sampler = os.environ.get("BAMBI_COVARIANCE_SAMPLER", "pymc")
    data = pd.MultiIndex.from_product(
        [["s1", "s2", "s3"], ["a", "b"], ["c", "d"], [0, 1, 3]],
        names=["subject", "condition", "context", "year"],
    ).to_frame(index=False)
    data["y"] = np.arange(len(data)) % 2
    groups = ["subject", "subject:condition", "subject:context", "subject:condition:context"]
    model = bmb.Model(
        "y ~ 1 + " + " + ".join(f"ar1(0 + year | {group})" for group in groups),
        data,
        family="bernoulli",
    )
    model.build()
    settings = {"chain_method": "vectorized"} if sampler in {"numpyro", "blackjax"} else {}
    inference = model.fit(
        inference_method=sampler,
        chains=1,
        cores=1,
        draws=40,
        tune=40,
        random_seed=20260908,
        progressbar=False,
        nuts=settings,
    )
    prediction = model.predict(inference, data=data.iloc[:4], inplace=False)
    assert np.isfinite(prediction.posterior["p"]).all()
    model.compute_log_likelihood(inference)
    assert np.isfinite(inference.log_likelihood.to_dataset().to_array()).all()
