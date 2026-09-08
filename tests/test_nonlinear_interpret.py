import pandas as pd
import pytest

import bambi as bmb

from bambi import interpret


@pytest.mark.parametrize(
    "function, kwargs",
    [
        (interpret.predictions, {}),
        (interpret.comparisons, {"contrast": "x"}),
        (interpret.slopes, {"wrt": "x"}),
        (interpret.plot_predictions, {"conditional": "x"}),
        (interpret.plot_comparisons, {"contrast": "x", "conditional": "x"}),
        (interpret.plot_slopes, {"wrt": "x", "conditional": "x"}),
    ],
)
def test_interpret_rejects_nonlinear_models(function, kwargs):
    model = bmb.Model(
        bmb.Formula("y ~ a * x", "a ~ 1", nonlinear=True),
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}),
        priors={"a": {"Intercept": bmb.Prior("Normal", mu=0, sigma=1)}},
    )

    with pytest.raises(NotImplementedError, match="interpret API does not support nonlinear"):
        function(model, None, **kwargs)
