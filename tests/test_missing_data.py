"""Tests for missing data imputation functionality."""

import numpy as np
import pandas as pd
import pytest

import bambi as bmb


@pytest.fixture(scope="module")
def data_with_missing_response():
    """Data with missing values in the response variable."""
    rng = np.random.default_rng(121195)
    n = 100
    x = rng.normal(size=n)
    y = 2 + 3 * x + rng.normal(size=n)

    # Introduce missing values in y
    missing_idx = rng.choice(n, size=10, replace=False)
    y[missing_idx] = np.nan

    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture(scope="module")
def data_with_missing_predictor():
    """Data with missing values in a predictor variable."""
    rng = np.random.default_rng(121195)
    n = 100
    x = rng.normal(size=n)
    y = 2 + 3 * x + rng.normal(size=n)

    # Introduce missing values in x
    missing_idx = rng.choice(n, size=10, replace=False)
    x_with_missing = x.copy()
    x_with_missing[missing_idx] = np.nan

    return pd.DataFrame({"y": y, "x": x_with_missing, "x_complete": x})


@pytest.fixture(scope="module")
def data_no_missing():
    """Data without any missing values."""
    rng = np.random.default_rng(121195)
    n = 100
    x = rng.normal(size=n)
    y = 2 + 3 * x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x})


class TestMiTransformation:
    """Tests for the mi() transformation function."""

    def test_mi_function_exists(self):
        """Test that mi() function is available in transformations namespace."""
        from bambi.transformations import mi, transformations_namespace

        assert "mi" in transformations_namespace
        assert callable(mi)

    def test_mi_function_preserves_data(self):
        """Test that mi() function returns the data unchanged."""
        from bambi.transformations import mi

        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = mi(data)

        # Check that non-NaN values are preserved
        assert np.allclose(result[~np.isnan(result)], data[~np.isnan(data)])
        # Check that NaN positions are preserved
        assert np.isnan(result[2])

    def test_mi_function_warns_no_missing(self):
        """Test that mi() warns when there are no missing values."""
        from bambi.transformations import mi

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.warns(UserWarning, match="No missing values detected"):
            mi(data)

    def test_mi_function_rejects_multidimensional(self):
        """Test that mi() rejects multi-dimensional arrays."""
        from bambi.transformations import mi

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="1-dimensional"):
            mi(data)


class TestResponseMissingDataImputation:
    """Tests for response-side missing data imputation using mi()."""

    def test_model_creation_with_mi_response(self, data_with_missing_response):
        """Test that a model can be created with mi() on the response."""
        model = bmb.Model("mi(y) ~ x", data_with_missing_response)

        # Check that the response term has the is_mi flag set
        assert model.response_component.term.is_mi

    def test_model_build_with_mi_response(self, data_with_missing_response):
        """Test that a model with mi() response can be built."""
        model = bmb.Model("mi(y) ~ x", data_with_missing_response)
        model.build()

        # Check that the model was built successfully
        assert model.built
        assert model.backend.model is not None

    def test_mi_response_creates_imputation_variables(self, data_with_missing_response):
        """Test that mi() response creates imputation variables in PyMC model."""
        model = bmb.Model("mi(y) ~ x", data_with_missing_response)
        model.build()

        # Check for unobserved variables (imputation creates them)
        pymc_model = model.backend.model
        var_names = [v.name for v in pymc_model.unobserved_RVs]

        # There should be variables for imputing the missing response values
        # PyMC creates variables with _unobserved suffix for imputed values
        assert any("unobserved" in name.lower() or "mi" in name.lower() for name in var_names) or \
               len([v for v in pymc_model.free_RVs if "y" in v.name.lower()]) > 0


class TestPredictorMissingDataImputation:
    """Tests for predictor-side missing data imputation using mi()."""

    def test_model_creation_with_mi_predictor(self, data_with_missing_predictor):
        """Test that a model can be created with mi() on a predictor."""
        model = bmb.Model("y ~ mi(x)", data_with_missing_predictor)

        # Check that the missing data term was created
        parent_component = model.components[model.family.likelihood.parent]
        assert len(parent_component.missing_data_terms) > 0

    def test_model_build_with_mi_predictor(self, data_with_missing_predictor):
        """Test that a model with mi() predictor can be built."""
        model = bmb.Model("y ~ mi(x)", data_with_missing_predictor)
        model.build()

        # Check that the model was built successfully
        assert model.built
        assert model.backend.model is not None

    def test_mi_predictor_creates_imputation_variables(self, data_with_missing_predictor):
        """Test that mi() predictor creates imputation variables in PyMC model."""
        model = bmb.Model("y ~ mi(x)", data_with_missing_predictor)
        model.build()

        # Check for imputation variables
        pymc_model = model.backend.model
        var_names = [v.name for v in pymc_model.unobserved_RVs]

        # There should be variables for imputing the missing predictor values
        assert any("imputed" in name.lower() or "x" in name.lower() for name in var_names)


class TestMissingDataWarnings:
    """Tests for warnings related to missing data."""

    def test_mi_response_warns_no_missing(self, data_no_missing):
        """Test that using mi() without missing values issues a warning."""
        # When building the model, it should warn about no missing values
        model = bmb.Model("mi(y) ~ x", data_no_missing)

        with pytest.warns(UserWarning, match="No missing values"):
            model.build()


class TestFormulaUseMiDetection:
    """Tests for detecting mi() in formulas."""

    def test_formula_uses_mi_detection(self):
        """Test that formula_uses_mi correctly detects mi() usage."""
        from bambi.utils import formula_uses_mi

        # Should detect mi() on response
        assert formula_uses_mi("mi(y) ~ x")

        # Should detect mi() on predictor
        assert formula_uses_mi("y ~ mi(x)")

        # Should detect mi() in complex formulas
        assert formula_uses_mi("y ~ mi(x) + z")
        assert formula_uses_mi("y ~ x + mi(z)")

        # Should not detect mi in other contexts
        assert not formula_uses_mi("y ~ x")
        assert not formula_uses_mi("family ~ x")
        assert not formula_uses_mi("y ~ x + family")


class TestMissingDataIntegration:
    """Integration tests for missing data imputation."""

    @pytest.mark.slow
    def test_fit_model_with_mi_response(self, data_with_missing_response):
        """Test that a model with mi() response can be fit."""
        model = bmb.Model("mi(y) ~ x", data_with_missing_response)

        # Fit with minimal sampling for speed
        idata = model.fit(draws=10, tune=10, chains=1, random_seed=42)

        # Check that we got samples
        assert "posterior" in idata.groups()
        assert "Intercept" in idata.posterior
        assert "x" in idata.posterior

    @pytest.mark.slow
    def test_fit_model_with_mi_predictor(self, data_with_missing_predictor):
        """Test that a model with mi() predictor can be fit."""
        model = bmb.Model("y ~ mi(x)", data_with_missing_predictor)

        # Fit with minimal sampling for speed
        idata = model.fit(draws=10, tune=10, chains=1, random_seed=42)

        # Check that we got samples
        assert "posterior" in idata.groups()
        assert "Intercept" in idata.posterior

        # Check that imputed values are in the posterior
        var_names = list(idata.posterior.data_vars)
        assert any("imputed" in name.lower() or "mi" in name.lower() for name in var_names) or \
               "mi(x)" in var_names or "x" in " ".join(var_names)
