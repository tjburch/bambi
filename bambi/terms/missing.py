"""Missing data term for predictor-side imputation."""

import numpy as np
import formulae.terms

from bambi.terms.base import BaseTerm
from bambi.terms.utils import is_single_component, is_call_component, is_call_of_kind


def is_mi_term(term):
    """Determines if a formulae term represents a predictor with missing data imputation.

    Bambi uses this function to detect mi() wrapped terms and treat them specially
    by creating imputation distributions for their missing values.
    """
    if not is_single_component(term):
        return False
    component = term.components[0]
    if not is_call_component(component):
        return False
    return is_call_of_kind(component, "mi")


class MissingDataTerm(BaseTerm):
    """A term for predictors with missing values that need imputation.

    This term handles predictor variables wrapped with mi() in the formula.
    It creates a mechanism for Bayesian imputation of missing values in predictors.

    Parameters
    ----------
    term : formulae.terms.Term
        The underlying formulae term.
    prior : Prior, int, float, or None
        The prior for the coefficient of this term.
    imputation_prior : Prior, optional
        The prior for the imputation model. If None, a default Normal prior is used.
    """

    def __init__(self, term, prior, imputation_prior=None):
        self.term = term
        self.prior = prior
        self.imputation_prior = imputation_prior

        # Get the raw data and identify missing values
        raw_data = np.squeeze(np.asarray(term.data, dtype=float))
        self._missing_mask = np.isnan(raw_data)
        self._data = raw_data
        self._n_missing = np.sum(self._missing_mask)

        if self._n_missing == 0:
            import warnings
            warnings.warn(
                f"No missing values detected in term '{self.name}' wrapped with mi(). "
                "Consider removing the mi() wrapper.",
                UserWarning
            )

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, (formulae.terms.terms.Term, formulae.terms.terms.Intercept))
        self._term = value

    @property
    def name(self):
        return self.term.name

    @property
    def coords(self):
        """Obtain PyMC coordinates for this term."""
        coords = {}
        if self.categorical:
            name = self.name + "_dim"
            coords[name] = self.levels
        elif self._data.ndim > 1 and self._data.shape[1] > 1:
            name = self.name + "_dim"
            coords[name] = np.arange(self._data.shape[1])
        return coords

    @property
    def data(self):
        """Return the data with NaN values preserved."""
        return self._data

    @property
    def missing_mask(self):
        """Boolean array indicating which values are missing."""
        return self._missing_mask

    @property
    def observed_mask(self):
        """Boolean array indicating which values are observed."""
        return ~self._missing_mask

    @property
    def n_missing(self):
        """Number of missing values."""
        return self._n_missing

    @property
    def n_observed(self):
        """Number of observed (non-missing) values."""
        return len(self._data) - self._n_missing

    @property
    def kind(self):
        return "mi"

    @property
    def shape(self):
        return self._data.shape

    @property
    def categorical(self):
        return self.term.kind == "categoric"

    @property
    def levels(self):
        return self.term.levels

    def __str__(self):
        args = [
            f"n_missing: {self._n_missing}",
            f"n_observed: {self.n_observed}",
        ]
        if self.coords:
            args.append(f"coords: {self.coords}")
        return self.make_str(args)
