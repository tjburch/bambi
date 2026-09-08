"""Formula lowering and coefficient designs for structured group effects."""

from dataclasses import dataclass, field
import json
from numbers import Real
from itertools import combinations
import warnings

import formulae as fm
from formulae import expr
from formulae.parser import Parser
from formulae.scanner import Scanner
from formulae.matrices import CommonEffectsMatrix
from formulae.terms.call import Call
from formulae.transforms import CategoricalBox
import numpy as np
import pandas as pd

from bambi.priors.prior import Prior
from bambi.terms.group_specific import GroupSpecificTerm

STRUCTURES = {"ar1", "ou", "cs", "toep", "us"}


def warn_confounded_blocks(terms):
    """Flag indistinguishable structured designs without changing their priors."""
    blocks = [term for term in terms if isinstance(term, StructuredGroupSpecificTerm)]
    for left, right in combinations(blocks, 2):
        if left.block.kind != right.block.kind or repr(left.prior) != repr(right.prior):
            continue
        if not np.array_equal(left.predictor, right.predictor):
            continue
        if not np.array_equal(left.block.coordinates, right.block.coordinates):
            continue
        _, first_left, inverse_left = np.unique(
            left.group_index, return_index=True, return_inverse=True
        )
        _, first_right, inverse_right = np.unique(
            right.group_index, return_index=True, return_inverse=True
        )
        if np.array_equal(first_left[inverse_left], first_right[inverse_right]):
            warnings.warn(
                f"Structured blocks '{left.name}' and '{right.name}' have identical grouping "
                "partitions and coefficient designs. Their variance components may not be "
                "distinguishable from the data.",
                RuntimeWarning,
                stacklevel=3,
            )


class StructuredGroupSpecificTerm(GroupSpecificTerm):
    """A group-specific coefficient block with a joint Gaussian prior."""

    def __init__(self, term, block, prior, prefix, noncentered):
        self.block = block
        super().__init__(term, prior, prefix, noncentered)

    @property
    def name(self):
        return f"{self.prefix}_{self.block.name}" if self.prefix else self.block.name

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("Structured priors must be a dictionary of covariance parameters.")
        keys = {
            "ar1": {"sd", "rho"},
            "ou": {"sd", "decay"},
            "cs": {"sd", "rho"},
            "toep": {"sd", "partial"},
            "us": {"sd", "eta", "correlation"},
        }[self.block.kind]
        if set(value) - keys:
            raise ValueError(f"Unknown covariance prior keys: {sorted(set(value) - keys)}")
        if "eta" in value and "correlation" in value:
            raise ValueError("Use either a fixed correlation matrix or an LKJ eta, not both.")
        self._prior = value.copy()

    def build_prior(self, response_scale=1):
        # Temporal scales describe a latent process, not dummy-column frequencies.
        self._prior.setdefault("sd", Prior("HalfNormal", sigma=2.5 * response_scale))
        if self.block.kind in {"ar1", "cs"}:
            self._prior.setdefault("rho", Prior("Normal", mu=0, sigma=0.5))
        elif self.block.kind == "ou":
            self._prior.setdefault("decay", Prior("Exponential", lam=1))
        elif self.block.kind == "toep":
            self._prior.setdefault("partial", Prior("Normal", mu=0, sigma=0.5))
        elif "correlation" not in self._prior:
            self._prior.setdefault("eta", 2)


def _unwrap(node):
    while isinstance(node, expr.Grouping):
        node = node.expression
    return node


def _source(node):  # pylint: disable=too-many-return-statements
    """Serialize a formula AST without changing operator precedence."""
    if isinstance(node, expr.Binary):
        return f"({_source(node.left)} {node.operator.lexeme} {_source(node.right)})"
    if isinstance(node, expr.Grouping):
        return f"({_source(node.expression)})"
    if isinstance(node, expr.Unary):
        return f"{node.operator.lexeme}{_source(node.right)}"
    if isinstance(node, expr.Variable):
        name = node.name.lexeme
        return name if node.level is None else f"{name}[{_source(node.level)}]"
    if isinstance(node, expr.QuotedName):
        return node.expression.lexeme
    if isinstance(node, expr.Literal):
        return node.lexeme if node.lexeme is not None else repr(node.value)
    if isinstance(node, expr.Assign):
        return f"{_source(node.name)}={_source(node.value)}"
    if isinstance(node, expr.Call):
        return f"{_source(node.callee)}({', '.join(_source(arg) for arg in node.args)})"
    raise ValueError(f"Unsupported formula expression: {type(node).__name__}")


def _sum_source(node):
    node = _unwrap(node)
    if isinstance(node, expr.Binary) and node.operator.kind == "PLUS":
        return f"{_sum_source(node.left)} + {_sum_source(node.right)}"
    return _source(node)


def _names(node):  # pylint: disable=too-many-return-statements
    if isinstance(node, (expr.Variable, expr.QuotedName)):
        return [_source(node)]
    if isinstance(node, expr.Call):
        return [name for arg in node.args for name in _names(arg)]
    if isinstance(node, expr.Binary):
        return _names(node.left) + _names(node.right)
    if isinstance(node, expr.Grouping):
        return _names(node.expression)
    if isinstance(node, expr.Unary):
        return _names(node.right)
    if isinstance(node, expr.Assign):
        return _names(node.value)
    return []


def _group_names(node):
    node = _unwrap(node)
    if isinstance(node, (expr.Variable, expr.QuotedName)):
        return [_source(node)]
    if isinstance(node, expr.Binary) and node.operator.kind == "COLON":
        return _group_names(node.left) + _group_names(node.right)
    raise ValueError("Structured grouping factors must be names joined with ':'.")


def encode_groups(*columns):
    """Encode tuples without delimiter collisions or unobserved Cartesian cells."""
    if any(pd.isna(column).any() for column in columns):
        raise ValueError("Structured grouping factors cannot contain missing values.")

    def key(value):
        if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
            if not np.isfinite(value):
                raise ValueError("Numeric grouping identifiers must be finite.")
            return ("number", str(int(value)) if value == int(value) else str(float(value)))
        return (type(value).__name__, str(value))

    values = [json.dumps([key(value) for value in row]) for row in zip(*columns)]
    return pd.Categorical(values)


@dataclass
class StructuredBlock:  # pylint: disable=too-many-instance-attributes
    kind: str
    name: str
    expression: str
    variables: tuple[str, ...]
    group_variables: tuple[str, ...]
    encoder_name: str
    group_encoder_name: str
    max_lag: int | None = None
    namespace: dict = field(default_factory=dict, repr=False)
    coordinates: np.ndarray | None = field(default=None, init=False)
    design: CommonEffectsMatrix | None = field(default=None, init=False, repr=False)

    def __call__(self, *columns):
        if self.kind == "us":
            data = pd.DataFrame(dict(zip((name.strip("`") for name in self.variables), columns)))
            if self.design is None:
                self.design = fm.design_matrices(
                    self.expression, data, extra_namespace=self.namespace
                ).common
                if self.design is None:
                    raise ValueError("An unstructured block needs at least one coefficient.")
                self.coordinates = np.asarray(self.design.as_dataframe().columns)
                return np.asarray(self.design.design_matrix)
            self._validate_categories(data)
            return np.asarray(self.design.evaluate_new_data(data).design_matrix)

        values = np.asarray(columns[0])
        if pd.isna(values).any():
            raise ValueError("Structured time or visit values cannot be missing.")
        if self.kind in {"ar1", "ou", "toep"}:
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                raise ValueError(f"{self.kind} requires finite numeric time values.")
            if self.kind != "ou" and not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{self.kind} requires integer time values.")
            if np.any(np.abs(values.astype(float)) > 2**52):
                raise ValueError("Time magnitudes must not exceed 2**52; rescale or recenter time.")
        if self.coordinates is None:
            if self.kind == "cs" and isinstance(columns[0].dtype, pd.CategoricalDtype):
                self.coordinates = np.asarray(columns[0].cat.categories)
            else:
                self.coordinates = np.unique(values)
            if len(self.coordinates) < 2:
                raise ValueError(
                    "A structured correlation requires at least two time/visit levels."
                )
            if self.kind == "toep" and np.ptp(self.coordinates) > self.max_lag:
                raise ValueError("Time span exceeds the declared Toeplitz max_lag.")
        indices = pd.Index(self.coordinates).get_indexer(values)
        if (indices < 0).any():
            raise ValueError("Prediction requires declared visit levels for this structure.")
        return np.eye(len(self.coordinates))[indices]

    def _validate_categories(self, data):
        """Check fitted coefficient levels, including transformed categorical calls."""
        for term in self.design.terms.values():
            for component in getattr(term, "components", ()):
                if component.kind != "categoric":
                    continue
                if isinstance(component, Call):
                    try:
                        values = component.call.eval(data, component.env)
                    except ValueError as error:
                        raise ValueError(
                            f"Cannot evaluate '{component.name}' on prediction data: {error}. "
                            "Explicit C(..., levels=...) declarations require all declared "
                            "levels in each data set; use an ordered categorical "
                            "column for subset predictions."
                        ) from error
                    if isinstance(values, CategoricalBox):
                        values = values.data
                else:
                    values = data[component.name]
                if not set(values).issubset(component.levels):
                    raise ValueError(
                        "Unstructured prediction requires declared coefficient levels."
                    )

    def prediction_design(self, data):
        """Return target rows and an ordered extension of the fitted coordinates."""
        columns = [data[name.strip("`")] for name in self.variables]
        if self.kind in {"us", "cs"}:
            return self(*columns), ()
        values = np.asarray(columns[0])
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("Prediction times must be finite numeric values.")
        if self.kind != "ou" and not np.equal(values, np.floor(values)).all():
            raise ValueError("AR1 and Toeplitz prediction times must be integers.")
        if np.any(np.abs(values.astype(float)) > 2**52):
            raise ValueError("Time magnitudes must not exceed 2**52; rescale or recenter time.")
        unseen = np.setdiff1d(values, self.coordinates)
        coordinates = np.concatenate([self.coordinates, unseen])
        if self.kind == "toep" and np.ptp(coordinates.astype(float)) > self.max_lag:
            raise ValueError("Prediction times exceed the declared Toeplitz horizon.")
        indices = pd.Index(coordinates).get_indexer(values)
        return np.eye(len(coordinates))[indices], tuple(unseen)


def lower_covariance_formula(formula, namespace, prefix="", data_columns=None):
    """Lower additive wrappers to explicit formulae coefficient and grouping calls."""
    tree = Parser(Scanner(formula).scan()).parse()
    blocks = {}
    names_seen = set()

    def visit(node, additive=True):
        node = _unwrap(node)
        if isinstance(node, expr.Call) and _source(node.callee) in STRUCTURES:
            if not additive:
                raise ValueError("Covariance wrappers must be additive group-specific terms.")
            kind = _source(node.callee)
            if not node.args:
                raise ValueError("A covariance wrapper requires '(expression | group)'.")
            argument = _unwrap(node.args[0])
            if not isinstance(argument, expr.Binary) or argument.operator.kind != "PIPE":
                raise ValueError("A covariance wrapper requires '(expression | group)'.")
            options = {}
            for option in node.args[1:]:
                if not isinstance(option, expr.Assign):
                    raise ValueError("Covariance options must be named.")
                key = _source(option.name)
                if key in options or key != "max_lag" or kind != "toep":
                    raise ValueError(f"Invalid covariance option: {key}")
                if not isinstance(option.value, expr.Literal):
                    raise ValueError("max_lag must be a positive integer literal.")
                options[key] = option.value.value
            max_lag = options.get("max_lag")
            if kind == "toep" and (
                isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 1
            ):
                raise ValueError("toep requires a positive integer max_lag.")
            expression = _sum_source(argument.left)
            variables = tuple(dict.fromkeys(_names(argument.left)))
            if data_columns is not None:
                variables = tuple(
                    name
                    for name in variables
                    if name.strip("`") in data_columns or name.strip("`") not in namespace
                )
            group_variables = tuple(_group_names(argument.right))
            if len(set(group_variables)) != len(group_variables):
                raise ValueError("Grouping interaction components must be distinct.")
            if kind != "us":
                left = _unwrap(argument.left)
                if not (
                    isinstance(left, expr.Binary)
                    and left.operator.kind == "PLUS"
                    and isinstance(_unwrap(left.left), expr.Literal)
                    and _unwrap(left.left).value == 0
                    and isinstance(_unwrap(left.right), (expr.Variable, expr.QuotedName))
                ):
                    raise ValueError(f"Use {kind}(0 + time | group) with one time/visit column.")
            if not variables:
                raise ValueError("A covariance coefficient expression needs a data column.")
            group_name = ":".join(group_variables)
            suffix = f", max_lag={max_lag}" if max_lag is not None else ""
            name = f"{kind}({expression} | {group_name}{suffix})"
            identity = (kind, expression, group_variables, max_lag)
            if identity in names_seen:
                raise ValueError(f"Duplicate covariance block: {name}")
            names_seen.add(identity)
            index = len(blocks)
            encoder = f"bambi_cov_internal_{prefix}{index}"
            group_encoder = f"bambi_group_internal_{prefix}{index}"
            if encoder in namespace or group_encoder in namespace:
                raise ValueError(
                    "The bambi_cov_internal_ and bambi_group_internal_ namespaces are reserved."
                )
            block = StructuredBlock(
                kind,
                name,
                expression,
                variables,
                group_variables,
                encoder,
                group_encoder,
                max_lag,
                namespace.copy(),
            )
            blocks[encoder] = block
            namespace[encoder] = block
            namespace[group_encoder] = encode_groups
            expression_call = f"{encoder}({', '.join(variables)})"
            group_call = f"{group_encoder}({', '.join(group_variables)})"
            return f"(0 + {expression_call} | {group_call})"
        if isinstance(node, expr.Binary):
            is_additive = additive and node.operator.kind in {"PLUS", "TILDE"}
            left = visit(node.left, is_additive and node.operator.kind != "TILDE")
            right = visit(node.right, is_additive)
            result = f"{left} {node.operator.lexeme} {right}"
            return result if node.operator.kind == "TILDE" else f"({result})"
        if isinstance(node, expr.Call):
            return f"{_source(node.callee)}({', '.join(visit(arg, False) for arg in node.args)})"
        if isinstance(node, expr.Unary):
            return node.operator.lexeme + visit(node.right, False)
        return _source(node)

    lowered = visit(tree)
    return (lowered if blocks else formula), blocks
