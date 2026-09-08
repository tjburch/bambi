import ast
from dataclasses import dataclass

import pandas as pd
import formulae as fm

SUPPORTED_FUNCTIONS = frozenset({"exp", "log", "sqrt"})


class ExpressionNode:
    """Base class for nodes in a nonlinear expression tree."""


@dataclass(frozen=True)
class Literal(ExpressionNode):
    """A numeric literal in a nonlinear expression."""

    value: int | float


@dataclass(frozen=True)
class Symbol(ExpressionNode):
    """A nonlinear parameter or observed data name."""

    name: str


@dataclass(frozen=True)
class UnaryOperation(ExpressionNode):
    """A unary arithmetic operation."""

    operator: str
    operand: ExpressionNode


@dataclass(frozen=True)
class BinaryOperation(ExpressionNode):
    """A binary arithmetic operation."""

    operator: str
    left: ExpressionNode
    right: ExpressionNode


@dataclass(frozen=True)
class FunctionCall(ExpressionNode):
    """A call to a supported single-argument function."""

    function: str
    argument: ExpressionNode


@dataclass(frozen=True)
class NonlinearExpression:
    """A parsed nonlinear expression.

    Attributes
    ----------
    source : str
        Original expression source.
    root : ExpressionNode
        Root of the parsed expression tree.
    symbols : frozenset of str
        Nonlinear parameter and observed data names used by the expression.
    """

    source: str
    root: ExpressionNode
    symbols: frozenset[str]

    @classmethod
    def parse(cls, source: str) -> "NonlinearExpression":
        """Parse a nonlinear expression from its source.

        Parameters
        ----------
        source : str
            Expression using supported arithmetic operators and functions.

        Returns
        -------
        NonlinearExpression
            Parsed expression and the symbols it references.

        Raises
        ------
        ValueError
            If the expression is malformed or contains unsupported syntax.
        """
        try:
            parsed = ast.parse(source, mode="eval")
        except SyntaxError as error:
            raise ValueError(f"Malformed nonlinear expression: {source!r}.") from error

        symbols = set()
        root = _convert_node(parsed.body, symbols)
        return cls(source=source, root=root, symbols=frozenset(symbols))


@dataclass
class NonlinearParameter:
    """Description of a likelihood parent defined by a nonlinear expression.

    Attributes
    ----------
    name : str
        Original likelihood parameter name.
    expression : NonlinearExpression
        Expression that defines the parameter on the link scale.
    data_names : tuple of str
        Observed data columns referenced directly by the expression.
    alias : str or None
        Name used in the backend graph and posterior output.
    is_parent : bool
        Whether this is the likelihood's parent parameter.
    """

    name: str
    expression: NonlinearExpression
    data_names: tuple[str, ...]
    alias: str | None = None
    is_parent: bool = True

    @property
    def label(self):
        """Return the aliased name when present, otherwise the original name."""
        return self.alias or self.name


def split_nonlinear_formula(formula: str) -> tuple[str, str]:
    """Separate a nonlinear formula into its response and expression.

    Parameters
    ----------
    formula : str
        Formula in the form ``response ~ expression``.

    Returns
    -------
    response_formula : str
        Intercept-only formula used to build the response design.
    expression : str
        Nonlinear expression from the right-hand side.

    Raises
    ------
    ValueError
        If the formula does not contain one response and one expression.
    """
    lhs, separator, rhs = formula.partition("~")
    if not separator or not lhs.strip() or not rhs.strip() or "~" in rhs:
        raise ValueError("A nonlinear formula must have the form 'response ~ expression'.")
    return f"{lhs.strip()} ~ 1", rhs.strip()


def resolve_nonlinear_data_names(expression, predictors, data) -> tuple[str, ...]:
    """Resolve and validate observed columns used by a nonlinear expression.

    Parameters
    ----------
    expression : NonlinearExpression
        Parsed nonlinear expression.
    predictors : Mapping
        Modeled nonlinear parameters keyed by their original names.
    data : pandas.DataFrame
        Model data containing observed expression inputs.

    Returns
    -------
    tuple of str
        Sorted names of numeric data columns used directly by the expression.

    Raises
    ------
    ValueError
        If a symbol is unresolved or an expression input is not numeric.
    """
    predictor_names = set(predictors)
    data_names = expression.symbols - predictor_names
    unknown = data_names - set(data.columns)
    if unknown:
        raise ValueError(
            "No nonlinear parameter formula or data column was found for symbol(s): "
            f"{sorted(unknown)}."
        )

    nonnumeric = [
        name for name in sorted(data_names) if not pd.api.types.is_numeric_dtype(data[name])
    ]
    if nonnumeric:
        raise ValueError(
            f"Nonlinear expression data must be numeric. Invalid column(s): {nonnumeric}."
        )
    return tuple(sorted(data_names))


def prepare_nonlinear_data(
    formula, expression, data, dropna, include_response=True, parameter_names=()
):
    """Prepare aligned, complete observations for every part of a nonlinear model.

    Parameters
    ----------
    formula : Formula
        Nonlinear model formula and its parameter formulas.
    expression : NonlinearExpression
        Parsed nonlinear expression.
    data : pandas.DataFrame
        Model or prediction data.
    dropna : bool
        Whether to remove incomplete rows instead of raising an error.
    include_response : bool
        Whether the response is required in ``data``.
    parameter_names : Collection of str
        Likelihood parameter names used to detect parameter dependencies.

    Returns
    -------
    pandas.DataFrame
        Data with a shared complete-row mask applied when requested.

    Raises
    ------
    ValueError
        If parameters depend on one another or required data are incomplete.
    """
    names = set(formula.additionals_lhs)
    variables = set(expression.symbols - names)
    if include_response:
        response_formula, _ = split_nonlinear_formula(formula.main)
        variables.update(fm.model_description(response_formula).var_names)
    for name, predictor_formula in zip(formula.additionals_lhs, formula.additionals):
        rhs = predictor_formula.partition("~")[2]
        predictor_variables = fm.model_description(rhs).var_names
        dependencies = (names | (set(parameter_names) - set(data.columns))) & predictor_variables
        if dependencies:
            raise ValueError(
                "Nonlinear parameters cannot depend on one another. "
                f"'{name}' references {sorted(dependencies)}."
            )
        variables.update(predictor_variables)

    columns = sorted(variables & set(data.columns))
    incomplete = data[columns].isna().any(axis=1)
    if incomplete.any():
        if not dropna:
            raise ValueError(f"'data' contains {incomplete.sum()} incomplete rows.")
        data = data.loc[~incomplete].copy()
    if len(data) == 0:
        raise ValueError("'data' does not contain any complete observation.")
    return data


_BINARY_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
}

_UNARY_OPERATORS = {
    ast.UAdd: "+",
    ast.USub: "-",
}


def _convert_node(node: ast.AST, symbols: set[str]) -> ExpressionNode:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Nonlinear expressions only support numeric literals.")
        return Literal(node.value)

    if isinstance(node, ast.Name):
        symbols.add(node.id)
        return Symbol(node.id)

    if isinstance(node, ast.BinOp):
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(
                f"Unsupported operator '{type(node.op).__name__}' in nonlinear expression."
            )
        return BinaryOperation(
            operator,
            _convert_node(node.left, symbols),
            _convert_node(node.right, symbols),
        )

    if isinstance(node, ast.UnaryOp):
        operator = _UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(
                f"Unsupported operator '{type(node.op).__name__}' in nonlinear expression."
            )
        return UnaryOperation(operator, _convert_node(node.operand, symbols))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Nonlinear functions must be referenced by name.")
        if node.func.id not in SUPPORTED_FUNCTIONS:
            supported = ", ".join(sorted(SUPPORTED_FUNCTIONS))
            raise ValueError(
                f"Unsupported nonlinear function '{node.func.id}'. "
                f"Supported functions: {supported}."
            )
        if len(node.args) != 1 or node.keywords:
            raise ValueError(
                f"Nonlinear function '{node.func.id}' requires exactly one positional argument."
            )
        return FunctionCall(node.func.id, _convert_node(node.args[0], symbols))

    raise ValueError(f"Unsupported syntax '{type(node).__name__}' in nonlinear expression.")
