import operator

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.utils import INVERSE_LINKS
from bambi.families import Family
from bambi.nonlinear import (
    BinaryOperation,
    FunctionCall,
    Literal,
    NonlinearParameter,
    SUPPORTED_FUNCTIONS,
    Symbol,
    UnaryOperation,
)

_BINARY_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}

_FUNCTIONS = {name: getattr(pt, name) for name in SUPPORTED_FUNCTIONS}


def nonlinear_data_name(parameter_label: str, symbol: str) -> str:
    return f"{parameter_label}__{symbol}_data"


def build_nonlinear_parameter(
    parameter: NonlinearParameter,
    predictor_values: dict[str, pt.Variable],
    data,
    model: pm.Model,
    family: Family,
    parameters: dict[str, pt.Variable],
) -> pt.Variable:
    values = predictor_values.copy()
    for name in parameter.data_names:
        values[name] = pm.Data(
            nonlinear_data_name(parameter.label, name),
            np.asarray(data[name], dtype=float),
            dims="__obs__",
            model=model,
        )

    value = evaluate_expression(parameter.expression.root, values)
    link = family.link[parameter.name]
    inverse_link = INVERSE_LINKS.get(link.name, link.inverse_link)
    transform_predictor = transforms_registry.get_predictor_transform(family, parameter.name)
    if transform_predictor:
        value = transform_predictor(value, parameters, inverse_link)
    else:
        value = inverse_link(value)
    value = pt.as_tensor_variable(value)
    if any(value is variable for variable in model.deterministics):
        # Keep the parent distinct when PyMC clones direct deterministic views.
        value = value.copy()
    if value.ndim == 0:
        value = pt.broadcast_to(value, (model.dim_lengths["__obs__"],))
    return pm.Deterministic(parameter.label, value, dims="__obs__", model=model)


def build_new_nonlinear_data(parameter: NonlinearParameter, data) -> dict[str, np.ndarray]:
    missing = set(parameter.data_names) - set(data.columns)
    if missing:
        raise ValueError(f"New data is missing nonlinear expression column(s): {sorted(missing)}.")
    return {
        nonlinear_data_name(parameter.label, name): np.asarray(data[name], dtype=float)
        for name in parameter.data_names
    }


def evaluate_expression(node, values):
    if isinstance(node, Literal):
        return pt.as_tensor_variable(node.value)
    if isinstance(node, Symbol):
        return values[node.name]
    if isinstance(node, UnaryOperation):
        value = evaluate_expression(node.operand, values)
        return value if node.operator == "+" else -value
    if isinstance(node, BinaryOperation):
        left = evaluate_expression(node.left, values)
        right = evaluate_expression(node.right, values)
        return _BINARY_OPERATORS[node.operator](left, right)
    if isinstance(node, FunctionCall):
        return _FUNCTIONS[node.function](evaluate_expression(node.argument, values))
    raise TypeError(f"Unexpected nonlinear expression node: {type(node).__name__}.")
