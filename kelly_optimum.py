#!/usr/bin/env python

import casadi
from attr import attrs
from typing import List, Dict, Union, Any, Iterable, Tuple, Callable
import pandas as pd
import sys
from toposort import toposort_flatten
from collections import OrderedDict
import numpy as np

"""
This module calculates growth optimal (Kelly optimal) portfolios.

Strategy:
- define the known unknowns: discrete random variates
- define a discrete bayesian network over these unknowns
- define assets. Every asset maps world states to values.
- define portfolios. Portfolios are linear combinations of assets.
- calculate kelly criterion of portfolio
- calculate gradient of kelly criterion using ad

TODO:
- calculate sensitivity of the expected log return to numeric inputs.
"""

DiscreteValue = Union[str, int, float]

# conditional probability tables are stored as nested dicts, forming
# trees with probabilities in their leaves.
ConditionalProbabilityTree = Union[Dict[DiscreteValue, float], Dict[DiscreteValue, Dict]]
    

@attrs(auto_attribs=True, slots=True)
class DiscreteUnknown:
    name: str
    determinants: List['DiscreteUnknown']  # unknown variables that affect this variable
    probabilities: ConditionalProbabilityTree

    def states_given(self, assignment: Dict) -> Iterable[Tuple[Any, float]]:
        #for value, probability in pivot.states_given(assignment):
        return self._states_given_from(self.probabilities, 0, assignment)

    def _states_given_from(self, p_root, determinant_pos, assignment):
        if determinant_pos == len(self.determinants):
            return p_root.items()
        det_name = self.determinants[determinant_pos]
        det_val = assignment[det_name]
        p_child = p_root[det_val]
        return self._states_given_from(p_child, determinant_pos+1, assignment)
    
    
@attrs(auto_attribs=True, slots=True)
class Asset:
    valuation: DiscreteUnknown
    current_price: float

    @property
    def name(self):
        return self.valuation.name

@attrs(auto_attribs=True, slots=True)
class Model:
    unknowns: Dict[str, DiscreteUnknown] # topologically ordered
    assets: List[Asset]


class DS:
    def __init__(self, *args, **kwargs):
        self.__dict__ = {}
        for d in args:
            self.__dict__.update(d)
        self.__dict__.update(kwargs)

        
@attrs(auto_attribs=True, init=False, slots=True)
class PortfolioOptimization:
    state_probabilities: np.ndarray # [state]
    asset_values: np.ndarray  # [state, asset]
    opti: casadi.Opti
    expressions: object

    def __init__(self, state_probabilities, asset_values):
        self.state_probabilities = state_probabilities
        self.asset_values = asset_values
        self.validate()

        n_states = self.n_states
        n_assets = self.n_assets
        
        self.opti = opti = casadi.Opti()
        self.expressions = {}
        p_asset_values = opti.parameter(n_states, n_assets)
        opti.set_value(p_asset_values, asset_values)
        p_state_probabilities = opti.parameter(n_states)
        opti.set_value(p_state_probabilities, state_probabilities)

        v_coded = opti.variable(n_assets)
        coded_total = casadi.sum1(v_coded)
        asset_weights = v_coded / (coded_total + 1)  # the rest is cash
        cash = 1.0 / (coded_total + 1) # 1-sum(weights)
        state_values = casadi.mtimes(p_asset_values, asset_weights) + cash
        kelly_criterion = casadi.dot(casadi.log(state_values), p_state_probabilities)
        opti.minimize(-kelly_criterion)
        opti.subject_to(v_coded >= 0)
        opti.set_initial(v_coded, np.ones(n_assets))
        opti.solver('ipopt')

        self.expressions = DS({
            name: value
            for name, value in locals().items()
            if isinstance(value, casadi.MX)})

    def solve(self, **kwargs):
        solution = self.opti.solve()
        e = self.expressions
        expected_log2_of_value = solution.value(e.kelly_criterion) / np.log(2)
        return dict(weights = np.reshape(solution.value(e.asset_weights), (self.n_assets,)),
                    doubling = expected_log2_of_value,
                    complete_solution = solution,
                    expressions = e)

        
    def validate(self):
        assert self.state_probabilities.shape == (self.n_states,)
        assert self.asset_values.shape == (self.n_states, self.n_assets)
        assert (self.state_probabilities >= 0).all()
        assert abs(self.state_probabilities.sum()-1) < 0.000001
    
    @property
    def n_assets(self):
        return self.asset_values.shape[1]

    @property
    def n_states(self):
        return len(self.state_probabilities)
    

def load_model(cpts_path: str):
    engine = "odf" if cpts_path.endswith("ods") else None
    sheets = pd.read_excel(cpts_path, engine=engine, sheet_name=None)
    CONSTITUENTS = "assets"
    PROBABILITY_KEYS = "probability", "conditional probability"
    CURRENT_PRICE = "current price"
    ASSET = "asset"

    try:
        current_df = sheets.pop(CONSTITUENTS)
    except KeyError:
        raise ValueError(f"missing sheet: {CONSTITUENTS}")
    assert len(current_df.columns) == 2
    assert current_df.columns[0] == ASSET
    assert current_df.columns[1] == CURRENT_PRICE
    current_prices = {k.strip(): float(v)
                      for k, v in (current_df.itertuples(index=False))}
    assert len(current_prices) == len(current_df), f"duplicate name in {CONSTITUENTS}"
    asset_names = list(current_prices)

    unknowns_inputs = {}
    dependencies = {}
    for sheet_name, df in sheets.items():
        columns = df.columns
        p_cols = [i for i,key in enumerate(columns) if key in PROBABILITY_KEYS]
        if len(p_cols) > 1:
            raise ValueError(f"There can only be one probability column in sheet {sheet_name!r}, but we have: {columns[p_cols]!r}")
        if len(p_cols) == 0:
            raise ValueError(f"One of the columns in sheet {sheet_name!r} must be one of: {PROBABILITY_KEYS!r}")
        p_col, = p_cols

        probability = df.values[:, p_col].astype(float)
        assert (probability >= 0).all(), f"negative probabilities in sheet {sheet_name!r}"

        determinants = columns.values[:p_col]
        
        for dependent_col in range(p_col+1, len(columns)):
            dependent_name = columns[dependent_col].strip()
            states = df.values[:, dependent_col]
            try:
                current_price = current_prices[dependent_name]
            except KeyError:
                pass
            else:
                states = states.astype(float) / current_price
            dependencies[dependent_name] = set(determinants)
            unknowns_inputs[dependent_name] = (df, p_col, dependent_col)

    order = toposort_flatten(dependencies)
    unknowns = OrderedDict()
    for dependent_name in order:
        df, p_col, dependent_col = unknowns_inputs[dependent_name]
        probability_tree = mk_probability_tree(df, 0, p_col, dependent_col)
        unknowns[dependent_name] = DiscreteUnknown(
            dependent_name, df.columns.values[:p_col], probability_tree)

    assets = [Asset(unknowns[name], current_prices[name]) for name in asset_names]
    return Model(unknowns, assets)


def strict_zip(x,y):
    if len(x)!=len(y):
        raise ValueError(f"lengths mismatch: {len(x)} != {len(y)}")
    return zip(x,y)


def mk_probability_tree(df, determinant_col, p_col, dependent_col) -> Dict:
    if determinant_col == p_col:
        p = df.values[:, p_col]
        assert abs(p.sum()-1) < 0.0001
        return dict(strict_zip(df.values[:, dependent_col], p))
    determinant = df.columns[determinant_col]
    return {condition: mk_probability_tree(group_df, determinant_col+1, p_col, dependent_col)
            for condition, group_df in df.groupby(determinant)}
    

def mk_optimization_problem(model: Model) -> PortfolioOptimization:
    state_probabilities = []
    asset_values = []
    unknown_order = list(model.unknowns)
    generate_states(state_probabilities, asset_values, 1.0, model, unknown_order, 0, {})
    return PortfolioOptimization(np.array(state_probabilities), np.array(asset_values))


def generate_states(state_probabilities, asset_values,
                    base_probabiity, model, order, generation_pos,
                    assignment) -> None:
    if generation_pos == len(order):
        state_probabilities.append(base_probabiity)
        asset_values.append([assignment[asset.valuation.name] for asset in model.assets])
        return
    
    name = order[generation_pos]
    pivot = model.unknowns[name]
    generation_pos += 1
    for value, probability in pivot.states_given(assignment):
        assignment[name] = value
        generate_states(state_probabilities, asset_values,
                        base_probabiity*probability, model, order,
                        generation_pos, assignment)
    del assignment[name]


def run(cpts_path: str) -> None:
    model = load_model(cpts_path)
    problem = mk_optimization_problem(model)
    solution = problem.solve()
    print(solution)
    for asset, weight in strict_zip(model.assets, solution['weights']):
        print(f"{asset.name}: {100*weight} %")
    cash_weight = 1 - solution['weights'].sum()
    print(f"cash: {100*cash_weight} %")
    print(f"doubling rate: {solution['doubling']}")


if __name__ == '__main__':
    try:
        cpts_path, = sys.argv[1:]
    except ValueError:
        print(f"Usage: {sys.argv[0]} /path/to/conditional_probability_tables.ods")
        sys.exit(1)
    else:
        run(cpts_path)

