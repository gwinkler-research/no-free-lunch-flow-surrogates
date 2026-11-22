# Adapter registry: maps an adapter name to its BenchmarkAdapter subclass. Entries
# are added by the @register_adapter decorator when the adapter submodules are
# imported in _load_builtins() below.
"""Adapter registry for the unified benchmark runner.

An adapter wraps a model class (tools/models/* or benchmark/adapters/_dynamics_models.py)
and exposes the uniform interface that benchmark.runner calls:

    build(cfg, seed)                     -> model_state (a ModelState dict)
    fit(model_state, train, val, cfg, seed) -> (model_state, loss_history: dict)
    predict_eval(model_state, test, split_spec, cfg) -> (arrays: dict, meta: dict)

`train`/`val`/`test` are dataset views produced by benchmark.runner
(see DatasetView there). `model_state` carries everything fit/predict need
(model object, normalisation stats, paths). `arrays` is the kvs/field schema
expected by benchmark.prediction_io.save_prediction:
    u_eval (n_test,201,n_eval_p2,2), p_eval (n_test,201,n_eval_p1),
    case_ids (n_test,), t_grid (201,).

The CMP-3D adapters (cmp3d_*) follow the same build/fit/predict_eval pattern but
are not registered here. scripts/run_cmp3d_gate2.py instantiates them directly,
and their predict_eval returns the full 3D velocity field instead of `arrays`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

ModelState = Dict[str, Any]
LossHistory = Dict[str, list]
EvalArrays = Dict[str, Any]
EvalMeta = Dict[str, Any]


class BenchmarkAdapter(ABC):
    """Stateless adapter: holds only the config; all run state lives in ModelState."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @abstractmethod
    def build(self, cfg: dict, seed: int) -> ModelState: ...

    @abstractmethod
    def fit(self, model_state: ModelState, train, val, cfg: dict, seed: int) -> Tuple[ModelState, LossHistory]: ...

    @abstractmethod
    def predict_eval(self, model_state: ModelState, test, split_spec, cfg: dict) -> Tuple[EvalArrays, EvalMeta]: ...


# Global registry: adapter name (lowercased) -> adapter class.
ADAPTERS: Dict[str, type] = {}


def register_adapter(name: str):
    def _wrap(cls: type) -> type:
        key = str(name).strip().lower()
        if key in ADAPTERS:
            raise KeyError(f"adapter {key!r} already registered")
        ADAPTERS[key] = cls
        return cls
    return _wrap


def get_adapter(name: str) -> type:
    key = str(name).strip().lower()
    if key not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[key]


# Import submodules for their @register_adapter side effects.
# Registered KVS adapters:
#   grid_convae        -> frozen ConvAE encoder (latent rows) and its roundtrip floor
#   latent_dynamics    -> AR-MLP and S4D/Mamba/GRU branch DeepONet rows (latent x AR)
#   latent_oneshot     -> GRU/S4D-branch one-shot rows (latent x one-shot)
#   kvs_field_deeponet -> field DeepONet row (direct x one-shot)
#   dmdc               -> DMDc row (linear latent baseline)
def _load_builtins() -> None:
    from . import (cmp3d_latent_ar, cmp3d_latent_oneshot, cmp3d_sdeeponet,  # noqa: F401
                   dmdc, grid_convae, kvs_field_deeponet, latent_dynamics, latent_oneshot)


_load_builtins()
