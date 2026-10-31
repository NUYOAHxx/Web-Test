#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Common IK engine protocol and engine factory.

Provides unified runtime interfaces (IKEngine, DualArmIKEngine), an extensible
engine registry, and factory functions (create_ik_solver, create_dual_arm_ik_solver)
for inverse kinematics solver engines across the ha_arm_ik framework.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class IKEngine(Protocol):
    """Runtime interface protocol shared by single-arm IK solver engines."""

    def solve(
        self,
        T_target: Any,
        frame: str = "local",
        q_prev: Any = None,
        update_prev: bool = True,
        verbose: bool | None = None,
        allow_approximate: bool = False,
        orientation_mode: str = "full",
        tool_axis: Any = None,
        tool_cone_half_angle: Any = None,
        *,
        joint_state: Any = None,
        **kwargs: Any,
    ) -> Tuple[Any, str, Any]:
        """Solve single-arm IK and return ``(q, status, IKResult)``."""
        ...

    def print_info(self) -> None:
        """Print engine configuration and status."""
        ...

    def close(self) -> None:
        """Release engine resources."""
        ...


@runtime_checkable
class DualArmIKEngine(Protocol):
    """Runtime interface protocol shared by dual-arm IK solver engines."""

    def solve(
        self,
        T_left: Any = None,
        T_right: Any = None,
        q_prev: Any = None,
        update_prev: bool = True,
        verbose: bool = False,
        allow_approximate: bool = False,
        orientation_mode: str = "full",
        tool_axis: Any = None,
        tool_cone_half_angle: Any = None,
        *,
        joint_state: Any = None,
        time_budget_s: Any = None,
        **kwargs: Any,
    ) -> Tuple[Any, str, Any]:
        """Solve dual-arm coordinated IK and return ``(q, status, IKResult)``."""
        ...

    def fk(self, q: Any) -> Dict[str, Any]:
        """Compute forward kinematics for left and right arms."""
        ...

    def reset_prev(self) -> None:
        """Reset previous joint state cache."""
        ...


# Canonical engine identifiers
ENGINE_DLS = "dls"
ENGINE_SINGLE_ARM = "single_arm"
ENGINE_DUAL_ARM = "dual_arm"
ENGINE_QP = "qp"

# Supported engine names tuple (for backward compatibility and introspection)
SUPPORTED_ENGINES: Tuple[str, ...] = (
    "dls",
    "single_arm",
    "dual_arm",
    "qp",
)

# Custom engine registry
_ENGINE_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_ik_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    override: bool = False,
) -> None:
    """Register a custom IK solver engine factory.

    Args:
        name: Engine name (case-insensitive).
        factory: Callable taking constructor kwargs and returning an IK solver.
        override: If True, allow overriding an existing registered engine.
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Engine name cannot be empty.")
    if not callable(factory):
        raise TypeError(f"Factory for engine '{key}' must be callable.")
    if key in _ENGINE_REGISTRY and not override:
        raise ValueError(
            f"Engine '{key}' is already registered. Set override=True to replace it."
        )
    _ENGINE_REGISTRY[key] = factory


def unregister_ik_engine(name: str) -> bool:
    """Unregister a custom IK solver engine.

    Returns:
        True if the engine was unregistered, False if it was not found.
    """
    key = str(name).strip().lower()
    return _ENGINE_REGISTRY.pop(key, None) is not None


def get_supported_engines() -> List[str]:
    """Return a list of all currently supported engine names (built-in + registered)."""
    engines = list(SUPPORTED_ENGINES)
    for name in _ENGINE_REGISTRY:
        if name not in engines:
            engines.append(name)
    return engines


def _create_dls_solver(**kwargs: Any) -> Any:
    """Create a DLS/portfolio single-arm IK solver."""
    try:
        from ha_arm_ik.single_arm_ik import SingleArmIKSolver
        return SingleArmIKSolver(**kwargs)
    except ImportError:
        from ha_arm_ik.solver import HAArmIKSolver
        return HAArmIKSolver(**kwargs)


def _create_dual_arm_solver(**kwargs: Any) -> Any:
    """Create a dual-arm coordinated IK solver."""
    from ha_arm_ik.dual_arm_ik import DualArmIKSolver
    return DualArmIKSolver(**kwargs)


def _create_qp_solver(**kwargs: Any) -> Any:
    """Create a QP/Pink-based IK solver."""
    # 1. Check for dedicated ha_arm_ik QP solver module
    try:
        from ha_arm_ik.solver_pink import PinkQPIKSolver  # type: ignore
        return PinkQPIKSolver(**kwargs)
    except ImportError:
        pass

    # 2. Check for humanoid_pink_ik module
    try:
        from core.solver.humanoid_pink_ik import HumanoidPinkIKSolver  # type: ignore
        return HumanoidPinkIKSolver(**kwargs)
    except ImportError:
        pass

    # 3. Informative error when no QP backend is installed or found
    supported_list = get_supported_engines()
    raise RuntimeError(
        "QP/Pink IK engine is not available. Please ensure 'pink' and a Pink-based solver "
        "module (e.g. ha_arm_ik.solver_pink or core.solver.humanoid_pink_ik) are installed and "
        "accessible on PYTHONPATH. Alternatively, register a custom QP solver using "
        f"register_ik_engine('qp', ...). Supported engines: {supported_list}"
    )


def create_ik_solver(engine: str = "dls", **kwargs: Any) -> Any:
    """Create an IK engine instance by name.

    Supported built-in engines:
      - ``"dls"`` / ``"single_arm"`` / ``"ha_arm"`` / ``"portfolio"``:
        Selects :class:`SingleArmIKSolver` (the numerical DLS/SNS portfolio solver
        with strict load protection, comfort windows, and jump limiting).
      - ``"dual"`` / ``"dual_arm"``:
        Selects :class:`DualArmIKSolver` (16-DoF dual-arm coordinated IK solver).
      - ``"qp"`` / ``"pink"``:
        Selects the differential-IK QP engine backed by Pink.

    Constructor options are engine-specific and passed directly through as ``**kwargs``.

    Args:
        engine: Engine name (case-insensitive).
        **kwargs: Arguments forwarded to the solver constructor (e.g. urdf, side, config).

    Returns:
        An instantiated IK solver engine adhering to :class:`IKEngine` or :class:`DualArmIKEngine`.

    Raises:
        ValueError: If engine name is unknown.
        RuntimeError: If the requested engine backend cannot be loaded.
    """
    normalized_engine = str(engine).strip().lower()

    # Check custom registered engines first
    if normalized_engine in _ENGINE_REGISTRY:
        return _ENGINE_REGISTRY[normalized_engine](**kwargs)

    # Built-in single-arm DLS portfolio engines
    if normalized_engine in ("dls", "ha_arm", "portfolio", "single", "single_arm"):
        return _create_dls_solver(**kwargs)

    # Built-in dual-arm engine
    if normalized_engine in ("dual", "dual_arm", "dual_arm_ik"):
        return _create_dual_arm_solver(**kwargs)

    # Built-in QP / Pink engine
    if normalized_engine in ("qp", "pink", "pink_qp"):
        return _create_qp_solver(**kwargs)

    valid_engines = get_supported_engines()
    raise ValueError(
        f"Unknown IK engine: '{engine}'. Expected one of {valid_engines}"
    )


def create_dual_arm_ik_solver(**kwargs: Any) -> Any:
    """Convenience helper to create a :class:`DualArmIKSolver`."""
    return create_ik_solver(engine=ENGINE_DUAL_ARM, **kwargs)


__all__ = [
    "IKEngine",
    "DualArmIKEngine",
    "SUPPORTED_ENGINES",
    "ENGINE_DLS",
    "ENGINE_SINGLE_ARM",
    "ENGINE_DUAL_ARM",
    "ENGINE_QP",
    "create_ik_solver",
    "create_dual_arm_ik_solver",
    "register_ik_engine",
    "unregister_ik_engine",
    "get_supported_engines",
]
