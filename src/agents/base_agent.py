"""
base_agent.py

Abstract base class for all AI agents in this project.
Provides a consistent interface, built-in logging, error handling,
and a simple run() lifecycle that subclasses implement.

Inheritance example:
    class MyAgent(BaseAgent):
        def _execute(self, inputs):
            return {"result": "done"}
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Standardised result returned by every agent run."""

    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    agent_name: str = ""

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return (
            f"[{self.agent_name}] status={status} "
            f"latency={self.latency_ms:.1f}ms steps={len(self.steps)}"
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Responsibilities:
    - Enforce a run() -> AgentResult interface
    - Centralise timing, logging, and error handling
    - Provide a step log that subclasses can populate

    Subclasses must implement:
    - _execute(inputs: Dict) -> Dict
    """

    def __init__(self, name: str, max_retries: int = 1) -> None:
        """
        Args:
            name: Human-readable agent identifier used in logs.
            max_retries: How many times to retry on unexpected exceptions.
        """
        self.name = name
        self.max_retries = max_retries
        self._steps: List[Dict[str, Any]] = []
        logger.info(f"Agent '{self.name}' initialised (max_retries={max_retries})")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute the agent with optional retry logic.

        Args:
            inputs: Arbitrary input dictionary passed to _execute().

        Returns:
            AgentResult containing output, steps, and timing.
        """
        inputs = inputs or {}
        self._steps = []  # reset step log for each new run
        attempt = 0
        t0 = time.perf_counter()

        while attempt <= self.max_retries:
            attempt += 1
            try:
                logger.info(f"[{self.name}] run attempt {attempt}/{self.max_retries + 1}")
                output = self._execute(inputs)
                latency_ms = (time.perf_counter() - t0) * 1000
                result = AgentResult(
                    success=True,
                    output=output,
                    steps=list(self._steps),
                    latency_ms=latency_ms,
                    agent_name=self.name,
                )
                logger.info(f"[{self.name}] completed successfully in {latency_ms:.1f}ms")
                return result

            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                logger.warning(
                    f"[{self.name}] attempt {attempt} failed: {exc}\n{tb}"
                )
                if attempt > self.max_retries:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return AgentResult(
                        success=False,
                        error=str(exc),
                        steps=list(self._steps),
                        latency_ms=latency_ms,
                        agent_name=self.name,
                    )

    # ------------------------------------------------------------------
    # Abstract method subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core agent logic.

        Args:
            inputs: Input data for this run.

        Returns:
            Output dictionary with agent results.
        """
        ...

    # ------------------------------------------------------------------
    # Protected helpers for subclasses
    # ------------------------------------------------------------------

    def _log_step(
        self,
        step_name: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an intermediate step in the step log.
        Call this from _execute() to make progress visible.

        Args:
            step_name: Short description of this step.
            data: Optional metadata to attach.
        """
        step = {"step": step_name, "data": data or {}}
        self._steps.append(step)
        logger.debug(f"[{self.name}] step: {step_name}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
