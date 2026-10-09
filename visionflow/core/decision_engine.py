from typing import Optional, Any, Dict, List


class DecisionEngine:
    """
    Decision-making engine for VisionFlow.

    DecisionEngine does NOT perform ML/model optimization.

    Its responsibility is to:
        1. Analyze current performance
        2. Recommend the next action
        3. Validate constraints
        4. Select the best valid configuration
    """

    def __init__(
        self,
        min_accuracy: Optional[float] = None,
        max_latency: Optional[float] = None,
        max_memory: Optional[float] = None,
        target_fps: Optional[float] = None,
        device: Optional[str] = None
    ):
        self.min_accuracy = self._validate_threshold(
            min_accuracy, "min_accuracy", minimum=0.0, maximum=1.0
        )
        self.max_latency = self._validate_threshold(
            max_latency, "max_latency", minimum=0.0
        )
        self.max_memory = self._validate_threshold(
            max_memory, "max_memory", minimum=0.0
        )
        self.target_fps = self._validate_threshold(
            target_fps, "target_fps", minimum=0.0
        )
        self.device = device
        self.state = {}

    # ---------------------------------------------------------
    # VALIDATION HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _validate_threshold(
        value: Optional[float],
        name: str,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None
    ) -> Optional[float]:

        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if value != value:  # NaN
            raise ValueError(f"{name} cannot be NaN")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be <= {maximum}")
        return float(value)

    @staticmethod
    def _validate_metric(
        value: Any,
        name: str,
        minimum: Optional[float] = None
    ) -> Optional[float]:

        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if value != value:
            raise ValueError(f"{name} cannot be NaN")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return float(value)

    # ---------------------------------------------------------
    # ANALYZE
    # ---------------------------------------------------------

    def analyze(
        self,
        accuracy: Optional[float] = None,
        latency: Optional[float] = None,
        memory: Optional[float] = None,
        fps: Optional[float] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze the current model/system state."""

        accuracy = self._validate_metric(accuracy, "accuracy", minimum=0.0)
        latency = self._validate_metric(latency, "latency", minimum=0.0)
        memory = self._validate_metric(memory, "memory", minimum=0.0)
        fps = self._validate_metric(fps, "fps", minimum=0.0)

        constraints = self.validate_constraints(
            accuracy=accuracy, latency=latency, memory=memory, fps=fps
        )

        # violations are ordered by priority: accuracy, latency, memory, fps
        violations = constraints["violations"]
        state = f"{violations[0]}_violation" if violations else "valid"

        self.state = {
            "state": state,
            "accuracy": accuracy,
            "latency": latency,
            "memory": memory,
            "fps": fps,
            "device": device if device is not None else self.device,
            "constraints": constraints
        }
        return self.state

    # ---------------------------------------------------------
    # RECOMMEND
    # ---------------------------------------------------------

    def recommend(
        self,
        accuracy: Optional[float] = None,
        latency: Optional[float] = None,
        memory: Optional[float] = None,
        fps: Optional[float] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recommend the next optimization action.
        Only recommends; does not perform the action.
        """

        state = self.analyze(
            accuracy=accuracy, latency=latency,
            memory=memory, fps=fps, device=device
        )
        violations = state["constraints"]["violations"]

        # Accuracy has priority: never suggest a faster configuration
        # if minimum accuracy is already violated.
        if "accuracy" in violations:
            return {
                "action": "preserve_accuracy",
                "reason": "accuracy_below_minimum",
                "state": state
            }

        if "latency" in violations:
            return {
                "action": "reduce_latency",
                "reason": "latency_above_target",
                "suggestions": ["reduce_resolution", "fp16", "int8"],
                "state": state
            }

        if "memory" in violations:
            return {
                "action": "reduce_memory",
                "reason": "memory_above_limit",
                "suggestions": [
                    "reduce_model_size", "reduce_resolution", "fp16", "int8"
                ],
                "state": state
            }

        if "fps" in violations:
            return {
                "action": "increase_fps",
                "reason": "fps_below_target",
                "suggestions": [
                    "reduce_resolution", "reduce_inference_frequency",
                    "fp16", "int8"
                ],
                "state": state
            }

        return {
            "action": "keep_configuration",
            "reason": "all_constraints_satisfied",
            "state": state
        }

    # ---------------------------------------------------------
    # VALIDATE CONSTRAINTS
    # ---------------------------------------------------------

    def validate_constraints(
        self,
        accuracy: Optional[float] = None,
        latency: Optional[float] = None,
        memory: Optional[float] = None,
        fps: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate metrics against configured limits.

            accuracy >= minimum
            latency  <= maximum
            memory   <= maximum
            FPS      >= target
        """

        checks = {}
        violations = []

        # (name, limit, value, is_minimum_limit)
        for name, limit, value, is_min in (
            ("accuracy", self.min_accuracy, accuracy, True),
            ("latency", self.max_latency, latency, False),
            ("memory", self.max_memory, memory, False),
            ("fps", self.target_fps, fps, True),
        ):
            if limit is None:
                continue

            if value is None:
                checks[name] = {"valid": False, "reason": "missing_metric"}
                violations.append(name)
                continue

            valid = value >= limit if is_min else value <= limit
            checks[name] = {"value": value, "limit": limit, "valid": valid}
            if not valid:
                violations.append(name)

        return {
            "valid": not violations,
            "violations": violations,
            "checks": checks
        }

    # ---------------------------------------------------------
    # SELECT BEST
    # ---------------------------------------------------------

    def select_best(
        self,
        configurations: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best valid configuration.

        A configuration violating minimum accuracy is never selected
        merely because it is faster.

        Expected format:
        {"name": "model_fp16", "accuracy": 0.94,
         "latency": 25, "memory": 800, "fps": 40}
        """

        if configurations is None:
            raise ValueError("configurations cannot be None")
        if not isinstance(configurations, list):
            raise TypeError("configurations must be a list")
        if not configurations:
            return None

        inf = float("inf")
        best = None
        best_key = None
        best_result = None

        # Single pass: keep only the current best (no list of copies).
        for config in configurations:
            if not isinstance(config, dict):
                continue

            result = self.validate_constraints(
                accuracy=config.get("accuracy"),
                latency=config.get("latency"),
                memory=config.get("memory"),
                fps=config.get("fps")
            )
            if not result["valid"]:
                continue

            key = (
                config.get("latency", inf),
                -config.get("fps", 0),
                config.get("memory", inf),
                -config.get("accuracy", 0)
            )
            if best is None or key < best_key:  # strict: first wins ties
                best, best_key, best_result = config, key, result

        if best is None:
            return None

        candidate = dict(best)
        candidate["_validation"] = best_result
        return candidate