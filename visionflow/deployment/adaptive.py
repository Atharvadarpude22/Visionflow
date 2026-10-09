from typing import Any, Dict, List, Optional


class AdaptiveController:
    """
    Controls runtime adaptation for VisionFlow.

    The controller NEVER creates an unvalidated configuration.

    It only selects configurations that have already been validated
    and benchmarked by ModelOptimizer.

    Runtime flow:
        Monitor -> Detect degradation -> Select validated configuration
        -> Adapt -> Validate -> Keep / Restore
    """

    def __init__(
        self,
        configurations: Optional[List[Dict[str, Any]]] = None,
        target_fps: Optional[float] = None,
        max_latency: Optional[float] = None,
        max_memory: Optional[float] = None,
        min_accuracy: Optional[float] = None,
    ):
        self.configurations = configurations or []

        self.target_fps = target_fps
        self.max_latency = max_latency
        self.max_memory = max_memory
        self.min_accuracy = min_accuracy

        # Current runtime configuration
        self.current_configuration = None

        # Previous valid configuration
        self.previous_configuration = None

        # Latest runtime state
        self.state = {}

        # Adaptation history
        self.history = []

        self.status = "initialized"

    # =========================================================
    # CONFIGURATION MANAGEMENT
    # =========================================================

    def set_configurations(self, configurations):
        """
        Register previously validated configurations.

        Example:
            [{"name": "512_fp16", "resolution": 512, "precision": "FP16",
              "fps": 30, "latency": 33, "memory": 1200, "accuracy": 0.94}]
        """

        if configurations is None:
            raise ValueError("Configurations cannot be None")

        if not isinstance(configurations, list):
            raise TypeError("Configurations must be a list")

        self.configurations = configurations

        return self

    def set_configuration(self, configuration):
        """Set the initial validated runtime configuration."""

        if configuration is None:
            raise ValueError("Configuration cannot be None")

        if not self._is_valid_configuration(configuration):
            raise ValueError(
                "Configuration does not satisfy the required constraints"
            )

        self.current_configuration = configuration
        self.status = "running"

        return self

    # =========================================================
    # MONITOR
    # =========================================================

    def monitor_state(self, metrics: Optional[Dict[str, Any]] = None):
        """
        Read current runtime state.

        Metrics can include fps, latency, memory, workload, accuracy;
        additional metrics are allowed.
        """

        if metrics is None:
            metrics = {}

        if not isinstance(metrics, dict):
            raise TypeError("Runtime metrics must be a dictionary")

        self.state = dict(metrics)

        return self.state

    # =========================================================
    # PERFORMANCE DEGRADATION
    # =========================================================

    def detect_degradation(self, state=None):
        """Determine whether runtime performance has degraded."""

        if state is None:
            state = self.state

        if not state:
            return {
                "degraded": False,
                "reasons": []
            }

        reasons = []

        fps = state.get("fps")
        latency = state.get("latency")
        memory = state.get("memory")

        if (
            self.target_fps is not None
            and fps is not None
            and fps < self.target_fps
        ):
            reasons.append("fps_below_target")

        if (
            self.max_latency is not None
            and latency is not None
            and latency > self.max_latency
        ):
            reasons.append("latency_above_target")

        if (
            self.max_memory is not None
            and memory is not None
            and memory > self.max_memory
        ):
            reasons.append("memory_above_limit")

        return {
            "degraded": bool(reasons),
            "reasons": reasons
        }

    # =========================================================
    # CONFIGURATION VALIDATION
    # =========================================================

    def validate_constraints(self, configuration):
        """
        Check whether a configuration satisfies user constraints.

        Accuracy must NEVER be sacrificed simply to obtain
        better FPS or lower latency.
        """

        if configuration is None:
            return False

        get = configuration.get

        accuracy = get("accuracy")
        latency = get("latency")
        memory = get("memory")
        fps = get("fps")

        if (
            self.min_accuracy is not None
            and accuracy is not None
            and accuracy < self.min_accuracy
        ):
            return False

        if (
            self.max_latency is not None
            and latency is not None
            and latency > self.max_latency
        ):
            return False

        if (
            self.max_memory is not None
            and memory is not None
            and memory > self.max_memory
        ):
            return False

        if (
            self.target_fps is not None
            and fps is not None
            and fps < self.target_fps
        ):
            return False

        return True

    # =========================================================
    # VALIDATED CONFIGURATION CHECK
    # =========================================================

    def _is_valid_configuration(self, configuration):
        """
        Make sure the configuration is one of the configurations
        previously supplied by ModelOptimizer.

        AdaptiveController must NOT invent configurations.
        """

        if not self.validate_constraints(configuration):
            return False

        if not self.configurations:
            return True

        return configuration in self.configurations

    # =========================================================
    # SELECT CONFIGURATION
    # =========================================================

    def select_configuration(self, reason=None):
        """
        Select the next validated configuration.

        Configurations that violate constraints are ignored.
        """

        current = self.current_configuration

        has_valid = False
        best = None
        best_key = None

        # Single pass: no intermediate "valid" / "alternatives" lists
        # and no full sort just to read the top element.
        for config in self.configurations:

            if not self.validate_constraints(config):
                continue

            has_valid = True

            if config == current:
                continue

            # Prefer higher FPS and lower latency.
            # Accuracy remains protected by validate_constraints().
            key = (
                config.get("fps", 0),
                -config.get("latency", float("inf"))
            )

            # strict ">" keeps the first of equal keys, same as a
            # stable descending sort
            if best is None or key > best_key:
                best = config
                best_key = key

        if not has_valid:
            return None

        if best is None:
            return current

        return best

    # =========================================================
    # ADAPTATION
    # =========================================================

    def _adapt(self, action, configuration, reason):
        """Switch to a validated configuration and record it."""

        if configuration is None:
            configuration = self.select_configuration(reason=reason)

        if configuration is None:
            return self.current_configuration

        if not self._is_valid_configuration(configuration):
            return self.current_configuration

        self.previous_configuration = self.current_configuration
        self.current_configuration = configuration

        self.history.append({
            "action": action,
            "configuration": configuration
        })

        return configuration

    def adapt_resolution(self, configuration=None):
        """
        Switch to a previously validated resolution.

        No new resolution is generated here.
        """

        return self._adapt("adapt_resolution", configuration, "resolution")

    def adapt_frequency(self, configuration=None):
        """
        Adapt inference frequency using a validated configuration.

            skip=0 -> every frame
            skip=1 -> every second frame
            skip=2 -> every third frame
        """

        return self._adapt("adapt_frequency", configuration, "frequency")

    # =========================================================
    # VALIDATE CURRENT CONFIGURATION
    # =========================================================

    def validate(self, configuration=None):
        """Validate the currently active configuration."""

        if configuration is None:
            configuration = self.current_configuration

        if configuration is None:
            return False

        return self._is_valid_configuration(configuration)

    # =========================================================
    # RESTORE
    # =========================================================

    def restore_configuration(self):
        """
        Restore the previous valid configuration.

        Used when an adaptation produces worse performance.
        """

        if self.previous_configuration is None:
            return self.current_configuration

        if not self._is_valid_configuration(self.previous_configuration):
            return self.current_configuration

        current = self.current_configuration

        self.current_configuration = self.previous_configuration
        self.previous_configuration = current

        self.history.append({
            "action": "restore_configuration",
            "configuration": self.current_configuration
        })

        self.status = "restored"

        return self.current_configuration

    # =========================================================
    # CHECK ADAPTATION RESULT
    # =========================================================

    def _adaptation_improved(self, current_state):
        """Determine whether the new runtime state is acceptable."""

        if self.target_fps is not None:
            fps = current_state.get("fps")

            if fps is not None and fps < self.target_fps:
                return False

        if self.max_latency is not None:
            latency = current_state.get("latency")

            if latency is not None and latency > self.max_latency:
                return False

        if self.max_memory is not None:
            memory = current_state.get("memory")

            if memory is not None and memory > self.max_memory:
                return False

        return True

    # =========================================================
    # MAIN UPDATE
    # =========================================================

    def update(self, metrics=None):
        """
        Main runtime adaptation method.

        monitor -> detect -> select -> adapt -> validate
        """

        # 1. Monitor
        state = self.monitor_state(metrics)

        # 2. Detect degradation
        degradation = self.detect_degradation(state)

        if not degradation["degraded"]:

            self.status = "stable"

            return {
                "status": "stable",
                "degraded": False,
                "configuration": self.current_configuration,
                "reasons": []
            }

        reasons = degradation["reasons"]

        # 3. Select validated configuration
        candidate = self.select_configuration(reason=reasons)

        if candidate is None:

            self.status = "no_valid_configuration"

            return {
                "status": "no_valid_configuration",
                "degraded": True,
                "configuration": self.current_configuration,
                "reasons": reasons
            }

        # 4. Store current configuration
        old_configuration = self.current_configuration

        self.previous_configuration = old_configuration

        # 5. Adapt
        self.current_configuration = candidate

        # 6. Validate selected configuration
        if not self.validate(candidate):

            self.current_configuration = old_configuration
            self.status = "adaptation_rejected"

            return {
                "status": "adaptation_rejected",
                "configuration": old_configuration,
                "reasons": reasons
            }

        # 7. Record adaptation
        self.history.append({
            "action": "adapt",
            "from": old_configuration,
            "to": candidate,
            "reasons": reasons
        })

        self.status = "adapted"

        return {
            "status": "adapted",
            "degraded": True,
            "configuration": candidate,
            "previous_configuration": old_configuration,
            "reasons": reasons
        }

    # =========================================================
    # INFORMATION
    # =========================================================

    def current(self):
        """Return the currently active configuration."""

        return self.current_configuration

    def get_history(self):
        """Return runtime adaptation history."""

        return list(self.history)

    def summary(self):
        """Return controller state."""

        return {
            "status": self.status,
            "current_configuration": self.current_configuration,
            "previous_configuration": self.previous_configuration,
            "runtime_state": self.state,
            "available_configurations": len(self.configurations),
            "adaptations": len(self.history)
        }