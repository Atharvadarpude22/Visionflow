# visionflow/monitoring/monitor.py

from __future__ import annotations

import time
import platform
import statistics
from pathlib import Path
from typing import Optional, Any


class Monitor:
    """
    VisionFlow runtime monitoring.

    Collects:
        - inference latency
        - FPS
        - memory usage
        - temperature
        - runtime errors

    Designed for:
        - desktop
        - GPU systems
        - edge devices
        - embedded devices
        - Raspberry Pi
        - NVIDIA Jetson
    """

    def __init__(
        self,
        device: str = "auto",
        config: Optional[dict] = None
    ):
        self.device = device
        self.config = config or {}

        self._running = False
        self._start_time = None

        # -----------------------------------------------------
        # Runtime metrics
        # -----------------------------------------------------

        self.metrics = {
            "latency": [],
            "fps": [],
            "memory": [],
            "temperature": [],
            "errors": []
        }

        self._last_frame_time = None

    # =========================================================
    # START
    # =========================================================

    def start(self):
        """
        Starts runtime monitoring.
        """

        self._running = True
        self._start_time = time.perf_counter()

        self._last_frame_time = None

        return {
            "status": "MONITORING_STARTED",
            "device": self.device,
            "timestamp": time.time()
        }

    # =========================================================
    # RECORD LATENCY
    # =========================================================

    def record_latency(
        self,
        latency: float,
        unit: str = "seconds"
    ):
        """
        Record inference latency.

        Parameters
        ----------
        latency:
            Latency value.

        unit:
            "seconds" or "milliseconds"
        """

        if unit.lower() in {
            "ms",
            "millisecond",
            "milliseconds"
        }:
            latency_seconds = (
                float(latency) / 1000.0
            )

        else:
            latency_seconds = float(
                latency
            )

        self.metrics[
            "latency"
        ].append(
            latency_seconds
        )

        return latency_seconds

    # =========================================================
    # RECORD FPS
    # =========================================================

    def record_fps(
        self,
        fps: Optional[float] = None
    ):
        """
        Record FPS.

        If FPS is not supplied, it is calculated using
        the time between consecutive calls.
        """

        if fps is None:

            now = time.perf_counter()

            if self._last_frame_time is None:

                self._last_frame_time = now

                return None

            elapsed = (
                now -
                self._last_frame_time
            )

            self._last_frame_time = now

            if elapsed <= 0:
                return None

            fps = 1.0 / elapsed

        fps = float(fps)

        self.metrics[
            "fps"
        ].append(
            fps
        )

        return fps

    # =========================================================
    # RECORD MEMORY
    # =========================================================

    def record_memory(
        self,
        memory_mb: Optional[float] = None
    ):
        """
        Record memory usage.

        If memory_mb is not provided, process RAM usage
        is automatically collected when psutil is available.
        """

        if memory_mb is None:

            try:

                import psutil

                process = (
                    psutil.Process()
                )

                memory_mb = (
                    process
                    .memory_info()
                    .rss
                    / (
                        1024 * 1024
                    )
                )

            except Exception:

                return None

        memory_mb = float(
            memory_mb
        )

        self.metrics[
            "memory"
        ].append(
            memory_mb
        )

        return memory_mb

    # =========================================================
    # RECORD TEMPERATURE
    # =========================================================

    def record_temperature(
        self,
        temperature: Optional[float] = None
    ):
        """
        Record device temperature.

        If temperature is not provided, VisionFlow attempts
        to read available system sensors.

        Temperature is not available on every platform.
        """

        if temperature is None:

            temperature = (
                self._read_temperature()
            )

        if temperature is None:
            return None

        temperature = float(
            temperature
        )

        self.metrics[
            "temperature"
        ].append(
            temperature
        )

        return temperature

    # =========================================================
    # TEMPERATURE READER
    # =========================================================

    def _read_temperature(self):
        """
        Attempts to read temperature from available
        system sensors.

        Returns:
            temperature in Celsius
            or None when unsupported.
        """

        # -----------------------------------------------------
        # Linux / Raspberry Pi / Jetson
        # -----------------------------------------------------

        if platform.system() == "Linux":

            # ---------------------------------------------
            # Try psutil
            # ---------------------------------------------

            try:

                import psutil

                sensors = (
                    psutil.sensors_temperatures()
                )

                if sensors:

                    values = []

                    for entries in (
                        sensors.values()
                    ):

                        for entry in entries:

                            current = getattr(
                                entry,
                                "current",
                                None
                            )

                            if current is not None:
                                values.append(
                                    float(current)
                                )

                    if values:

                        return max(values)

            except Exception:
                pass

            # ---------------------------------------------
            # Raspberry Pi / Linux thermal zone
            # ---------------------------------------------

            thermal_path = Path(
                "/sys/class/thermal"
            )

            if thermal_path.exists():

                temperatures = []

                for zone in thermal_path.glob(
                    "thermal_zone*/temp"
                ):

                    try:

                        value = float(
                            zone.read_text(
                                encoding="utf-8"
                            ).strip()
                        )

                        # Linux thermal values are
                        # commonly represented in millidegrees.
                        if value > 1000:
                            value /= 1000.0

                        temperatures.append(
                            value
                        )

                    except Exception:
                        continue

                if temperatures:

                    return max(
                        temperatures
                    )

        # -----------------------------------------------------
        # NVIDIA GPU
        # -----------------------------------------------------

        try:

            import torch

            if torch.cuda.is_available():

                # NVML through pynvml
                try:

                    import pynvml

                    pynvml.nvmlInit()

                    device_count = (
                        pynvml.nvmlDeviceGetCount()
                    )

                    temperatures = []

                    for index in range(
                        device_count
                    ):

                        handle = (
                            pynvml
                            .nvmlDeviceGetHandleByIndex(
                                index
                            )
                        )

                        temp = (
                            pynvml
                            .nvmlDeviceGetTemperature(
                                handle,
                                pynvml.NVML_TEMPERATURE_GPU
                            )
                        )

                        temperatures.append(
                            float(temp)
                        )

                    if temperatures:

                        return max(
                            temperatures
                        )

                except Exception:
                    pass

        except Exception:
            pass

        return None

    # =========================================================
    # RECORD ERRORS
    # =========================================================

    def record_errors(
        self,
        error: Any
    ):
        """
        Records a runtime error.

        Accepts:
            Exception
            string
            dictionary
        """

        if isinstance(
            error,
            Exception
        ):

            error_data = {
                "type":
                    type(error).__name__,

                "message":
                    str(error),

                "timestamp":
                    time.time()
            }

        elif isinstance(
            error,
            dict
        ):

            error_data = dict(
                error
            )

            error_data.setdefault(
                "timestamp",
                time.time()
            )

        else:

            error_data = {
                "type": "RuntimeError",

                "message": str(error),

                "timestamp": time.time()
            }

        self.metrics[
            "errors"
        ].append(
            error_data
        )

        return error_data

    # =========================================================
    # GET METRICS
    # =========================================================

    def get_metrics(self):
        """
        Returns current raw runtime metrics.
        """

        return {
            "latency": list(
                self.metrics[
                    "latency"
                ]
            ),

            "fps": list(
                self.metrics[
                    "fps"
                ]
            ),

            "memory": list(
                self.metrics[
                    "memory"
                ]
            ),

            "temperature": list(
                self.metrics[
                    "temperature"
                ]
            ),

            "errors": list(
                self.metrics[
                    "errors"
                ]
            )
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self):
        """
        Creates a runtime summary.
        """

        latency = self.metrics[
            "latency"
        ]

        fps = self.metrics[
            "fps"
        ]

        memory = self.metrics[
            "memory"
        ]

        temperature = self.metrics[
            "temperature"
        ]

        errors = self.metrics[
            "errors"
        ]

        # -----------------------------------------------------
        # Latency
        # -----------------------------------------------------

        latency_summary = None

        if latency:

            latency_summary = {

                "average_ms":
                    statistics.mean(
                        latency
                    ) * 1000,

                "minimum_ms":
                    min(latency) * 1000,

                "maximum_ms":
                    max(latency) * 1000
            }

        # -----------------------------------------------------
        # FPS
        # -----------------------------------------------------

        fps_summary = None

        if fps:

            fps_summary = {

                "average":
                    statistics.mean(
                        fps
                    ),

                "minimum":
                    min(fps),

                "maximum":
                    max(fps)
            }

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        memory_summary = None

        if memory:

            memory_summary = {

                "average_mb":
                    statistics.mean(
                        memory
                    ),

                "minimum_mb":
                    min(memory),

                "maximum_mb":
                    max(memory)
            }

        # -----------------------------------------------------
        # Temperature
        # -----------------------------------------------------

        temperature_summary = None

        if temperature:

            temperature_summary = {

                "average_celsius":
                    statistics.mean(
                        temperature
                    ),

                "minimum_celsius":
                    min(temperature),

                "maximum_celsius":
                    max(temperature)
            }

        # -----------------------------------------------------
        # Uptime
        # -----------------------------------------------------

        uptime = None

        if self._start_time is not None:

            uptime = (
                time.perf_counter()
                -
                self._start_time
            )

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------

        return {

            "status":
                "RUNNING"
                if self._running
                else "STOPPED",

            "device":
                self.device,

            "platform":
                platform.platform(),

            "uptime_seconds":
                uptime,

            "latency":
                latency_summary,

            "fps":
                fps_summary,

            "memory":
                memory_summary,

            "temperature":
                temperature_summary,

            "error_count":
                len(errors),

            "errors":
                errors
        }

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):
        """
        Stops monitoring and returns final summary.
        """

        self._running = False

        return self.summary()

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):
        """
        Clears all collected metrics.
        """

        self.metrics = {
            "latency": [],
            "fps": [],
            "memory": [],
            "temperature": [],
            "errors": []
        }

        self._last_frame_time = None

        self._start_time = (
            time.perf_counter()
            if self._running
            else None
        )

        return {
            "status": "RESET"
        }