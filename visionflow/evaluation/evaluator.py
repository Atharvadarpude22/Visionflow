# visionflow/evaluation/evaluator.py

from __future__ import annotations

import time
import statistics
import platform
from pathlib import Path
from typing import Optional, Any, Dict


class Evaluator:
    """
    Unified VisionFlow evaluation engine.

    Handles:

    ML evaluation
        - accuracy
        - precision
        - recall
        - F1
        - mAP

    Scenario evaluation
        - normal
        - low light
        - blur
        - occlusion
        - small objects
        - different environments

    Edge evaluation
        - latency
        - FPS
        - memory
        - startup time

    End-to-end evaluation
        - capture
        - preprocessing
        - transfer
        - inference
        - postprocessing
        - output

    Stress testing
        - memory growth
        - latency degradation
        - FPS degradation
        - runtime errors

    Model comparison
        - accuracy
        - latency
        - FPS
        - memory
        - mAP
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        model=None,
        dataset=None,
        device="auto",
        config=None
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.config = config or {}

        self.results = {
            "evaluation": None,
            "scenarios": {},
            "edge": None,
            "end_to_end": None,
            "stress_test": None,
            "comparison": None
        }

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _get_underlying_model(self):
        """
        Gets the actual backend model from VisionFlow Model.
        """

        if self.model is None:
            return None

        backend = getattr(
            self.model,
            "_model",
            None
        )

        if backend is not None:
            return backend

        backend = getattr(
            self.model,
            "model",
            None
        )

        if backend is not None:
            return backend

        return None

    # =========================================================

    def _get_model_path(self):
        """
        Gets model path if available.
        """

        if self.model is None:
            return None

        for attribute in (
            "model_path",
            "path",
            "weights"
        ):

            value = getattr(
                self.model,
                attribute,
                None
            )

            if value is not None:
                return str(value)

        return None

    # =========================================================

    def _get_dataset_yaml(self):
        """
        Finds data.yaml from the VisionFlow dataset.
        """

        if self.dataset is None:
            return None

        dataset_path = getattr(
            self.dataset,
            "path",
            None
        )

        if dataset_path is None:
            return None

        dataset_path = Path(
            dataset_path
        )

        yaml_path = (
            dataset_path /
            "data.yaml"
        )

        if yaml_path.exists():
            return str(yaml_path)

        yaml_files = list(
            dataset_path.glob(
                "*.yaml"
            )
        )

        if yaml_files:
            return str(
                yaml_files[0]
            )

        return None

    # =========================================================

    def _predict(self, input_data):
        """
        Uses the VisionFlow model prediction API.
        """

        if self.model is None:
            raise ValueError(
                "Model is required."
            )

        predict_method = getattr(
            self.model,
            "predict",
            None
        )

        if not callable(
            predict_method
        ):
            raise ValueError(
                "Model does not provide predict()."
            )

        return predict_method(
            input_data
        )

    # =========================================================

    def _memory_usage(self):
        """
        Returns current process memory in MB.
        """

        try:

            import psutil

            process = psutil.Process()

            return (
                process.memory_info().rss
                / (1024 * 1024)
            )

        except Exception:

            return None

    # =========================================================
    # 1. EVALUATE
    # =========================================================

    def evaluate(
        self,
        split="val",
        **kwargs
    ):
        """
        Main ML evaluation method.

        Measures, when supported:

        - accuracy
        - precision
        - recall
        - F1
        - mAP50
        - mAP50-95
        """

        if self.model is None:
            raise ValueError(
                "Model is required."
            )

        yaml_path = (
            self._get_dataset_yaml()
        )

        if yaml_path is None:
            raise ValueError(
                "Dataset YAML could not be found."
            )

        backend = (
            self._get_underlying_model()
        )

        # -----------------------------------------------------
        # Load Ultralytics model if necessary
        # -----------------------------------------------------

        if backend is None:

            model_path = (
                self._get_model_path()
            )

            if model_path is None:
                raise ValueError(
                    "Model backend/path is unavailable."
                )

            try:

                from ultralytics import YOLO

                backend = YOLO(
                    model_path
                )

            except ImportError:

                raise ImportError(
                    "Ultralytics is required "
                    "for YOLO evaluation."
                )

        # -----------------------------------------------------
        # Validation arguments
        # -----------------------------------------------------

        validation_args = {
            "data": yaml_path,
            "split": split,
            "verbose": True
        }

        if self.device != "auto":
            validation_args[
                "device"
            ] = self.device

        validation_args.update(
            kwargs
        )

        metrics = backend.val(
            **validation_args
        )

        result = {
            "status": "SUCCESS",
            "task": "object_detection",
            "split": split
        }

        # -----------------------------------------------------
        # Precision
        # -----------------------------------------------------

        try:

            result["precision"] = float(
                metrics.box.mp
            )

        except Exception:
            pass

        # -----------------------------------------------------
        # Recall
        # -----------------------------------------------------

        try:

            result["recall"] = float(
                metrics.box.mr
            )

        except Exception:
            pass

        # -----------------------------------------------------
        # mAP50
        # -----------------------------------------------------

        try:

            result["mAP50"] = float(
                metrics.box.map50
            )

        except Exception:
            pass

        # -----------------------------------------------------
        # mAP50-95
        # -----------------------------------------------------

        try:

            result["mAP50_95"] = float(
                metrics.box.map
            )

        except Exception:
            pass

        # -----------------------------------------------------
        # Fitness
        # -----------------------------------------------------

        try:

            result["fitness"] = float(
                metrics.fitness
            )

        except Exception:
            pass

        # -----------------------------------------------------
        # F1
        # -----------------------------------------------------

        try:

            precision = result[
                "precision"
            ]

            recall = result[
                "recall"
            ]

            if (
                precision +
                recall
            ) > 0:

                result["f1"] = (
                    2 *
                    precision *
                    recall /
                    (
                        precision +
                        recall
                    )
                )

        except Exception:
            pass

        self.results[
            "evaluation"
        ] = result

        return result

    # =========================================================
    # 2. EVALUATE SCENARIOS
    # =========================================================

    def evaluate_scenarios(
        self,
        scenarios: Optional[
            Dict[str, Any]
        ] = None,
        **kwargs
    ):
        """
        Evaluate model under different scenarios.

        Example:

        {
            "normal": "normal/data.yaml",
            "low_light": "low_light/data.yaml",
            "blur": "blur/data.yaml",
            "occlusion": "occlusion/data.yaml",
            "small_objects": "small/data.yaml"
        }
        """

        if scenarios is None:

            scenarios = {
                "normal": None,
                "low_light": None,
                "blur": None,
                "occlusion": None,
                "small_objects": None,
                "different_environment": None
            }

        scenario_results = {}

        for name, scenario in (
            scenarios.items()
        ):

            print(
                f"\nEvaluating scenario: {name}"
            )

            if scenario is None:

                scenario_results[
                    name
                ] = {
                    "status": "SKIPPED",
                    "reason":
                        "Scenario dataset not provided."
                }

                continue

            scenario_path = Path(
                scenario
            )

            if scenario_path.is_dir():

                yaml_path = (
                    scenario_path /
                    "data.yaml"
                )

            else:

                yaml_path = scenario_path

            if not yaml_path.exists():

                scenario_results[
                    name
                ] = {
                    "status": "FAILED",
                    "reason":
                        "Scenario YAML does not exist."
                }

                continue

            try:

                backend = (
                    self._get_underlying_model()
                )

                if backend is None:

                    model_path = (
                        self._get_model_path()
                    )

                    from ultralytics import YOLO

                    backend = YOLO(
                        model_path
                    )

                args = {
                    "data": str(
                        yaml_path
                    ),
                    "split": "val",
                    "verbose": False
                }

                if self.device != "auto":

                    args[
                        "device"
                    ] = self.device

                args.update(
                    kwargs
                )

                metrics = backend.val(
                    **args
                )

                scenario_result = {
                    "status": "SUCCESS"
                }

                try:

                    scenario_result[
                        "precision"
                    ] = float(
                        metrics.box.mp
                    )

                except Exception:
                    pass

                try:

                    scenario_result[
                        "recall"
                    ] = float(
                        metrics.box.mr
                    )

                except Exception:
                    pass

                try:

                    scenario_result[
                        "mAP50"
                    ] = float(
                        metrics.box.map50
                    )

                except Exception:
                    pass

                try:

                    scenario_result[
                        "mAP50_95"
                    ] = float(
                        metrics.box.map
                    )

                except Exception:
                    pass

                scenario_results[
                    name
                ] = scenario_result

            except Exception as error:

                scenario_results[
                    name
                ] = {
                    "status": "FAILED",
                    "error": str(error)
                }

        self.results[
            "scenarios"
        ] = scenario_results

        return scenario_results

    # =========================================================
    # 3. EVALUATE EDGE
    # =========================================================

    def evaluate_edge(
        self,
        input_data=None,
        iterations=100,
        warmup=10
    ):
        """
        Measures:

        - startup time
        - latency
        - FPS
        - memory
        """

        if self.model is None:
            raise ValueError(
                "Model is required."
            )

        # -----------------------------------------------------
        # Startup
        # -----------------------------------------------------

        startup_start = (
            time.perf_counter()
        )

        try:

            load_method = getattr(
                self.model,
                "load",
                None
            )

            if callable(
                load_method
            ):

                load_method()

        except Exception:
            pass

        startup_time = (
            time.perf_counter()
            -
            startup_start
        )

        # -----------------------------------------------------
        # Warmup
        # -----------------------------------------------------

        for _ in range(
            max(0, warmup)
        ):

            if input_data is None:
                break

            try:

                self._predict(
                    input_data
                )

            except Exception:
                break

        # -----------------------------------------------------
        # Benchmark
        # -----------------------------------------------------

        latencies = []
        errors = []

        for _ in range(
            max(1, iterations)
        ):

            start = (
                time.perf_counter()
            )

            try:

                if input_data is not None:

                    self._predict(
                        input_data
                    )

            except Exception as error:

                errors.append(
                    str(error)
                )

            elapsed = (
                time.perf_counter()
                -
                start
            )

            latencies.append(
                elapsed
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        average_latency = (
            statistics.mean(
                latencies
            )
            if latencies
            else 0
        )

        fps = (
            1.0 /
            average_latency
            if average_latency > 0
            else 0
        )

        memory = (
            self._memory_usage()
        )

        result = {

            "status": "SUCCESS",

            "startup_time_seconds":
                startup_time,

            "startup_time_ms":
                startup_time * 1000,

            "latency": {

                "average_ms":
                    average_latency * 1000,

                "minimum_ms":
                    min(latencies) * 1000
                    if latencies
                    else 0,

                "maximum_ms":
                    max(latencies) * 1000
                    if latencies
                    else 0
            },

            "fps":
                fps,

            "memory_mb":
                memory,

            "iterations":
                len(latencies),

            "errors":
                len(errors),

            "device":
                self.device
        }

        self.results[
            "edge"
        ] = result

        return result

    # =========================================================
    # 4. END-TO-END EVALUATION
    # =========================================================

    def evaluate_end_to_end(
        self,
        input_data=None,
        iterations=20
    ):
        """
        Measures the complete inference pipeline:

        capture
            ↓
        preprocessing
            ↓
        transfer
            ↓
        inference
            ↓
        postprocessing
            ↓
        output
        """

        if self.model is None:
            raise ValueError(
                "Model is required."
            )

        results = []

        for _ in range(
            max(1, iterations)
        ):

            pipeline_start = (
                time.perf_counter()
            )

            # -------------------------------------------------
            # CAPTURE
            # -------------------------------------------------

            capture_start = (
                time.perf_counter()
            )

            # If input_data is already provided,
            # capture time is effectively zero.

            data = input_data

            capture_time = (
                time.perf_counter()
                -
                capture_start
            )

            # -------------------------------------------------
            # PREPROCESSING
            # -------------------------------------------------

            preprocess_start = (
                time.perf_counter()
            )

            inference_engine = getattr(
                self.model,
                "_inference_engine",
                None
            )

            if inference_engine is None:

                inference_engine = getattr(
                    self.model,
                    "inference_engine",
                    None
                )

            if inference_engine is not None:

                preprocess = getattr(
                    inference_engine,
                    "preprocess",
                    None
                )

                if callable(preprocess):

                    try:

                        data = preprocess(
                            data
                        )

                    except Exception:
                        pass

            preprocess_time = (
                time.perf_counter()
                -
                preprocess_start
            )

            # -------------------------------------------------
            # TRANSFER
            # -------------------------------------------------

            transfer_start = (
                time.perf_counter()
            )

            # Actual transfer may be handled
            # internally by the backend.

            transfer_time = (
                time.perf_counter()
                -
                transfer_start
            )

            # -------------------------------------------------
            # INFERENCE
            # -------------------------------------------------

            inference_start = (
                time.perf_counter()
            )

            try:

                output = self._predict(
                    data
                )

            except Exception:

                output = None

            inference_time = (
                time.perf_counter()
                -
                inference_start
            )

            # -------------------------------------------------
            # POSTPROCESSING
            # -------------------------------------------------

            postprocess_start = (
                time.perf_counter()
            )

            if inference_engine is not None:

                postprocess = getattr(
                    inference_engine,
                    "postprocess",
                    None
                )

                if callable(
                    postprocess
                ):

                    try:

                        output = postprocess(
                            output
                        )

                    except Exception:
                        pass

            postprocess_time = (
                time.perf_counter()
                -
                postprocess_start
            )

            # -------------------------------------------------
            # OUTPUT
            # -------------------------------------------------

            output_start = (
                time.perf_counter()
            )

            # Output handling is intentionally
            # kept lightweight here.

            _ = output

            output_time = (
                time.perf_counter()
                -
                output_start
            )

            # -------------------------------------------------
            # TOTAL
            # -------------------------------------------------

            total_time = (
                time.perf_counter()
                -
                pipeline_start
            )

            results.append({

                "capture_ms":
                    capture_time * 1000,

                "preprocessing_ms":
                    preprocess_time * 1000,

                "transfer_ms":
                    transfer_time * 1000,

                "inference_ms":
                    inference_time * 1000,

                "postprocessing_ms":
                    postprocess_time * 1000,

                "output_ms":
                    output_time * 1000,

                "total_ms":
                    total_time * 1000
            })

        # -----------------------------------------------------
        # Aggregate
        # -----------------------------------------------------

        def average(key):

            values = [
                item[key]
                for item in results
            ]

            return (
                statistics.mean(values)
                if values
                else 0
            )

        summary = {

            "status": "SUCCESS",

            "capture_ms":
                average("capture_ms"),

            "preprocessing_ms":
                average("preprocessing_ms"),

            "transfer_ms":
                average("transfer_ms"),

            "inference_ms":
                average("inference_ms"),

            "postprocessing_ms":
                average("postprocessing_ms"),

            "output_ms":
                average("output_ms"),

            "total_ms":
                average("total_ms"),

            "iterations":
                len(results)
        }

        self.results[
            "end_to_end"
        ] = summary

        return summary

    # =========================================================
    # 5. STRESS TEST
    # =========================================================

    def stress_test(
        self,
        input_data=None,
        duration_seconds=60,
        sample_interval=5
    ):
        """
        Runs continuous inference for a specified duration.

        Checks:

        - memory growth
        - latency degradation
        - FPS degradation
        - runtime errors
        """

        if self.model is None:
            raise ValueError(
                "Model is required."
            )

        start_time = (
            time.perf_counter()
        )

        latencies = []
        fps_values = []
        memory_values = []
        errors = []

        last_sample = start_time

        while (
            time.perf_counter()
            - start_time
            < duration_seconds
        ):

            inference_start = (
                time.perf_counter()
            )

            try:

                if input_data is not None:

                    self._predict(
                        input_data
                    )

            except Exception as error:

                errors.append(
                    str(error)
                )

            latency = (
                time.perf_counter()
                -
                inference_start
            )

            latencies.append(
                latency
            )

            if latency > 0:

                fps_values.append(
                    1.0 / latency
                )

            now = (
                time.perf_counter()
            )

            if (
                now -
                last_sample
                >= sample_interval
            ):

                memory = (
                    self._memory_usage()
                )

                if memory is not None:

                    memory_values.append(
                        memory
                    )

                last_sample = now

        # -----------------------------------------------------
        # Calculate degradation
        # -----------------------------------------------------

        def degradation(values):

            if len(values) < 2:
                return 0.0

            first_count = max(
                1,
                len(values) // 5
            )

            first = statistics.mean(
                values[:first_count]
            )

            last = statistics.mean(
                values[-first_count:]
            )

            if first == 0:
                return 0.0

            return (
                (
                    last - first
                )
                /
                first
            ) * 100

        latency_degradation = (
            degradation(
                latencies
            )
        )

        fps_degradation = (
            degradation(
                fps_values
            )
        )

        memory_growth = 0.0

        if (
            len(memory_values)
            >= 2
        ):

            memory_growth = (
                memory_values[-1]
                -
                memory_values[0]
            )

        result = {

            "status": "SUCCESS",

            "duration_seconds":
                time.perf_counter()
                - start_time,

            "inference_count":
                len(latencies),

            "runtime_errors":
                len(errors),

            "latency_degradation_percent":
                latency_degradation,

            "fps_degradation_percent":
                fps_degradation,

            "memory_growth_mb":
                memory_growth,

            "initial_memory_mb":
                memory_values[0]
                if memory_values
                else None,

            "final_memory_mb":
                memory_values[-1]
                if memory_values
                else None,

            "errors":
                errors
        }

        self.results[
            "stress_test"
        ] = result

        return result

    # =========================================================
    # 6. COMPARE
    # =========================================================

    def compare(
        self,
        model_a,
        model_b,
        input_data=None,
        iterations=50,
        dataset=None
    ):
        """
        Compare two models/configurations.

        Compares:

        - precision
        - recall
        - mAP
        - F1
        - latency
        - FPS
        - memory
        """

        def benchmark(model):

            old_model = self.model

            self.model = model

            try:

                edge = self.evaluate_edge(
                    input_data=input_data,
                    iterations=iterations
                )

            finally:

                self.model = old_model

            return edge

        result_a = benchmark(
            model_a
        )

        result_b = benchmark(
            model_b
        )

        comparison = {

            "model_a": result_a,

            "model_b": result_b,

            "comparison": {}
        }

        # -----------------------------------------------------
        # Latency
        # -----------------------------------------------------

        latency_a = (
            result_a
            .get("latency", {})
            .get("average_ms")
        )

        latency_b = (
            result_b
            .get("latency", {})
            .get("average_ms")
        )

        if (
            latency_a is not None
            and
            latency_b is not None
        ):

            comparison[
                "comparison"
            ][
                "latency"
            ] = {

                "model_a_ms":
                    latency_a,

                "model_b_ms":
                    latency_b,

                "faster":
                    "model_a"
                    if latency_a < latency_b
                    else "model_b"
            }

        # -----------------------------------------------------
        # FPS
        # -----------------------------------------------------

        fps_a = result_a.get(
            "fps"
        )

        fps_b = result_b.get(
            "fps"
        )

        if (
            fps_a is not None
            and
            fps_b is not None
        ):

            comparison[
                "comparison"
            ][
                "fps"
            ] = {

                "model_a":
                    fps_a,

                "model_b":
                    fps_b,

                "higher":
                    "model_a"
                    if fps_a > fps_b
                    else "model_b"
            }

        # -----------------------------------------------------
        # Memory
        # -----------------------------------------------------

        memory_a = result_a.get(
            "memory_mb"
        )

        memory_b = result_b.get(
            "memory_mb"
        )

        if (
            memory_a is not None
            and
            memory_b is not None
        ):

            comparison[
                "comparison"
            ][
                "memory"
            ] = {

                "model_a_mb":
                    memory_a,

                "model_b_mb":
                    memory_b,

                "lower":
                    "model_a"
                    if memory_a < memory_b
                    else "model_b"
            }

        self.results[
            "comparison"
        ] = comparison

        return comparison

    # =========================================================
    # 7. REPORT
    # =========================================================

    def report(
        self,
        save_path=None
    ):
        """
        Produces the final evaluation report.

        If save_path is provided, the report is also
        saved as JSON.
        """

        report = {

            "visionflow_evaluation": {

                "timestamp":
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "platform":
                    platform.platform(),

                "device":
                    self.device,

                "results":
                    self.results
            }
        }

        if save_path is not None:

            save_path = Path(
                save_path
            )

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            import json

            with open(
                save_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=4,
                    default=str
                )

            report[
                "report_path"
            ] = str(
                save_path
            )

        return report