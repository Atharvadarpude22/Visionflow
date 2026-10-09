from __future__ import annotations

import copy
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelOptimizer:
    """
    VisionFlow edge/deployment model optimization engine.

    Responsibilities
    ----------------
    - Profile target device
    - Analyze model
    - Generate candidate configurations
    - Quantize when supported
    - Optimize input resolution
    - Prune when supported
    - Optional knowledge distillation
    - Export model
    - Benchmark configurations
    - Select the best valid configuration

    This class intentionally combines these responsibilities instead of
    creating separate Quantizer, Pruner, ResolutionOptimizer, etc.
    """

    def __init__(
        self,
        model: Any = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.device = device
        self.config = config or {}

        self.device_profile = {}
        self.model_profile = {}
        self.variants = []
        self.benchmark_results = []
        self.best_configuration = None

        self.status = "created"

    # =========================================================
    # DEVICE PROFILING
    # =========================================================

    def profile_device(self) -> Dict[str, Any]:
        """
        Determine available CPU, GPU, RAM, accelerator,
        precision and runtime information.
        """

        profile = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "gpu": None,
            "gpu_available": False,
            "ram_gb": None,
            "accelerator": None,
            "supported_precision": ["FP32"],
            "runtime": [],
        }

        # RAM
        try:
            import psutil

            profile["ram_gb"] = round(
                psutil.virtual_memory().total / (1024 ** 3),
                2,
            )
        except ImportError:
            profile["ram_gb"] = None

        # PyTorch / CUDA
        try:
            import torch

            if torch.cuda.is_available():

                profile["gpu_available"] = True
                profile["gpu"] = torch.cuda.get_device_name(0)
                profile["accelerator"] = "CUDA"

                profile["supported_precision"] = ["FP32", "FP16"]

                profile["runtime"].append("CUDA")

        except (ImportError, RuntimeError):
            pass

        # TensorRT
        try:
            import tensorrt  # noqa: F401

            profile["runtime"].append("TensorRT")

        except ImportError:
            pass

        # ONNX Runtime
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()

            profile["runtime"].append("ONNX")

            if "CUDAExecutionProvider" in providers:
                profile["supported_precision"].append("FP16")

        except ImportError:
            pass

        self.device_profile = profile
        self.status = "device_profiled"

        return profile

    # =========================================================
    # MODEL ANALYSIS
    # =========================================================

    def analyze_model(self) -> Dict[str, Any]:
        """
        Analyze model characteristics: size, parameters, input
        resolution, supported operations when available.
        """

        profile = {
            "model_type": type(self.model).__name__
            if self.model is not None
            else None,
            "model_size_mb": None,
            "parameters": None,
            "input_resolution": None,
            "supported_operations": None,
        }

        if self.model is None:
            self.model_profile = profile
            return profile

        # Model path / file size
        possible_path = None

        if isinstance(self.model, (str, Path)):
            possible_path = Path(self.model)

        else:
            model_path = getattr(self.model, "model_path", None)

            if model_path:
                possible_path = Path(model_path)

        if possible_path is not None:

            # One stat call instead of exists() + stat()
            try:

                profile["model_size_mb"] = round(
                    possible_path.stat().st_size / (1024 ** 2),
                    3,
                )

            except FileNotFoundError:
                pass

        # Underlying model object
        underlying = getattr(self.model, "_model", self.model)

        # Parameter count. An Ultralytics-style `.model` takes
        # precedence, so the (full) first traversal is only done
        # when that one does not apply or fails.
        parameters = None
        counted = False

        try:

            if hasattr(underlying, "model"):

                torch_model = underlying.model

                if hasattr(torch_model, "parameters"):

                    parameters = sum(
                        p.numel() for p in torch_model.parameters()
                    )

                    counted = True

        except Exception:
            pass

        if not counted:

            try:

                if hasattr(underlying, "parameters"):

                    parameters = sum(
                        parameter.numel()
                        for parameter in underlying.parameters()
                    )

            except Exception:
                parameters = None

        profile["parameters"] = parameters

        # Input resolution
        if hasattr(self.model, "config"):

            profile["input_resolution"] = self.model.config.get(
                "input_resolution"
            )

        self.model_profile = profile
        self.status = "model_analyzed"

        return profile

    # =========================================================
    # VARIANT GENERATION
    # =========================================================

    def generate_variants(
        self,
        resolutions: Optional[List[int]] = None,
        include_fp16: bool = True,
        include_int8: bool = True,
        include_fp32: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate model configurations.

        Only generates FP16 when the device supports it.
        INT8 is included as a candidate only when configured.
        """

        if not self.device_profile:
            self.profile_device()

        if resolutions is None:

            resolutions = self.config.get(
                "resolutions",
                [640, 512, 416],
            )

        variants = []

        supported_precision = self.device_profile.get(
            "supported_precision",
            ["FP32"],
        )

        if include_fp32:

            variants.extend(
                {"precision": "FP32", "resolution": resolution}
                for resolution in resolutions
            )

        if include_fp16 and "FP16" in supported_precision:

            variants.extend(
                {"precision": "FP16", "resolution": resolution}
                for resolution in resolutions
            )

        # INT8 is an explicit optimization candidate; it should only
        # be applied if the backend supports it.
        if include_int8 and self.config.get("enable_int8", False):

            variants.extend(
                {"precision": "INT8", "resolution": resolution}
                for resolution in resolutions
            )

        self.variants = variants

        return variants

    # =========================================================
    # QUANTIZATION
    # =========================================================

    def quantize(
        self,
        precision: str = "INT8",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply supported quantization.

        This method intentionally does not silently perform unsafe
        quantization. The actual backend must be available.
        """

        precision = precision.upper()

        if precision not in {"FP16", "INT8"}:
            raise ValueError(
                "Supported quantization modes are FP16 and INT8"
            )

        if not self.device_profile:
            self.profile_device()

        if (
            precision == "FP16"
            and "FP16"
            not in self.device_profile.get("supported_precision", [])
        ):
            raise RuntimeError(
                "FP16 is not supported by the current device/backend"
            )

        if precision == "INT8":

            if not self.config.get("enable_int8", False):
                raise RuntimeError(
                    "INT8 optimization is disabled in configuration"
                )

        underlying = getattr(self.model, "_model", self.model)

        result = {
            "status": "configured",
            "precision": precision,
            "model": underlying,
        }

        # FP16 for PyTorch
        if precision == "FP16":

            try:

                if hasattr(underlying, "half"):

                    underlying.half()

                    result["status"] = "applied"

            except Exception as exc:

                result["status"] = "failed"
                result["error"] = str(exc)

        # INT8 requires calibration/backend-specific handling.
        elif precision == "INT8":

            result["status"] = "pending_backend"
            result["message"] = (
                "INT8 requires a supported quantization/export "
                "backend and calibration data."
            )

        self.status = "quantized"

        return result

    # =========================================================
    # RESOLUTION OPTIMIZATION
    # =========================================================

    def optimize_resolution(
        self,
        resolutions: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Test candidate input resolutions.

        Actual benchmarking is delegated to benchmark().
        """

        if resolutions is None:

            resolutions = self.config.get(
                "resolutions",
                [640, 512, 416],
            )

        return {
            "resolutions": [
                {"resolution": resolution, "status": "candidate"}
                for resolution in resolutions
            ]
        }

    # =========================================================
    # PRUNING
    # =========================================================

    def prune(
        self,
        amount: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply pruning when supported.

        Pruning is deliberately explicit because it can affect accuracy.
        """

        if not 0.0 < amount < 1.0:
            raise ValueError("Pruning amount must be between 0 and 1")

        underlying = getattr(self.model, "_model", self.model)

        if underlying is None:
            raise ValueError("Model is not available")

        # PyTorch pruning
        try:

            import torch.nn.utils.prune as prune_utils

            found = False

            # Pruned inline while walking the modules instead of
            # first collecting every (module, name) pair in a list.
            for module in underlying.modules():

                if not hasattr(module, "weight"):
                    continue

                found = True

                try:

                    prune_utils.l1_unstructured(
                        module,
                        name="weight",
                        amount=amount,
                    )

                except Exception:
                    continue

            if not found:

                return {
                    "status": "unsupported",
                    "reason": "No prunable parameters found",
                }

            self.status = "pruned"

            return {
                "status": "applied",
                "amount": amount,
            }

        except ImportError:

            return {
                "status": "unsupported",
                "reason": "PyTorch pruning backend unavailable",
            }

    # =========================================================
    # DISTILLATION
    # =========================================================

    def distill(
        self,
        teacher=None,
        student=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Optional knowledge distillation.

        It is never automatically executed because it requires
        additional training.
        """

        if not self.config.get("enable_distillation", False):
            return {
                "status": "skipped",
                "reason": "Knowledge distillation is disabled",
            }

        if teacher is None:
            raise ValueError("Teacher model is required")

        if student is None:
            raise ValueError("Student model is required")

        return {
            "status": "configured",
            "teacher": type(teacher).__name__,
            "student": type(student).__name__,
            "message": "Distillation requires the training backend.",
        }

    # =========================================================
    # EXPORT
    # =========================================================

    def export(
        self,
        format: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export the model to a supported format."""

        if self.model is None:
            raise ValueError("Model is not available")

        format = format.lower()

        if format not in {"onnx", "tensorrt", "torchscript"}:
            raise ValueError(f"Unsupported export format: {format}")

        if output_path is None:
            output_path = f"visionflow_model.{format}"

        underlying = getattr(self.model, "_model", self.model)

        # Use model's native export if available
        if hasattr(underlying, "export"):

            result = underlying.export(format=format, **kwargs)

            self.status = "exported"

            return {
                "status": "success",
                "format": format,
                "output": result,
            }

        # PyTorch ONNX export
        if format == "onnx" and hasattr(underlying, "state_dict"):

            return {
                "status": "requires_export_backend",
                "format": "onnx",
                "output_path": output_path,
                "message": (
                    "A representative input tensor is required "
                    "for generic PyTorch ONNX export."
                ),
            }

        return {
            "status": "unsupported",
            "format": format,
        }

    # =========================================================
    # BENCHMARK
    # =========================================================

    def benchmark(
        self,
        variants: Optional[List[Dict[str, Any]]] = None,
        benchmark_fn=None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Benchmark candidate configurations.

        benchmark_fn should return a dictionary containing any of:
        accuracy, latency, fps, memory, model_size
        """

        if variants is None:
            variants = self.variants

        if not variants:
            variants = self.generate_variants()

        results = []

        for variant in variants:

            start = time.perf_counter()

            try:

                if benchmark_fn is not None:

                    metrics = benchmark_fn(variant, self.model, **kwargs)

                else:

                    metrics = {
                        "accuracy": None,
                        "latency": None,
                        "fps": None,
                        "memory": None,
                        "model_size": None,
                    }

                results.append({
                    **variant,
                    **metrics,
                    "benchmark_time": time.perf_counter() - start,
                })

            except Exception as exc:

                results.append({
                    **variant,
                    "status": "failed",
                    "error": str(exc),
                })

        self.benchmark_results = results
        self.status = "benchmarked"

        return results

    # =========================================================
    # BEST CONFIGURATION
    # =========================================================

    def find_best_configuration(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        min_accuracy: Optional[float] = None,
        max_latency: Optional[float] = None,
        max_memory: Optional[float] = None,
        min_fps: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best VALID configuration.

        A configuration that violates user constraints is never selected.
        """

        if results is None:
            results = self.benchmark_results

        if not results:
            return None

        config = self.config

        if min_accuracy is None:
            min_accuracy = config.get("min_accuracy")

        if max_latency is None:
            max_latency = config.get("max_latency")

        if max_memory is None:
            max_memory = config.get("max_memory")

        if min_fps is None:
            min_fps = config.get("min_fps")

        inf = float("inf")

        best = None
        best_latency = None

        # Single pass: keep only the best valid result instead of
        # collecting every valid one and sorting.
        for result in results:

            if result.get("status") == "failed":
                continue

            accuracy = result.get("accuracy")
            latency = result.get("latency")
            memory = result.get("memory")
            fps = result.get("fps")

            if (
                min_accuracy is not None
                and accuracy is not None
                and accuracy < min_accuracy
            ):
                continue

            if (
                max_latency is not None
                and latency is not None
                and latency > max_latency
            ):
                continue

            if (
                max_memory is not None
                and memory is not None
                and memory > max_memory
            ):
                continue

            if (
                min_fps is not None
                and fps is not None
                and fps < min_fps
            ):
                continue

            # Prefer lower latency; strict "<" keeps the first of
            # equal values, same as a stable ascending sort.
            key = inf if latency is None else latency

            if best is None or key < best_latency:
                best = result
                best_latency = key

        self.best_configuration = best

        return best

    # =========================================================
    # COMPLETE OPTIMIZATION
    # =========================================================

    def optimize(
        self,
        benchmark_fn=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the complete model optimization workflow.

        No expensive optimization such as distillation is automatically
        executed unless explicitly enabled.
        """

        self.profile_device()

        self.analyze_model()

        self.generate_variants()

        benchmark_results = self.benchmark(benchmark_fn=benchmark_fn)

        best = self.find_best_configuration()

        self.status = "optimized"

        return {
            "status": self.status,
            "device": self.device_profile,
            "model": self.model_profile,
            "variants": self.variants,
            "benchmarks": benchmark_results,
            "best_configuration": best,
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> Dict[str, Any]:
        """Return optimizer state."""

        return {
            "status": self.status,
            "device_profiled": bool(self.device_profile),
            "model_analyzed": bool(self.model_profile),
            "variants": len(self.variants),
            "benchmarks": len(self.benchmark_results),
            "best_configuration": self.best_configuration,
        }