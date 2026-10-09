from pathlib import Path
from typing import Optional, Any, Dict


class Model:
    """
    Public VisionFlow Model API.

    Example
    -------

    model = vf.Model(
        "yolov8n.pt",
        auto_download=True
    )

    model.train(
        dataset=dataset,
        epochs=10
    )

    VisionFlow handles:

        Model existence
             ↓
        Download if required
             ↓
        Load
             ↓
        Trainer
             ↓
        Validation
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_download: bool = True
    ):

        self.model_path = (
            str(model_path)
            if model_path is not None
            else None
        )

        self.config = config or {}

        self.auto_download = auto_download

        # Actual backend model
        self._model = None

        # Internal components
        self._trainer = None
        self._evaluator = None
        self._optimizer = None
        self._inference_engine = None
        self._adaptive_controller = None

        # Results
        self.training_result = None
        self.evaluation_result = None
        self.optimization_result = None
        self.benchmark_result = None
        self.deployment_result = None
        self.export_result = None

        self.status = "created"

    # =========================================================
    # LOAD
    # =========================================================

    def load(
        self,
        model_path: Optional[str] = None,
        auto_download: Optional[bool] = None
    ):

        if model_path is not None:

            self.model_path = str(
                model_path
            )

        if auto_download is not None:

            self.auto_download = (
                auto_download
            )

        if self.model_path is None:

            raise ValueError(
                "Model path/name is required."
            )

        model_file = Path(
            self.model_path
        )

        # =====================================================
        # MODEL EXISTS
        # =====================================================

        if model_file.exists():

            print(
                f"Loading model:\n"
                f"{model_file}"
            )

            return self._load_ultralytics(
                str(model_file)
            )

        # =====================================================
        # MODEL DOES NOT EXIST
        # =====================================================

        if not self.auto_download:

            raise FileNotFoundError(
                f"Model does not exist:\n"
                f"{self.model_path}\n\n"
                "Set auto_download=True "
                "to obtain a pretrained model."
            )

        return self._download_and_load()

    # =========================================================
    # ULTRALYTICS LOADER
    # =========================================================

    def _load_ultralytics(
        self,
        model_path
    ):

        try:

            from ultralytics import YOLO

        except ImportError:

            raise ImportError(
                "Ultralytics is required.\n"
                "Install using:\n"
                "pip install ultralytics"
            )

        try:

            self._model = YOLO(
                model_path
            )

        except Exception as error:

            raise RuntimeError(
                f"Could not load model: "
                f"{error}"
            )

        self.model_path = str(
            model_path
        )

        self.status = "loaded"

        return self

    # =========================================================
    # AUTOMATIC MODEL DOWNLOAD
    # =========================================================

    def _download_and_load(self):

        """
        Let Ultralytics obtain the pretrained model.

        Example:

            YOLO("yolov8n.pt")

        If the file is not present locally,
        Ultralytics downloads the pretrained
        weights.
        """

        try:

            from ultralytics import YOLO

        except ImportError:

            raise ImportError(
                "Ultralytics is required for "
                "automatic model download.\n\n"
                "Install using:\n"
                "pip install ultralytics"
            )

        print(
            "\nModel not found locally."
        )

        print(
            f"Obtaining pretrained model:"
            f" {self.model_path}"
        )

        try:

            self._model = YOLO(
                self.model_path
            )

        except Exception as error:

            raise RuntimeError(
                "VisionFlow could not obtain "
                f"the model '{self.model_path}'.\n"
                f"Error: {error}"
            )

        self.status = "loaded"

        return self

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(
        self,
        data,
        **kwargs
    ):

        if self._model is None:

            self.load()

        if hasattr(
            self._model,
            "predict"
        ):

            return self._model.predict(
                data,
                **kwargs
            )

        raise RuntimeError(
            "Loaded model does not support "
            "prediction."
        )

    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        dataset=None,
        **kwargs
    ):

        if dataset is None:

            raise ValueError(
                "Dataset is required for training."
            )

        # -----------------------------------------------------
        # Automatically load/download model
        # -----------------------------------------------------

        if self._model is None:

            self.load()

        # -----------------------------------------------------
        # Trainer
        # -----------------------------------------------------

        from visionflow.training import Trainer

        self._trainer = Trainer(

            model=self,

            dataset=dataset,

            output_dir=kwargs.pop(
                "output_dir",
                "runs/visionflow"
            ),

            device=kwargs.pop(
                "device",
                "auto"
            ),

            edge_device=kwargs.pop(
                "edge_device",
                None
            ),

            config=kwargs.pop(
                "config",
                None
            )
        )

        self.training_result = (
            self._trainer.train(
                dataset=dataset,
                **kwargs
            )
        )

        self.status = "trained"

        return self.training_result

    # =========================================================
    # VALIDATE
    # =========================================================

    def validate(
        self,
        dataset=None,
        **kwargs
    ):

        if self._model is None:

            self.load()

        if dataset is None:

            raise ValueError(
                "Dataset is required."
            )

        if self._trainer is None:

            from visionflow.training import Trainer

            self._trainer = Trainer(

                model=self,

                dataset=dataset,

                device=kwargs.pop(
                    "device",
                    "auto"
                )
            )

        self.evaluation_result = (
            self._trainer.validate(
                dataset=dataset,
                **kwargs
            )
        )

        self.status = "validated"

        return self.evaluation_result

    # =========================================================
    # EVALUATE
    # =========================================================

    def evaluate(
        self,
        dataset=None,
        **kwargs
    ):

        if dataset is None:

            raise ValueError(
                "Dataset is required."
            )

        if self._evaluator is None:

            raise NotImplementedError(
                "Evaluator is not configured."
            )

        self.evaluation_result = (
            self._evaluator.evaluate(
                model=self,
                dataset=dataset,
                **kwargs
            )
        )

        self.status = "evaluated"

        return self.evaluation_result

    # =========================================================
    # OPTIMIZE
    # =========================================================

    def optimize(
        self,
        **kwargs
    ):

        if self._model is None:

            self.load()

        if self._optimizer is None:

            raise NotImplementedError(
                "ModelOptimizer is not configured."
            )

        self.optimization_result = (
            self._optimizer.optimize(
                model=self,
                **kwargs
            )
        )

        self.status = "optimized"

        return self.optimization_result

    # =========================================================
    # BENCHMARK
    # =========================================================

    def benchmark(
        self,
        data=None,
        **kwargs
    ):

        if self._model is None:

            self.load()

        if hasattr(
            self._model,
            "predict"
        ):

            import time

            start = time.perf_counter()

            self._model.predict(
                data,
                verbose=False,
                **kwargs
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            self.benchmark_result = {

                "latency_ms":
                    elapsed * 1000,

                "fps":
                    1.0 / elapsed
                    if elapsed > 0
                    else None,

            }

            return self.benchmark_result

        raise RuntimeError(
            "Model does not support benchmarking."
        )

    # =========================================================
    # ENABLE ADAPTIVE INFERENCE
    # =========================================================

    def enable_adaptive_inference(
        self,
        **kwargs
    ):

        from .adaptive import (
            AdaptiveController
        )

        self._adaptive_controller = (
            AdaptiveController(
                model=self,
                **kwargs
            )
        )

        self.adaptive_inference = True

        return self._adaptive_controller

    # =========================================================
    # EXPORT
    # =========================================================

    def export(
        self,
        format="onnx",
        **kwargs
    ):

        if self._model is None:

            self.load()

        if not hasattr(
            self._model,
            "export"
        ):

            raise RuntimeError(
                "Model backend does not "
                "support export."
            )

        result = self._model.export(
            format=format,
            **kwargs
        )

        self.export_result = {

            "format": format,

            "result": str(result)

        }

        self.status = "exported"

        return self.export_result

    # =========================================================
    # DEPLOY
    # =========================================================

    def deploy(
        self,
        **kwargs
    ):

        if self._model is None:

            self.load()

        self.deployment_result = {

            "status": "ready",

            "model":
                self.model_path,

            "configuration":
                kwargs

        }

        self.status = "deployed"

        return self.deployment_result

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        output_path
    ):

        if self._model is None:

            raise ValueError(
                "Model is not loaded."
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if hasattr(
            self._model,
            "save"
        ):

            self._model.save(
                str(output_path)
            )

        else:

            raise RuntimeError(
                "Model backend does not "
                "support save()."
            )

        self.model_path = str(
            output_path
        )

        return {

            "status": "saved",

            "path":
                str(output_path)

        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self):

        return {

            "model":
                self.model_path,

            "status":
                self.status,

            "loaded":
                self._model is not None,

            "training":
                self.training_result,

            "evaluation":
                self.evaluation_result,

            "optimization":
                self.optimization_result,

            "benchmark":
                self.benchmark_result,

            "deployment":
                self.deployment_result,

            "export":
                self.export_result,

        }