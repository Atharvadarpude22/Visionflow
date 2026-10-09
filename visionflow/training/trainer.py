from pathlib import Path
from typing import Optional, Any, Dict
import json
import shutil
import time


class Trainer:
    """
    VisionFlow training engine.

    Responsibilities
    ----------------
    - Model training
    - GPU/CPU detection
    - Dataset YAML handling
    - Checkpoints
    - Validation
    - Early stopping
    - Training history
    - Edge deployment target configuration

    VisionFlow does NOT train directly on edge devices by default.

    Instead:

        Training machine
              ↓
           Train
              ↓
        Validate model
              ↓
        Optimize for edge
              ↓
        Deploy to edge
    """

    def __init__(
        self,
        model=None,
        dataset=None,
        output_dir: str = "runs/visionflow",
        device: str = "auto",
        edge_device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):

        self.model = model
        self.dataset = dataset

        self.output_dir = Path(output_dir)

        self.device = device

        self.edge_device = edge_device

        self.config = config or {}

        # =====================================================
        # TRAINING STATE
        # =====================================================

        self.history = []

        self.current_epoch = 0

        self.best_metric = None

        self.best_checkpoint = None

        self.training_result = None

        self.status = "created"

        self.start_time = None

        # =====================================================
        # DEFAULT TRAINING CONFIG
        # =====================================================

        self.training_config = {

            "epochs": 10,

            "batch": 16,

            "imgsz": 640,

            "lr0": 0.01,

            "patience": 10,

            "workers": 4,

            "save": True,

            "verbose": True,

        }

        # User config overrides defaults
        self.training_config.update(
            self.config.get(
                "training",
                {}
            )
        )

        # =====================================================
        # EDGE CONFIGURATION
        # =====================================================

        self.edge_config = {

            "device": edge_device,

            "target": edge_device,

            "precision": "fp32",

            "imgsz": self.training_config["imgsz"],

        }

        if edge_device:

            self.edge_config.update(
                self._get_edge_defaults(
                    edge_device
                )
            )

    # =========================================================
    # DEVICE DETECTION
    # =========================================================

    def detect_device(self) -> str:
        """
        Automatically select CUDA when available.

        Returns
        -------
        'cuda' or 'cpu'
        """

        if self.device != "auto":

            return self.device

        try:

            import torch

            if torch.cuda.is_available():

                return "cuda"

        except ImportError:

            pass

        return "cpu"

    # =========================================================
    # EDGE DEFAULTS
    # =========================================================

    def _get_edge_defaults(
        self,
        edge_device: str
    ) -> Dict[str, Any]:

        device = edge_device.lower()

        if "jetson" in device:

            return {

                "precision": "fp16",

                "imgsz": 640,

                "runtime": "tensorrt",

            }

        if "raspberry" in device:

            return {

                "precision": "int8",

                "imgsz": 416,

                "runtime": "onnx",

            }

        if "cpu" in device:

            return {

                "precision": "int8",

                "imgsz": 416,

                "runtime": "onnx",

            }

        return {

            "precision": "fp32",

            "imgsz":
                self.training_config["imgsz"],

            "runtime": "onnx",

        }

    # =========================================================
    # DATASET YAML
    # =========================================================

    def get_dataset_yaml(self):

        if self.dataset is None:

            raise ValueError(
                "Dataset is required for training."
            )

        # Dataset public method
        if hasattr(
            self.dataset,
            "get_yaml"
        ):

            yaml_path = (
                self.dataset.get_yaml()
            )

            if yaml_path:

                return Path(yaml_path)

        # Dataset training config
        if hasattr(
            self.dataset,
            "training_config"
        ):

            information = (
                self.dataset.training_config()
            )

            yaml_path = information.get(
                "yaml"
            )

            if yaml_path:

                return Path(yaml_path)

        # Direct dataset path
        dataset_path = Path(
            self.dataset.path
        )

        for filename in [
            "dataset.yaml",
            "data.yaml",
        ]:

            candidate = (
                dataset_path / filename
            )

            if candidate.exists():

                return candidate

        # =====================================================
        # AUTO GENERATION
        # =====================================================

        return self._generate_dataset_yaml()

    # =========================================================
    # GENERATE YOLO YAML
    # =========================================================

    def _generate_dataset_yaml(self):

        try:
            import yaml
        except ImportError:

            raise ImportError(
                "PyYAML is required. "
                "Install using: pip install pyyaml"
            )

        dataset_path = Path(
            self.dataset.path
        )

        train_path = dataset_path / "train"
        val_path = dataset_path / "val"
        test_path = dataset_path / "test"

        # Also support validation naming
        if not val_path.exists():

            val_path = (
                dataset_path / "valid"
            )

        # Prepared dataset may itself be train
        if (
            (dataset_path / "images").exists()
            and
            (dataset_path / "labels").exists()
        ):

            train_path = dataset_path

        if not train_path.exists():

            raise FileNotFoundError(
                "Training split could not be found.\n"
                f"Expected:\n"
                f"{dataset_path / 'train'}"
            )

        # =====================================================
        # DETECT CLASS NAMES
        # =====================================================

        names = self._detect_class_names(
            dataset_path
        )

        if not names:

            raise ValueError(
                "Could not determine dataset classes.\n"
                "Provide dataset.yaml."
            )

        yaml_path = (
            dataset_path / "visionflow_data.yaml"
        )

        data = {

            "path": str(dataset_path),

            "train": str(
                train_path.relative_to(
                    dataset_path
                )
            ),

            "names": names,

        }

        if val_path.exists():

            data["val"] = str(
                val_path.relative_to(
                    dataset_path
                )
            )

        if test_path.exists():

            data["test"] = str(
                test_path.relative_to(
                    dataset_path
                )
            )

        with open(
            yaml_path,
            "w",
            encoding="utf-8"
        ) as file:

            yaml.safe_dump(
                data,
                file,
                sort_keys=False
            )

        return yaml_path

    # =========================================================
    # CLASS DETECTION
    # =========================================================

    def _detect_class_names(
        self,
        dataset_path: Path
    ):

        # Existing YAML
        for filename in [
            "dataset.yaml",
            "data.yaml"
        ]:

            yaml_file = (
                dataset_path / filename
            )

            if yaml_file.exists():

                try:

                    import yaml

                    with open(
                        yaml_file,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        data = yaml.safe_load(
                            file
                        ) or {}

                    names = data.get(
                        "names"
                    )

                    if names:

                        if isinstance(
                            names,
                            list
                        ):

                            return {
                                i: name
                                for i, name
                                in enumerate(names)
                            }

                        return names

                except Exception:

                    pass

        # Dataset class directories
        images_path = (
            dataset_path / "images"
        )

        if images_path.exists():

            directories = [
                p.name
                for p in images_path.iterdir()
                if p.is_dir()
            ]

            if directories:

                return {
                    i: name
                    for i, name
                    in enumerate(
                        sorted(directories)
                    )
                }

        return None

    # =========================================================
    # TRAIN
    # =========================================================

    def train(
        self,
        dataset=None,
        epochs=None,
        batch=None,
        imgsz=None,
        device=None,
        **kwargs
    ):

        if dataset is not None:

            self.dataset = dataset

        if self.dataset is None:

            raise ValueError(
                "Dataset is required."
            )

        self.status = "preparing"

        self.start_time = time.time()

        # =====================================================
        # DEVICE
        # =====================================================

        selected_device = (
            device
            if device is not None
            else self.detect_device()
        )

        # =====================================================
        # TRAINING PARAMETERS
        # =====================================================

        epochs = (
            epochs
            if epochs is not None
            else self.training_config["epochs"]
        )

        batch = (
            batch
            if batch is not None
            else self.training_config["batch"]
        )

        imgsz = (
            imgsz
            if imgsz is not None
            else self.edge_config["imgsz"]
        )

        patience = self.training_config[
            "patience"
        ]

        # =====================================================
        # DATASET YAML
        # =====================================================

        data_yaml = self.get_dataset_yaml()

        print(
            "\nVisionFlow Training"
        )

        print(
            f"Dataset : {self.dataset.path}"
        )

        print(
            f"YAML    : {data_yaml}"
        )

        print(
            f"Device  : {selected_device}"
        )

        print(
            f"Epochs  : {epochs}"
        )

        print(
            f"Batch   : {batch}"
        )

        print(
            f"Image   : {imgsz}"
        )

        if self.edge_device:

            print(
                f"Edge target: "
                f"{self.edge_device}"
            )

        # =====================================================
        # ULTRALYTICS
        # =====================================================

        try:

            from ultralytics import YOLO

        except ImportError:

            raise ImportError(
                "Ultralytics is required for "
                "YOLO training.\n\n"
                "Install:\n"
                "pip install ultralytics"
            )

        # =====================================================
        # GET MODEL
        # =====================================================

        yolo_model = self._get_yolo_model()

        # =====================================================
        # OUTPUT
        # =====================================================

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =====================================================
        # ACTUAL TRAINING
        # =====================================================

        self.status = "training"

        results = yolo_model.train(

            data=str(data_yaml),

            epochs=epochs,

            batch=batch,

            imgsz=imgsz,

            device=selected_device,

            patience=patience,

            workers=self.training_config[
                "workers"
            ],

            project=str(
                self.output_dir
            ),

            name="train",

            exist_ok=True,

            **kwargs
        )

        # =====================================================
        # SAVE RESULT
        # =====================================================

        self.training_result = {

            "status": "completed",

            "device": selected_device,

            "dataset": str(
                self.dataset.path
            ),

            "dataset_yaml": str(
                data_yaml
            ),

            "epochs": epochs,

            "batch": batch,

            "image_size": imgsz,

            "edge_target":
                self.edge_device,

            "precision":
                self.edge_config[
                    "precision"
                ],

            "runtime":
                self.edge_config.get(
                    "runtime"
                ),

            "results":
                str(results),

        }

        self.status = "trained"

        return self.training_result

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _get_yolo_model(self):

        try:

            from ultralytics import YOLO

        except ImportError:

            raise ImportError(
                "Install Ultralytics:\n"
                "pip install ultralytics"
            )

        # Model object from Model class
        if self.model is not None:

            if hasattr(
                self.model,
                "_model"
            ):

                if self.model._model is not None:

                    return self.model._model

            if hasattr(
                self.model,
                "model"
            ):

                if self.model.model is not None:

                    return self.model.model

        raise ValueError(
            "Trainer could not obtain the "
            "YOLO model."
        )

    # =========================================================
    # VALIDATE
    # =========================================================

    def validate(
        self,
        dataset=None,
        **kwargs
    ):

        if dataset is not None:

            self.dataset = dataset

        if self.dataset is None:

            raise ValueError(
                "Dataset is required."
            )

        if self.model is None:

            raise ValueError(
                "Model is required."
            )

        yolo_model = self._get_yolo_model()

        data_yaml = (
            self.get_dataset_yaml()
        )

        device = self.detect_device()

        result = yolo_model.val(

            data=str(data_yaml),

            device=device,

            **kwargs

        )

        self.status = "validated"

        return result

    # =========================================================
    # SAVE CHECKPOINT
    # =========================================================

    def save_checkpoint(
        self,
        checkpoint_path=None
    ):

        if checkpoint_path is None:

            checkpoint_path = (
                self.output_dir
                / "checkpoint.pt"
            )

        checkpoint_path = Path(
            checkpoint_path
        )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if self.model is not None:

            if hasattr(
                self.model,
                "_model"
            ):

                yolo_model = (
                    self.model._model
                )

                if hasattr(
                    yolo_model,
                    "save"
                ):

                    yolo_model.save(
                        str(checkpoint_path)
                    )

                    self.best_checkpoint = (
                        checkpoint_path
                    )

                    return {
                        "checkpoint":
                            str(checkpoint_path),
                        "status":
                            "saved"
                    }

        return {
            "status":
                "checkpoint_not_saved"
        }

    # =========================================================
    # LOAD CHECKPOINT
    # =========================================================

    def load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        try:

            from ultralytics import YOLO

            model = YOLO(
                str(checkpoint_path)
            )

        except ImportError:

            raise ImportError(
                "Install ultralytics."
            )

        if self.model is not None:

            self.model._model = model

            self.model.model_path = (
                str(checkpoint_path)
            )

        self.status = "checkpoint_loaded"

        return model

    # =========================================================
    # TRAIN ONE EPOCH
    # =========================================================

    def train_epoch(
        self,
        epoch: int,
        **kwargs
    ):

        """
        Compatibility method.

        Ultralytics manages its own epoch loop.
        Therefore VisionFlow delegates the epoch execution
        to the backend rather than duplicating it.
        """

        self.current_epoch = epoch

        return {
            "epoch": epoch,
            "status":
                "delegated_to_training_backend"
        }

    # =========================================================
    # EARLY STOP
    # =========================================================

    def early_stop(
        self,
        current_metric,
        patience=5,
        mode="max"
    ):

        if not self.history:

            self.history.append(
                current_metric
            )

            return False

        self.history.append(
            current_metric
        )

        if len(
            self.history
        ) <= patience:

            return False

        recent = self.history[
            -patience:
        ]

        if mode == "max":

            best = max(recent)

            return (
                recent[-1] < best
            )

        best = min(recent)

        return (
            recent[-1] > best
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self):

        return {

            "status":
                self.status,

            "device":
                self.detect_device(),

            "edge_device":
                self.edge_device,

            "edge_configuration":
                self.edge_config,

            "training_configuration":
                self.training_config,

            "dataset":
                (
                    str(self.dataset.path)
                    if self.dataset is not None
                    else None
                ),

            "output_directory":
                str(self.output_dir),

            "training_result":
                self.training_result,

        }