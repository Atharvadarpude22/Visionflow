from pathlib import Path
from typing import Optional, Any, Dict


class Dataset:
    """
    Public Dataset API for VisionFlow.

    Example
    -------
    dataset = vf.Dataset("data/")

    The Dataset class coordinates the internal dataset components.
    """

    def __init__(
        self,
        path: str,
        config: Optional[Any] = None
    ):
        if path is None:
            raise ValueError("Dataset path cannot be None")

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.path}"
            )

        if not self.path.is_dir():
            raise NotADirectoryError(
                f"Dataset path must be a directory: {self.path}"
            )

        self.config = config or {}

        # Internal components
        from .preparer import DatasetPreparer
        from .analyzer import DatasetAnalyzer
        from .optimizer import DatasetOptimizer
        from .intelligence import DatasetIntelligence
        from .versioner import DatasetVersioner

        self._preparer = DatasetPreparer(path=self.path, config=config)
        self._analyzer = DatasetAnalyzer(path=self.path)
        self._optimizer = DatasetOptimizer(path=self.path)

        # Intelligence / Versioner: try the supported constructor
        # signatures in order (dataset=, positional, no arguments).
        self._intelligence = self._build(DatasetIntelligence)
        self._versioner = self._build(DatasetVersioner)

        # Results
        self._preparation_result = None
        self._analysis_result = None
        self._optimization_result = None
        self._gap_result = None
        self._version_result = None

    def _build(self, component):
        """Construct a component with whichever signature it supports."""

        try:
            return component(dataset=self)
        except TypeError:
            pass

        try:
            return component(self)
        except TypeError:
            return component()

    # =========================================================
    # DATASET YAML / TRAINING CONFIGURATION
    # =========================================================

    def get_yaml(self) -> Optional[Path]:
        """
        Find an existing dataset YAML file.

        Searches: explicit config, dataset.yaml, data.yaml,
        config.yaml, then *.yaml / *.yml in the dataset folder.
        """

        # Explicit configuration
        if isinstance(self.config, (str, Path)):
            config_path = Path(self.config)

            if config_path.exists():
                return config_path

        # Dictionary configuration
        if isinstance(self.config, dict):

            yaml_path = (
                self.config.get("yaml")
                or self.config.get("data_yaml")
                or self.config.get("dataset_yaml")
            )

            if yaml_path:

                yaml_path = Path(yaml_path)

                if yaml_path.exists():
                    return yaml_path

        # Common names
        for name in ("dataset.yaml", "data.yaml", "config.yaml"):

            candidate = self.path / name

            if candidate.exists():
                return candidate

        # First YAML file only (no full directory listing kept)
        for pattern in ("*.yaml", "*.yml"):

            first = next(self.path.glob(pattern), None)

            if first is not None:
                return first

        return None

    # =========================================================
    # TRAINING INFORMATION
    # =========================================================

    def training_config(self) -> Dict[str, Any]:
        """Return dataset information required by Trainer."""

        yaml_path = self.get_yaml()

        return {
            "dataset_path": str(self.path),
            "yaml": str(yaml_path) if yaml_path else None,
            "exists": self.path.exists(),
        }

    # =========================================================
    # PIPELINE ACTIONS
    # =========================================================

    def prepare(self, **kwargs):

        self._preparation_result = self._preparer.prepare(**kwargs)

        return self._preparation_result

    def analyze(self, **kwargs):

        self._analysis_result = self._analyzer.report(**kwargs)

        return self._analysis_result

    def optimize(self, **kwargs):

        self._optimization_result = self._optimizer.optimize(**kwargs)

        return self._optimization_result

    def find_gaps(
        self,
        model=None,
        **kwargs
    ):

        self._gap_result = self._intelligence.find_gaps(
            model=model,
            **kwargs
        )

        return self._gap_result

    def version(
        self,
        description: str = "",
        changes=None,
        model=None,
        metadata=None
    ):

        self._version_result = self._versioner.create(
            description=description,
            changes=changes,
            model=model,
            metadata=metadata
        )

        return self._version_result

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> dict:

        return {
            "path": str(self.path),
            "exists": self.path.exists(),
            "training_config": self.training_config(),
            "preparation": self._preparation_result,
            "analysis": self._analysis_result,
            "optimization": self._optimization_result,
            "gaps": self._gap_result,
            "version": self._version_result
        }