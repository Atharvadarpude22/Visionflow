from typing import Optional, Dict, Any, List


class TrainingOptimizer:
    """
    Analyzes the training process and recommends improvements.

    TrainingOptimizer does NOT train the model itself.

    Responsibilities
    ----------------
    - Analyze training history
    - Detect overfitting
    - Detect underfitting
    - Recommend training changes
    - Decide whether training should stop

    Expected history format:

    [
        {
            "epoch": 1,
            "train": {
                "loss": 0.80,
                "accuracy": 0.65
            },
            "validation": {
                "val_loss": 0.75,
                "val_accuracy": 0.68
            }
        },
        ...
    ]
    """

    def __init__(
        self,
        config: Optional[dict] = None
    ):
        self.config = config or {}

        # -------------------------------------------------
        # Thresholds
        # -------------------------------------------------

        self.overfitting_patience = self.config.get(
            "overfitting_patience",
            3
        )

        self.underfitting_threshold = self.config.get(
            "underfitting_threshold",
            0.60
        )

        self.min_improvement = self.config.get(
            "min_improvement",
            0.001
        )

        self.loss_gap_threshold = self.config.get(
            "loss_gap_threshold",
            0.10
        )

        self.accuracy_gap_threshold = self.config.get(
            "accuracy_gap_threshold",
            0.10
        )

        self.max_epochs = self.config.get(
            "max_epochs",
            100
        )

        # -------------------------------------------------
        # Internal state
        # -------------------------------------------------

        self.history: List[dict] = []

        self.analysis_result: Dict[str, Any] = {}

        self.recommendations: List[dict] = []

        self.stop_training = False

    # =====================================================
    # SET HISTORY
    # =====================================================

    def set_history(
        self,
        history: Optional[list]
    ):
        """
        Set training history.

        Accepts either:

        1. VisionFlow history:

        [
            {
                "epoch": 1,
                "train": {...},
                "validation": {...}
            }
        ]

        2. Simple history:

        [
            {
                "epoch": 1,
                "loss": 0.5,
                "val_loss": 0.6,
                "accuracy": 0.8,
                "val_accuracy": 0.75
            }
        ]
        """

        if history is None:

            self.history = []

            return self

        if not isinstance(
            history,
            list
        ):

            raise TypeError(
                "Training history must be a list."
            )

        self.history = history

        return self

    # =====================================================
    # EXTRACT METRIC
    # =====================================================

    def _get_metric(
        self,
        record: dict,
        metric: str
    ):
        """
        Safely extract a metric from different
        history formats.
        """

        if not isinstance(
            record,
            dict
        ):
            return None

        # -------------------------------------------------
        # Direct format
        # -------------------------------------------------

        if metric in record:

            return record[metric]

        # -------------------------------------------------
        # Nested VisionFlow format
        # -------------------------------------------------

        train = record.get(
            "train",
            {}
        )

        validation = record.get(
            "validation",
            {}
        )

        if metric in train:

            return train[metric]

        if metric in validation:

            return validation[metric]

        # -------------------------------------------------
        # Aliases
        # -------------------------------------------------

        aliases = {

            "loss": [
                "train_loss"
            ],

            "val_loss": [
                "validation_loss",
                "valid_loss"
            ],

            "accuracy": [
                "train_accuracy"
            ],

            "val_accuracy": [
                "validation_accuracy",
                "valid_accuracy"
            ]
        }

        for alias in aliases.get(
            metric,
            []
        ):

            if alias in record:

                return record[alias]

            if alias in train:

                return train[alias]

            if alias in validation:

                return validation[alias]

        return None

    # =====================================================
    # EXTRACT EPOCH
    # =====================================================

    def _get_epoch(
        self,
        record: dict,
        index: int
    ) -> int:

        if isinstance(
            record,
            dict
        ):

            epoch = record.get(
                "epoch"
            )

            if epoch is not None:

                return epoch

        return index + 1

    # =====================================================
    # GET METRIC SERIES
    # =====================================================

    def _metric_series(
        self,
        metric: str
    ) -> List[float]:

        values = []

        for record in self.history:

            value = self._get_metric(
                record,
                metric
            )

            if value is not None:

                try:

                    values.append(
                        float(value)
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    continue

        return values

    # =====================================================
    # ANALYZE TRAINING
    # =====================================================

    def analyze_training(
        self,
        history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Analyze training curves.

        Looks at:

        - training loss
        - validation loss
        - training accuracy
        - validation accuracy
        - improvement
        - generalization gap
        """

        if history is not None:

            self.set_history(
                history
            )

        if not self.history:

            return {
                "status": "insufficient_data",
                "message": "Training history is empty."
            }

        train_loss = self._metric_series(
            "loss"
        )

        val_loss = self._metric_series(
            "val_loss"
        )

        train_accuracy = self._metric_series(
            "accuracy"
        )

        val_accuracy = self._metric_series(
            "val_accuracy"
        )

        result = {

            "epochs": len(
                self.history
            ),

            "training_loss": train_loss,

            "validation_loss": val_loss,

            "training_accuracy": train_accuracy,

            "validation_accuracy": val_accuracy,

            "loss_gap": None,

            "accuracy_gap": None,

            "loss_improving": None,

            "accuracy_improving": None
        }

        # -------------------------------------------------
        # Loss gap
        # -------------------------------------------------

        if train_loss and val_loss:

            result["loss_gap"] = (
                train_loss[-1]
                - val_loss[-1]
            )

        # -------------------------------------------------
        # Accuracy gap
        # -------------------------------------------------

        if train_accuracy and val_accuracy:

            result["accuracy_gap"] = (
                train_accuracy[-1]
                - val_accuracy[-1]
            )

        # -------------------------------------------------
        # Loss improvement
        # -------------------------------------------------

        if len(train_loss) >= 2:

            result["loss_improving"] = (
                train_loss[-1]
                < train_loss[0]
            )

        # -------------------------------------------------
        # Accuracy improvement
        # -------------------------------------------------

        if len(train_accuracy) >= 2:

            result["accuracy_improving"] = (
                train_accuracy[-1]
                > train_accuracy[0]
            )

        self.analysis_result = result

        return result

    # =====================================================
    # DETECT OVERFITTING
    # =====================================================

    def detect_overfitting(
        self,
        history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Detect overfitting.

        Typical pattern:

            Training loss      ↓
            Validation loss   ↑

        OR

            Training accuracy      ↑
            Validation accuracy   ↓
        """

        if history is not None:

            self.set_history(
                history
            )

        if len(self.history) < 2:

            return {
                "detected": False,
                "reason": "Insufficient history."
            }

        train_loss = self._metric_series(
            "loss"
        )

        val_loss = self._metric_series(
            "val_loss"
        )

        train_accuracy = self._metric_series(
            "accuracy"
        )

        val_accuracy = self._metric_series(
            "val_accuracy"
        )

        signals = []

        # -------------------------------------------------
        # Loss-based detection
        # -------------------------------------------------

        if len(train_loss) >= 2 and len(val_loss) >= 2:

            training_improving = (
                train_loss[-1]
                < train_loss[-2]
            )

            validation_worsening = (
                val_loss[-1]
                > val_loss[-2]
            )

            if (
                training_improving
                and validation_worsening
            ):

                signals.append(
                    "training_loss_decreasing_while_validation_loss_increases"
                )

        # -------------------------------------------------
        # Accuracy-based detection
        # -------------------------------------------------

        if (
            len(train_accuracy) >= 2
            and len(val_accuracy) >= 2
        ):

            training_improving = (
                train_accuracy[-1]
                > train_accuracy[-2]
            )

            validation_worsening = (
                val_accuracy[-1]
                < val_accuracy[-2]
            )

            if (
                training_improving
                and validation_worsening
            ):

                signals.append(
                    "training_accuracy_increasing_while_validation_accuracy_decreases"
                )

        # -------------------------------------------------
        # Generalization gap
        # -------------------------------------------------

        if (
            train_accuracy
            and val_accuracy
        ):

            gap = (
                train_accuracy[-1]
                - val_accuracy[-1]
            )

            if gap >= self.accuracy_gap_threshold:

                signals.append(
                    "large_training_validation_accuracy_gap"
                )

        if (
            train_loss
            and val_loss
        ):

            loss_gap = (
                val_loss[-1]
                - train_loss[-1]
            )

            if loss_gap >= self.loss_gap_threshold:

                signals.append(
                    "large_training_validation_loss_gap"
                )

        detected = (
            len(signals) > 0
        )

        result = {

            "detected":
                detected,

            "signals":
                signals,

            "severity":
                (
                    "high"
                    if len(signals) >= 2
                    else "moderate"
                    if detected
                    else "none"
                )
        }

        return result

    # =====================================================
    # DETECT UNDERFITTING
    # =====================================================

    def detect_underfitting(
        self,
        history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Detect underfitting.

        Typical signs:

        - training accuracy remains low
        - validation accuracy remains low
        - loss remains high
        - training itself is not improving sufficiently
        """

        if history is not None:

            self.set_history(
                history
            )

        if not self.history:

            return {
                "detected": False,
                "reason": "Insufficient history."
            }

        train_accuracy = self._metric_series(
            "accuracy"
        )

        val_accuracy = self._metric_series(
            "val_accuracy"
        )

        train_loss = self._metric_series(
            "loss"
        )

        signals = []

        # -------------------------------------------------
        # Low training accuracy
        # -------------------------------------------------

        if train_accuracy:

            if (
                train_accuracy[-1]
                < self.underfitting_threshold
            ):

                signals.append(
                    "low_training_accuracy"
                )

        # -------------------------------------------------
        # Low validation accuracy
        # -------------------------------------------------

        if val_accuracy:

            if (
                val_accuracy[-1]
                < self.underfitting_threshold
            ):

                signals.append(
                    "low_validation_accuracy"
                )

        # -------------------------------------------------
        # Training not improving
        # -------------------------------------------------

        if len(train_accuracy) >= 2:

            improvement = (
                train_accuracy[-1]
                - train_accuracy[0]
            )

            if improvement < self.min_improvement:

                signals.append(
                    "training_accuracy_not_improving"
                )

        # -------------------------------------------------
        # Loss not decreasing
        # -------------------------------------------------

        if len(train_loss) >= 2:

            if train_loss[-1] >= train_loss[0]:

                signals.append(
                    "training_loss_not_decreasing"
                )

        detected = (
            len(signals) >= 2
        )

        return {

            "detected":
                detected,

            "signals":
                signals,

            "severity":
                (
                    "moderate"
                    if detected
                    else "none"
                )
        }

    # =====================================================
    # RECOMMEND
    # =====================================================

    def recommend(
        self,
        history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Recommend training changes based on analysis.

        Possible recommendations:

        - reduce_learning_rate
        - increase_learning_rate
        - increase_regularization
        - increase_augmentation
        - increase_epochs
        - stop_training
        """

        if history is not None:

            self.set_history(
                history
            )

        analysis = self.analyze_training()

        overfitting = self.detect_overfitting()

        underfitting = self.detect_underfitting()

        recommendations = []

        # -------------------------------------------------
        # Overfitting recommendations
        # -------------------------------------------------

        if overfitting["detected"]:

            recommendations.extend([

                {
                    "action":
                        "increase_regularization",

                    "reason":
                        "Training performance is improving "
                        "while validation performance is degrading."
                },

                {
                    "action":
                        "increase_augmentation",

                    "reason":
                        "Additional data variation may improve "
                        "generalization."
                },

                {
                    "action":
                        "reduce_epochs",

                    "reason":
                        "Continuing training may increase "
                        "overfitting."
                }
            ])

        # -------------------------------------------------
        # Underfitting recommendations
        # -------------------------------------------------

        if underfitting["detected"]:

            recommendations.extend([

                {
                    "action":
                        "increase_epochs",

                    "reason":
                        "The model has not learned the training "
                        "data sufficiently."
                },

                {
                    "action":
                        "review_learning_rate",

                    "reason":
                        "The current learning rate may not be "
                        "allowing effective optimization."
                },

                {
                    "action":
                        "reduce_excessive_regularization",

                    "reason":
                        "Strong regularization can prevent "
                        "the model from learning sufficiently."
                }
            ])

        # -------------------------------------------------
        # No obvious problem
        # -------------------------------------------------

        if not recommendations:

            recommendations.append({

                "action":
                    "continue_training",

                "reason":
                    "No strong overfitting or underfitting "
                    "signal was detected."
            })

        self.recommendations = (
            recommendations
        )

        return {

            "analysis":
                analysis,

            "overfitting":
                overfitting,

            "underfitting":
                underfitting,

            "recommendations":
                recommendations
        }

    # =====================================================
    # SHOULD STOP
    # =====================================================

    def should_stop(
        self,
        history: Optional[list] = None
    ) -> bool:
        """
        Determine whether training should stop.

        Stops when:

        - severe overfitting is detected
        - validation metric stops improving
        - maximum epochs are reached
        """

        if history is not None:

            self.set_history(
                history
            )

        if not self.history:

            return False

        # -------------------------------------------------
        # Maximum epochs
        # -------------------------------------------------

        current_epoch = self._get_epoch(
            self.history[-1],
            len(self.history) - 1
        )

        if current_epoch >= self.max_epochs:

            self.stop_training = True

            return True

        # -------------------------------------------------
        # Overfitting
        # -------------------------------------------------

        overfitting = (
            self.detect_overfitting()
        )

        if (
            overfitting["detected"]
            and overfitting["severity"] == "high"
        ):

            self.stop_training = True

            return True

        # -------------------------------------------------
        # Validation loss plateau
        # -------------------------------------------------

        val_loss = self._metric_series(
            "val_loss"
        )

        if len(val_loss) >= (
            self.overfitting_patience + 1
        ):

            recent = val_loss[
                -self.overfitting_patience:
            ]

            improvement = (
                min(recent)
                < val_loss[
                    -self.overfitting_patience - 1
                ]
                - self.min_improvement
            )

            if not improvement:

                self.stop_training = True

                return True

        return False

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self):
        """
        Return optimizer state.
        """

        return {

            "epochs_analyzed":
                len(self.history),

            "stop_training":
                self.stop_training,

            "analysis":
                self.analysis_result,

            "recommendations":
                self.recommendations
        }