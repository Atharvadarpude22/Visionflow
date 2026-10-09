from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional

_MISSING = object()


class InferenceEngine:
    """
    VisionFlow real-time inference engine.

    Handles: preprocess, infer, postprocess, predict, process_video,
    process_stream, frame_skip, tracking, release.
    """

    def __init__(
        self,
        model: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.config = config or {}

        self._capture = None
        self._runtime = None

        self.frame_counter = 0
        self.last_result = None

        self.skip_interval = self.config.get("frame_skip", 0)

        self.tracking_enabled = False

        self.performance = {
            "frames": 0,
            "inference_time": 0.0,
            "latency_ms": 0.0,
            "fps": 0.0,
        }

        self.status = "created"

    # =========================================================
    # PREPROCESS
    # =========================================================

    def preprocess(
        self,
        data,
        size=None,
        normalize=None,
        **kwargs,
    ):
        """
        Preprocess input.

        Supports common NumPy/OpenCV input while keeping the
        implementation backend-independent.
        """

        if data is None:
            raise ValueError("Input data cannot be None")

        processed = data

        # Optional OpenCV resize
        if size is not None:

            try:
                import cv2

                if isinstance(size, int):
                    target = (size, size)
                else:
                    target = tuple(size)

                processed = cv2.resize(processed, target)

            except ImportError:
                raise RuntimeError(
                    "OpenCV is required for image resizing"
                )

        # Optional normalization
        if normalize:

            try:

                # In-place divide on the converted copy: one array
                # allocation fewer than astype(...) / 255.0.
                processed = processed.astype("float32")
                processed /= 255.0

            except AttributeError:
                pass

        return processed

    # =========================================================
    # INFERENCE
    # =========================================================

    def infer(
        self,
        data,
        **kwargs,
    ):
        """Run the actual model inference."""

        if self.model is None:
            raise ValueError("Model is not configured")

        # VisionFlow Model wrapper (single attribute lookup)
        underlying = getattr(self.model, "_model", _MISSING)

        if underlying is _MISSING:
            underlying = self.model

        if underlying is None:
            raise ValueError("Underlying model is not loaded")

        # predict()
        predict = getattr(underlying, "predict", _MISSING)

        if predict is not _MISSING:
            return predict(data, **kwargs)

        # callable model
        if callable(underlying):
            return underlying(data)

        raise NotImplementedError(
            "Model backend does not provide predict() "
            "or callable inference"
        )

    # =========================================================
    # POSTPROCESS
    # =========================================================

    def postprocess(
        self,
        results,
        confidence: Optional[float] = None,
        nms: bool = False,
        **kwargs,
    ):
        """
        Postprocess model output.

        Confidence filtering and NMS are handled by the model
        backend; backend-specific logic can be plugged in here
        without changing the pipeline.
        """

        # The previous per-result confidence loop did nothing (and
        # would exhaust a generator result), so results pass through.
        return results

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(
        self,
        data,
        preprocess=True,
        postprocess=True,
        **kwargs,
    ):
        """
        Complete inference pipeline: preprocess -> infer -> postprocess
        """

        start = time.perf_counter()

        if preprocess:

            data = self.preprocess(
                data,
                size=kwargs.pop("size", None),
                normalize=kwargs.pop("normalize", False),
            )

        results = self.infer(data, **kwargs)

        if postprocess:

            results = self.postprocess(
                results,
                confidence=kwargs.get("confidence"),
            )

        elapsed = time.perf_counter() - start

        performance = self.performance

        performance["frames"] += 1
        performance["inference_time"] += elapsed
        performance["latency_ms"] = elapsed * 1000

        total_time = performance["inference_time"]

        if total_time > 0:
            performance["fps"] = performance["frames"] / total_time

        self.last_result = results

        return results

    # =========================================================
    # FRAME SKIPPING
    # =========================================================

    def frame_skip(
        self,
        interval: Optional[int] = None,
    ):
        """
        Configure how often inference should run.

        interval=0  Process every frame.
        interval=2  Process every second frame.
        """

        if interval is None:
            interval = 0

        if interval < 0:
            raise ValueError("Frame skip interval cannot be negative")

        self.skip_interval = interval

        return self

    # =========================================================
    # TRACKING
    # =========================================================

    def track(
        self,
        enabled: bool = True,
        **kwargs,
    ):
        """
        Enable or disable tracking.

        Actual tracker implementation can be provided by the
        selected model/backend.
        """

        self.tracking_enabled = enabled

        return self

    # =========================================================
    # VIDEO PROCESSING
    # =========================================================

    def process_video(
        self,
        source,
        output_path: Optional[str] = None,
        display: bool = False,
        **kwargs,
    ):
        """Process a video file or camera source."""

        try:
            import cv2

        except ImportError:

            raise RuntimeError("OpenCV is required for video processing")

        capture = cv2.VideoCapture(source)

        if not capture.isOpened():

            raise RuntimeError(f"Unable to open video source: {source}")

        self._capture = capture

        writer = None

        try:

            fps = capture.get(cv2.CAP_PROP_FPS)

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if output_path:

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps if fps > 0 else 30,
                    (width, height),
                )

            # Annotated frames are only needed for the writer or the
            # display window; skip the (costly) drawing otherwise.
            needs_output = writer is not None or display

            processed_frames = 0
            skipped_frames = 0

            while True:

                success, frame = capture.read()

                if not success:
                    break

                self.frame_counter += 1

                # Frame skipping
                if (
                    self.skip_interval > 0
                    and self.frame_counter % (self.skip_interval + 1) != 0
                ):

                    skipped_frames += 1

                    if writer is not None:
                        writer.write(frame)

                    continue

                result = self.predict(frame, **kwargs)

                processed_frames += 1

                if not needs_output:
                    continue

                output_frame = self._draw_results(frame, result)

                if writer is not None:
                    writer.write(output_frame)

                if display:

                    cv2.imshow("VisionFlow", output_frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

            return {
                "status": "completed",
                "processed_frames": processed_frames,
                "skipped_frames": skipped_frames,
                "fps": self.performance["fps"],
                "latency_ms": self.performance["latency_ms"],
                "output": output_path,
            }

        finally:

            self.release()

    # =========================================================
    # LIVE STREAM
    # =========================================================

    def process_stream(
        self,
        source=0,
        display: bool = True,
        **kwargs,
    ):
        """
        Process a live camera/stream source.

        Press 'q' to stop when display=True.
        """

        return self.process_video(
            source=source,
            output_path=None,
            display=display,
            **kwargs,
        )

    # =========================================================
    # DRAW RESULTS
    # =========================================================

    def _draw_results(
        self,
        frame,
        results,
    ):
        """
        Draw results when the backend exposes a plot() method.
        Otherwise return the original frame.
        """

        try:

            if isinstance(results, (list, tuple)):
                first = results[0]

            else:
                first = results

            if hasattr(first, "plot"):
                return first.plot()

        except Exception:
            pass

        return frame

    # =========================================================
    # RELEASE
    # =========================================================

    def release(self):
        """
        Safely release camera, video capture and display resources.
        """

        if self._capture is not None:

            try:
                self._capture.release()

            except Exception:
                pass

            self._capture = None

        # OpenCV windows / CUDA cache: only touched if the library is
        # already loaded in this process. Importing cv2 or torch just
        # to clean up would be slow and could not free anything.
        cv2 = sys.modules.get("cv2")

        if cv2 is not None:

            try:
                cv2.destroyAllWindows()

            except Exception:
                pass

        torch = sys.modules.get("torch")

        if torch is not None:

            try:

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception:
                pass

        self.status = "released"

    # =========================================================
    # PERFORMANCE
    # =========================================================

    def performance_summary(self):
        """Return inference performance information."""

        return {
            "frames": self.performance["frames"],
            "latency_ms": round(self.performance["latency_ms"], 3),
            "fps": round(self.performance["fps"], 3),
            "frame_skip": self.skip_interval,
            "tracking": self.tracking_enabled,
        }