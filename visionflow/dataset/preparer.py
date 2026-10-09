from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict, Counter

import hashlib
import json
import random
import shutil

try:
    import yaml
except ImportError:
    yaml = None

try:
    from PIL import Image, ImageStat, ImageFilter
except ImportError:
    Image = None
    ImageStat = None
    ImageFilter = None


# Fast popcount (int.bit_count needs Python 3.10+)
if hasattr(int, "bit_count"):
    _popcount = int.bit_count
else:
    def _popcount(value):
        return bin(value).count("1")


class DatasetPreparer:
    """
    Prepare a raw computer-vision dataset for training.

    Workflow
    --------
    Raw Dataset -> Load -> Validate -> Annotation Checking ->
    Duplicate Detection -> Quality Checking -> Leakage Detection ->
    KEEP / REVIEW / QUARANTINE -> Source / Video-Aware Split ->
    train / valid / test (images + labels) -> data.yaml ->
    Prepared Dataset
    """

    # =========================================================
    # SUPPORTED FORMATS
    # =========================================================

    SUPPORTED_IMAGE_FORMATS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    }

    SUPPORTED_ANNOTATION_FORMATS = {
        ".txt", ".xml", ".json", ".yaml", ".yml",
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        path: str,
        config=None
    ):

        if path is None:
            raise ValueError("Dataset path cannot be None")

        self.path = Path(path)

        self.config = {}
        self.dataset_config = {}
        self._config_path = None

        # YAML / DICT CONFIGURATION
        if isinstance(config, dict):

            self.config = config

        elif config is not None:

            config_path = Path(config)

            if not config_path.exists():
                raise FileNotFoundError(
                    f"Dataset config not found: {config_path}"
                )

            if yaml is None:
                raise ImportError(
                    "PyYAML is required for YAML configuration. "
                    "Install it using: pip install pyyaml"
                )

            with open(config_path, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file) or {}

            if not isinstance(self.config, dict):
                raise ValueError(
                    "Dataset YAML must contain a mapping/object."
                )

            self._config_path = config_path.resolve()

        self.dataset_config = self.config

        # Resolve dataset path from YAML
        yaml_path = self.dataset_config.get("path")

        if yaml_path:

            yaml_path = Path(yaml_path)

            if not yaml_path.is_absolute() and self._config_path:
                yaml_path = self._config_path.parent / yaml_path

            self.path = yaml_path

        # Validate dataset directory
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.path}"
            )

        if not self.path.is_dir():
            raise NotADirectoryError(
                f"Dataset path must be a directory: {self.path}"
            )

        # Configuration
        self.seed = self.config.get("seed", 42)
        self.blur_threshold = self.config.get("blur_threshold", 20.0)
        self.dark_threshold = self.config.get("dark_threshold", 30.0)
        self.bright_threshold = self.config.get("bright_threshold", 240.0)
        self.min_width = self.config.get("min_width", 32)
        self.min_height = self.config.get("min_height", 32)

        # Samples and results
        self.samples = []

        self.validation_result = None
        self.duplicate_result = None
        self.quality_result = None
        self.leakage_result = None
        self.cleaning_result = None
        self.split_result = None
        self.manifest_result = None
        self.preparation_result = None

        # Directory-existence cache, only alive during load()
        self._dir_cache = {}

    # =========================================================
    # LOAD
    # =========================================================

    def load(
        self
    ) -> List[Dict[str, Any]]:

        self.samples = []
        self._dir_cache = {}

        image_files = []

        # Never scan generated prepared directories.
        ignored_directories = {"prepared", "_quarantine"}

        image_formats = self.SUPPORTED_IMAGE_FORMATS

        for file in self.path.rglob("*"):

            if file.suffix.lower() not in image_formats:
                continue

            if not file.is_file():
                continue

            relative_parts = {
                part.lower()
                for part in file.relative_to(self.path).parts[:-1]
            }

            if relative_parts & ignored_directories:
                continue

            image_files.append(file)

        # Build samples
        for image_path in sorted(image_files):

            annotation_path = self._find_annotation(image_path)

            sample = {
                "image": str(image_path),
                "annotation": (
                    str(annotation_path) if annotation_path else None
                ),
                "source": self._infer_source(image_path),
                "video": self._infer_video(image_path),
                "status": "UNVALIDATED",
                "issues": [],
                "warnings": [],
                "classes": self._extract_classes(annotation_path),
                "metadata": self._extract_image_metadata(image_path),
                "hash": None,
                "perceptual_hash": None,
                "split": None,
            }

            self.samples.append(sample)

        self._dir_cache = {}

        return self.samples

    # =========================================================
    # FIND ANNOTATION
    # =========================================================

    def _find_annotation(
        self,
        image_path: Path
    ) -> Optional[Path]:

        extensions = self.SUPPORTED_ANNOTATION_FORMATS

        # Same directory
        for extension in extensions:

            candidate = image_path.with_suffix(extension)

            if candidate.exists():
                return candidate

        # Common annotation directories
        cache = self._dir_cache
        stem = image_path.stem

        for directory in (
            image_path.parent / "labels",
            image_path.parent / "annotations",
            self.path / "labels",
            self.path / "annotations",
        ):

            exists = cache.get(directory)

            if exists is None:
                exists = cache[directory] = directory.exists()

            if not exists:
                continue

            for extension in extensions:

                candidate = directory / f"{stem}{extension}"

                if candidate.exists():
                    return candidate

        return None

    # =========================================================
    # SOURCE DETECTION
    # =========================================================

    def _infer_source(
        self,
        image_path: Path
    ) -> str:

        try:
            relative = image_path.relative_to(self.path)

        except ValueError:
            return "default_source"

        ignored = {
            "images", "image", "labels", "annotations",
            "train", "val", "valid", "validation", "test",
            "prepared",
        }

        for part in relative.parts[:-1]:

            if part.lower() not in ignored:
                return part

        return "default_source"

    # =========================================================
    # VIDEO DETECTION
    # =========================================================

    def _infer_video(
        self,
        image_path: Path
    ) -> Optional[str]:

        stem = image_path.stem.lower()

        video_keywords = ("video", "sequence", "seq", "frame")

        for parent in image_path.parents:

            if parent == self.path:
                continue

            name = parent.name.lower()

            if any(keyword in name for keyword in video_keywords):
                return name

        if "_frame" in stem:
            return stem.split("_frame")[0]

        if "-frame" in stem:
            return stem.split("-frame")[0]

        return None

    # =========================================================
    # IMAGE METADATA
    # =========================================================

    def _extract_image_metadata(
        self,
        image_path: Path
    ) -> Dict[str, Any]:

        metadata = {
            "format": image_path.suffix.lower(),
            "file_size": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
        }

        try:
            metadata["file_size"] = image_path.stat().st_size

        except OSError:
            pass

        if Image is None:
            return metadata

        try:

            with Image.open(image_path) as image:

                width, height = image.size

                metadata["width"] = width
                metadata["height"] = height

                if height > 0:
                    metadata["aspect_ratio"] = width / height

        except Exception:
            pass

        return metadata

    # =========================================================
    # VALIDATE
    # =========================================================

    def validate(
        self,
        validate_annotations: bool = True
    ) -> Dict[str, Any]:

        if not self.samples:
            self.load()

        valid = 0
        review = 0
        quarantine = 0

        issue_counts = Counter()

        for sample in self.samples:

            image_path = Path(sample["image"])

            annotation_path = (
                Path(sample["annotation"]) if sample["annotation"] else None
            )

            issues = []

            # IMAGE CHECKS
            if not image_path.exists():
                issues.append("missing_image")

            elif not self._is_valid_image(image_path):
                issues.append("corrupted_image")

            elif (
                image_path.suffix.lower()
                not in self.SUPPORTED_IMAGE_FORMATS
            ):
                issues.append("unsupported_image_format")

            # RESOLUTION
            metadata = sample["metadata"]

            width = metadata.get("width")
            height = metadata.get("height")

            if width is not None and width < self.min_width:
                issues.append("resolution_too_small")

            if height is not None and height < self.min_height:
                issues.append("resolution_too_small")

            # ANNOTATION
            if annotation_path is None:
                issues.append("missing_annotation")

            elif not annotation_path.exists():
                issues.append("missing_annotation")

            elif self._is_empty_annotation(annotation_path):
                issues.append("empty_label")

            elif validate_annotations:
                issues.extend(self._validate_annotation(annotation_path))

            # CLASSIFY
            status = self._classify_sample(issues)

            sample["issues"] = list(set(issues))
            sample["status"] = status

            issue_counts.update(sample["issues"])

            if status == "KEEP":
                valid += 1

            elif status == "REVIEW":
                review += 1

            else:
                quarantine += 1

        self.validation_result = {
            "total": len(self.samples),
            "valid": valid,
            "review": review,
            "quarantine": quarantine,
            "issue_counts": dict(issue_counts),
        }

        return self.validation_result

    # =========================================================
    # IMAGE VALIDATION
    # =========================================================

    def _is_valid_image(
        self,
        image_path: Path
    ) -> bool:

        if Image is None:
            return True

        try:

            with Image.open(image_path) as image:
                image.verify()

            return True

        except Exception:

            return False

    # =========================================================
    # ANNOTATION VALIDATION
    # =========================================================

    def _validate_annotation(
        self,
        annotation_path: Path
    ) -> List[str]:

        issues = set()

        if annotation_path.suffix.lower() != ".txt":
            return []

        has_lines = False

        # Streamed line by line instead of readlines().
        try:

            with open(annotation_path, "r", encoding="utf-8") as file:

                for line in file:

                    has_lines = True

                    parts = line.split()

                    if not parts:
                        continue

                    if len(parts) != 5:
                        issues.add("invalid_annotation_format")
                        continue

                    try:

                        class_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                    except ValueError:

                        issues.add("invalid_annotation_value")
                        continue

                    if class_id < 0:
                        issues.add("invalid_class_id")

                    if any(
                        value < 0 or value > 1
                        for value in (x, y, width, height)
                    ):
                        issues.add("annotation_out_of_bounds")

                    if width <= 0 or height <= 0:
                        issues.add("zero_area_annotation")

        except Exception:

            return ["annotation_read_error"]

        if not has_lines:
            return ["empty_label"]

        return list(issues)

    # =========================================================
    # CLASS EXTRACTION
    # =========================================================

    def _extract_classes(
        self,
        annotation_path: Optional[Path]
    ) -> List[str]:

        if annotation_path is None:
            return []

        if annotation_path.suffix.lower() != ".txt":
            return []

        classes = []

        try:

            with open(annotation_path, "r", encoding="utf-8") as file:

                for line in file:

                    parts = line.split()

                    if parts:
                        classes.append(parts[0])

        except Exception:

            return []

        return classes

    # =========================================================
    # EMPTY ANNOTATION
    # =========================================================

    def _is_empty_annotation(
        self,
        annotation_path: Path
    ) -> bool:

        try:
            return annotation_path.stat().st_size == 0

        except OSError:
            return True

    # =========================================================
    # SAMPLE CLASSIFICATION
    # =========================================================

    def _classify_sample(
        self,
        issues: List[str]
    ) -> str:

        if not issues:
            return "KEEP"

        quarantine_issues = {
            "missing_image",
            "corrupted_image",
            "unsupported_image_format",
            "invalid_annotation_format",
            "invalid_annotation_value",
            "annotation_read_error",
        }

        if any(issue in quarantine_issues for issue in issues):
            return "QUARANTINE"

        return "REVIEW"

    # =========================================================
    # EXACT DUPLICATES
    # =========================================================

    def detect_duplicates(
        self
    ) -> Dict[str, Any]:

        if not self.samples:
            self.load()

        # Only hash counts are needed, not the grouped samples.
        hash_counts = Counter()

        for sample in self.samples:

            image_path = Path(sample["image"])

            if not image_path.exists():
                continue

            try:

                file_hash = self._file_hash(image_path)

                sample["hash"] = file_hash

                hash_counts[file_hash] += 1

            except Exception:

                sample["status"] = "REVIEW"
                sample["issues"].append("hash_failed")

        group_count = 0
        duplicate_count = 0

        for count in hash_counts.values():

            if count > 1:
                group_count += 1
                duplicate_count += count - 1

        self.duplicate_result = {
            "group_count": group_count,
            "duplicate_count": duplicate_count,
        }

        return self.duplicate_result

    # =========================================================
    # FILE HASH
    # =========================================================

    def _file_hash(
        self,
        path: Path
    ) -> str:

        hasher = hashlib.sha256()

        # One reusable buffer instead of a new 1 MB bytes object
        # per chunk.
        buffer = bytearray(1024 * 1024)
        view = memoryview(buffer)

        with open(path, "rb") as file:

            while True:

                read = file.readinto(buffer)

                if not read:
                    break

                hasher.update(view[:read])

        return hasher.hexdigest()

    # =========================================================
    # PERCEPTUAL HASH
    # =========================================================

    def compute_perceptual_hash(
        self,
        image_path: Path
    ) -> Optional[str]:

        if Image is None:
            return None

        try:

            with Image.open(image_path) as image:

                pixels = image.convert("L").resize((8, 8)).tobytes()

            average = sum(pixels) / len(pixels)

            return "".join(
                "1" if pixel >= average else "0"
                for pixel in pixels
            )

        except Exception:

            return None

    # =========================================================
    # PERCEPTUAL DUPLICATES
    # =========================================================

    def detect_perceptual_duplicates(
        self,
        max_distance: int = 5
    ) -> Dict[str, Any]:

        if not self.samples:
            self.load()

        hashes = []
        values = []

        for sample in self.samples:

            image_path = Path(sample["image"])

            if not image_path.exists():
                continue

            phash = self.compute_perceptual_hash(image_path)

            sample["perceptual_hash"] = phash

            if phash is not None:

                hashes.append(sample)

                # Integer form: distance = XOR + bit count
                values.append(int(phash, 2))

        count = len(hashes)

        groups = []
        visited = bytearray(count)

        for index in range(count):

            if visited[index]:
                continue

            current_group = [index]

            visited[index] = 1

            first = values[index]

            for other_index in range(index + 1, count):

                if visited[other_index]:
                    continue

                if _popcount(first ^ values[other_index]) <= max_distance:

                    current_group.append(other_index)
                    visited[other_index] = 1

            if len(current_group) > 1:

                groups.append({"count": len(current_group)})

                for member in current_group:

                    item = hashes[member]

                    if "near_duplicate" not in item["issues"]:

                        item["issues"].append("near_duplicate")

                        if item["status"] == "KEEP":
                            item["status"] = "REVIEW"

        result = {
            "groups": groups,
            "group_count": len(groups),
        }

        if self.duplicate_result is None:
            self.duplicate_result = {}

        self.duplicate_result["perceptual_duplicates"] = result

        return result

    # =========================================================
    # HASH DISTANCE
    # =========================================================

    def _hash_distance(
        self,
        hash_a: str,
        hash_b: str
    ) -> int:

        if not hash_a or not hash_b:
            return 999999

        return sum(
            bit_a != bit_b
            for bit_a, bit_b in zip(hash_a, hash_b)
        )

    # =========================================================
    # QUALITY CHECK
    # =========================================================

    def check_quality(
        self
    ) -> Dict[str, Any]:

        if not self.samples:
            self.load()

        results = {
            "blurred": [],
            "dark": [],
            "bright": [],
            "small": [],
        }

        for sample in self.samples:

            image_path = Path(sample["image"])

            if not image_path.exists():
                continue

            if Image is None:
                continue

            try:

                with Image.open(image_path) as image:

                    gray = image.convert("L")

                    width, height = image.size

                brightness = ImageStat.Stat(gray).mean[0]

                if brightness < self.dark_threshold:

                    sample["issues"].append("dark_image")
                    results["dark"].append(sample["image"])

                elif brightness > self.bright_threshold:

                    sample["issues"].append("bright_image")
                    results["bright"].append(sample["image"])

                if width < self.min_width or height < self.min_height:

                    sample["issues"].append("low_resolution")
                    results["small"].append(sample["image"])

                if self._is_blurred(gray):

                    sample["issues"].append("blurred_image")
                    results["blurred"].append(sample["image"])

                del gray

            except Exception:

                sample["issues"].append("quality_check_failed")

                if sample["status"] == "KEEP":
                    sample["status"] = "REVIEW"

        quality_issues = {
            "dark_image",
            "bright_image",
            "blurred_image",
            "low_resolution",
        }

        for sample in self.samples:

            if sample["status"] == "KEEP" and any(
                issue in quality_issues for issue in sample["issues"]
            ):
                sample["status"] = "REVIEW"

        self.quality_result = {
            "counts": {key: len(value) for key, value in results.items()},
            "samples": results,
        }

        return self.quality_result

    # =========================================================
    # BLUR DETECTION
    # =========================================================

    def _is_blurred(
        self,
        gray_image
    ) -> bool:

        try:

            if ImageFilter is None:
                return False

            edges = gray_image.filter(ImageFilter.FIND_EDGES)

            return ImageStat.Stat(edges).var[0] < self.blur_threshold

        except Exception:

            return False

    # =========================================================
    # LEAKAGE DETECTION
    # =========================================================

    def detect_leakage(
        self
    ) -> Dict[str, Any]:

        if not self.samples:
            self.load()

        if self.duplicate_result is None:
            self.detect_duplicates()

        # Only the number of distinct groups is reported.
        sources = set()
        videos = set()

        for sample in self.samples:

            source = sample.get("source")
            video = sample.get("video")

            if source:
                sources.add(source)

            if video:
                videos.add(video)

        result = {
            "duplicate_count": self.duplicate_result.get(
                "duplicate_count", 0
            ),
            "source_group_count": len(sources),
            "video_group_count": len(videos),
            "warnings": [],
        }

        if videos:

            result["warnings"].append(
                "Video frames detected. "
                "Frames from the same video "
                "should remain in the same split."
            )

        if sources:

            result["warnings"].append(
                "Multiple source groups detected. "
                "Source-aware splitting is recommended."
            )

        self.leakage_result = result

        return result

    # =========================================================
    # CLEAN
    # =========================================================

    def clean(
        self,
        quarantine: bool = False
    ) -> Dict[str, Any]:

        if self.validation_result is None:
            self.validate()

        kept = 0
        review = 0
        quarantined = 0

        quarantine_dir = self.path / "_quarantine"

        if quarantine:
            quarantine_dir.mkdir(parents=True, exist_ok=True)

        for sample in self.samples:

            status = sample["status"]

            if status == "KEEP":
                kept += 1

            elif status == "REVIEW":
                review += 1

            elif status == "QUARANTINE":

                quarantined += 1

                if quarantine:
                    self._quarantine_sample(sample, quarantine_dir)

        self.cleaning_result = {
            "kept": kept,
            "review": review,
            "quarantined": quarantined,
            "quarantine_enabled": quarantine,
        }

        return self.cleaning_result

    # =========================================================
    # QUARANTINE
    # =========================================================

    def _quarantine_sample(
        self,
        sample: Dict[str, Any],
        quarantine_dir: Path
    ):

        image_path = Path(sample["image"])

        if not image_path.exists():
            return

        destination = quarantine_dir / image_path.name

        counter = 1

        while destination.exists():

            destination = (
                quarantine_dir
                / f"{image_path.stem}_{counter}{image_path.suffix}"
            )

            counter += 1

        try:

            shutil.move(str(image_path), str(destination))

            sample["image"] = str(destination)

        except OSError:

            sample["issues"].append("quarantine_failed")

    # =========================================================
    # CREATE GROUPS FOR SPLITTING
    # =========================================================

    def _create_split_groups(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:

        groups = defaultdict(list)

        for sample in samples:

            video = sample.get("video")
            source = sample.get("source")

            if video:
                group_id = f"video::{video}"

            elif source:
                group_id = f"source::{source}"

            else:
                # Each image becomes an independent group.
                group_id = f"image::{sample['image']}"

            groups[group_id].append(sample)

        return groups

    # =========================================================
    # YAML SPLIT CONFIGURATION
    # =========================================================

    def _get_split_config(
        self
    ):

        split_config = self.config.get("split", {})

        if not isinstance(split_config, dict):
            split_config = {}

        train_ratio = split_config.get("train", 0.70)

        validation_ratio = split_config.get(
            "val",
            split_config.get("validation", 0.15)
        )

        test_ratio = split_config.get("test", 0.15)

        enabled = split_config.get("enabled", True)

        return {
            "enabled": bool(enabled),
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(test_ratio),
        }

    # =========================================================
    # EXISTING SPLIT DETECTION
    # =========================================================

    def _get_existing_splits(
        self
    ):

        result = {
            "train": None,
            "val": None,
            "test": None,
        }

        # ONLY ACCEPT SPLITS WITH BOTH images/ AND labels/
        candidates = {
            "train": [self.path / "train"],
            "val": [
                self.path / "val",
                self.path / "val",
                self.path / "val",
            ],
            "test": [self.path / "test"],
        }

        for split_name, paths in candidates.items():

            for root in paths:

                images = root / "images"
                labels = root / "labels"

                if (
                    images.exists()
                    and images.is_dir()
                    and labels.exists()
                    and labels.is_dir()
                ):

                    # Must contain at least one image
                    # (stops at the first one found).
                    has_image = any(
                        file.is_file()
                        and file.suffix.lower()
                        in self.SUPPORTED_IMAGE_FORMATS
                        for file in images.rglob("*")
                    )

                    if has_image:

                        result[split_name] = root

                        break

        return result

    # =========================================================
    # CHECK EXISTING SPLITS
    # =========================================================

    def _existing_splits_are_valid(
        self,
        splits
    ):

        for name in ("train", "validation", "test"):

            path = splits.get(name)

            if path is None:
                return False

            path = Path(path)

            if (
                not (path / "images").exists()
                or not (path / "labels").exists()
            ):
                return False

        return True

    # =========================================================
    # COUNT SPLIT IMAGES
    # =========================================================

    def _count_split_images(
        self,
        split_path
    ):

        if split_path is None:
            return 0

        images_path = Path(split_path) / "images"

        if not images_path.exists():
            return 0

        return sum(
            1
            for file in images_path.rglob("*")
            if file.is_file()
            and file.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS
        )

    # =========================================================
    # SPLIT
    # =========================================================

    def split(
        self,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42
    ):
        """
        Split dataset into YOLO-compatible train/valid/test folders.

            prepared/
            ├── train/{images,labels}
            ├── valid/{images,labels}
            ├── test/{images,labels}
            └── data.yaml

        - Uses `valid`, NOT `validation`
        - Image/label pairs are kept together
        - Does not modify the original dataset
        """

        # VALIDATE RATIOS
        total_ratio = train_ratio + val_ratio + test_ratio

        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0"
            )

        if train_ratio <= 0:
            raise ValueError("train_ratio must be greater than 0")

        if val_ratio < 0 or test_ratio < 0:
            raise ValueError("val_ratio and test_ratio cannot be negative")

        # SOURCE DATASET
        source_path = Path(self.path)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Dataset does not exist: {source_path}"
            )

        # OUTPUT DIRECTORY
        output_path = getattr(self, "output_path", None)

        if output_path is None:
            output_path = (
                source_path.parent / f"{source_path.name}_prepared"
            )

        output_path = Path(output_path)

        # Remove old incorrect validation folder.
        old_validation = output_path / "validation"

        if old_validation.exists():
            shutil.rmtree(old_validation)

        # CREATE ONLY YOLO DIRECTORIES
        for split_name in ("train", "valid", "test"):

            (output_path / split_name / "images").mkdir(
                parents=True, exist_ok=True
            )

            (output_path / split_name / "labels").mkdir(
                parents=True, exist_ok=True
            )

        # FIND IMAGE/LABEL PAIRS (single pass, no intermediate list)
        image_extensions = self.SUPPORTED_IMAGE_FORMATS

        # Never read images from already-generated folders.
        ignored_directories = {
            "train", "valid", "validation", "test", "prepared"
        }

        pairs = []
        total_found = 0
        missing_labels = 0

        for image in source_path.rglob("*"):

            if image.suffix.lower() not in image_extensions:
                continue

            if not image.is_file():
                continue

            relative_parts = {
                part.lower()
                for part in image.relative_to(source_path).parts[:-1]
            }

            if relative_parts.intersection(ignored_directories):
                continue

            total_found += 1

            label = image.with_suffix(".txt")

            # Standard case
            if label.exists():

                pairs.append((image, label))

                continue

            # Search label directories
            found_label = None

            for candidate in (
                source_path / "labels" / f"{image.stem}.txt",
                image.parent / "labels" / f"{image.stem}.txt",
            ):

                if candidate.exists():

                    found_label = candidate

                    break

            if found_label is not None:
                pairs.append((image, found_label))

            else:
                missing_labels += 1

        # SHUFFLE (same order as random.seed + random.shuffle, but
        # without changing the global random state)
        random.Random(seed).shuffle(pairs)

        total_images = len(pairs)

        # CALCULATE SPLITS
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)

        # Everything remaining goes to test.
        train_pairs = pairs[:train_count]
        valid_pairs = pairs[train_count:train_count + val_count]
        test_pairs = pairs[train_count + val_count:]

        # COPY
        def copy_pairs(pair_list, split_name):

            image_dir = output_path / split_name / "images"
            label_dir = output_path / split_name / "labels"

            copied = 0

            for image, label in pair_list:

                shutil.copy2(image, image_dir / image.name)

                shutil.copy2(label, label_dir / f"{image.stem}.txt")

                copied += 1

            return copied

        train_copied = copy_pairs(train_pairs, "train")
        valid_copied = copy_pairs(valid_pairs, "valid")
        test_copied = copy_pairs(test_pairs, "test")

        # CREATE YOLO DATA.YAML
        yaml_path = output_path / "data.yaml"

        names = []

        config = getattr(self, "config", None)

        if isinstance(config, dict):
            names = config.get("names", [])

        # Try existing data.yaml
        if not names and yaml is not None:

            possible_yaml = source_path / "data.yaml"

            if possible_yaml.exists():

                try:

                    with open(possible_yaml, "r", encoding="utf-8") as f:
                        yaml_data = yaml.safe_load(f)

                    if isinstance(yaml_data, dict):
                        names = yaml_data.get("names", [])

                except Exception:
                    pass

        if isinstance(names, dict):

            names = [
                names[key]
                for key in sorted(
                    names.keys(),
                    key=lambda x: int(x) if str(x).isdigit() else str(x)
                )
            ]

        if not names:
            names = ["class_0"]

        # WRITE YAML
        yaml_content = [
            f"path: {output_path.as_posix()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(names)}",
            "names:",
        ]

        for index, name in enumerate(names):
            yaml_content.append(f"  {index}: {name}")

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_content))

        # SAVE RESULT
        self.split_result = {
            "status": "SUCCESS",
            "output": str(output_path),
            "total_images": total_found,
            "paired_images": total_images,
            "missing_labels": missing_labels,
            "train": {"images": train_copied, "labels": train_copied},
            "valid": {"images": valid_copied, "labels": valid_copied},
            "test": {"images": test_copied, "labels": test_copied},
            "yaml": str(yaml_path),
            "directories": [
                "train/images",
                "train/labels",
                "valid/images",
                "valid/labels",
                "test/images",
                "test/labels"
            ]
        }

        return self.split_result

    # =========================================================
    # COPY SAMPLES
    # =========================================================

    def _copy_samples(
        self,
        samples: List[Dict[str, Any]],
        destination
    ):
        """
        Copy images and annotations separately:

            images/image.jpg
            labels/image.txt
        """

        images_dir = Path(destination["images"])
        labels_dir = Path(destination["labels"])

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:

            image_path = Path(sample["image"])

            if not image_path.exists():
                continue

            target_image = self._unique_path(images_dir / image_path.name)

            try:
                shutil.copy2(image_path, target_image)

            except OSError:
                continue

            annotation = sample.get("annotation")

            if not annotation:
                continue

            annotation_path = Path(annotation)

            if not annotation_path.exists():
                continue

            # Label always uses the final image stem
            target_label = labels_dir / f"{target_image.stem}.txt"

            try:
                shutil.copy2(annotation_path, target_label)

            except OSError:
                continue

    # =========================================================
    # UNIQUE PATH
    # =========================================================

    def _unique_path(
        self,
        path: Path
    ) -> Path:

        if not path.exists():
            return path

        counter = 1

        while True:

            candidate = path.parent / f"{path.stem}_{counter}{path.suffix}"

            if not candidate.exists():
                return candidate

            counter += 1

    # =========================================================
    # CREATE YOLO YAML
    # =========================================================

    def _create_yolo_yaml(
        self,
        output_path: Path
    ) -> Path:

        yaml_path = output_path / "data.yaml"

        # Determine class IDs
        class_ids = set()

        for sample in self.samples:

            for class_id in sample.get("classes", []):

                try:
                    class_ids.add(int(class_id))

                except (ValueError, TypeError):
                    pass

        # Class names
        configured_names = self.config.get("names")

        if configured_names is None:
            configured_names = self.config.get("classes")

        if isinstance(configured_names, dict):

            max_id = max(class_ids if class_ids else {0})

            names = [
                configured_names.get(
                    str(i),
                    configured_names.get(i, f"class_{i}")
                )
                for i in range(max_id + 1)
            ]

        elif isinstance(configured_names, list):

            names = configured_names

        else:

            if class_ids:

                names = [f"class_{i}" for i in range(max(class_ids) + 1)]

            else:

                names = ["object"]

        output_path.mkdir(parents=True, exist_ok=True)

        data = {
            "path": str(output_path.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(names),
            "names": names,
        }

        if yaml is not None:

            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(data, file, sort_keys=False)

        else:

            lines = [
                f"path: '{output_path.resolve()}'",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                f"nc: {len(names)}",
                "names:",
            ]

            for index, name in enumerate(names):
                lines.append(f"  {index}: '{name}'")

            with open(yaml_path, "w", encoding="utf-8") as file:
                file.write("\n".join(lines))

        return yaml_path

    # =========================================================
    # MANIFEST
    # =========================================================

    def create_manifest(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:

        if output_path is None:
            output_file = self.path / "dataset_manifest.json"

        else:
            output_file = Path(output_path)

        manifest = {
            "dataset_path": str(self.path),
            "seed": self.seed,
            "sample_count": len(self.samples),
            "samples": self.samples,
        }

        try:

            with open(output_file, "w", encoding="utf-8") as file:

                json.dump(manifest, file, indent=4, default=str)

        except OSError as error:

            raise RuntimeError(f"Failed to create manifest: {error}")

        self.manifest_result = {
            "path": str(output_file),
            "sample_count": len(self.samples),
        }

        return self.manifest_result

    # =========================================================
    # REPORT
    # =========================================================

    def report(
        self
    ) -> Dict[str, Any]:

        status_counts = Counter(sample["status"] for sample in self.samples)

        class_counts = Counter()

        for sample in self.samples:
            class_counts.update(sample.get("classes", []))

        return {
            "dataset": str(self.path),
            "total_samples": len(self.samples),
            "status": dict(status_counts),
            "class_distribution": dict(class_counts),
            "validation": self.validation_result,
            "duplicates": self.duplicate_result,
            "quality": self.quality_result,
            "leakage": self.leakage_result,
            "cleaning": self.cleaning_result,
            "splitting": self.split_result,
            "manifest": self.manifest_result,
        }

    # =========================================================
    # COMPLETE PREPARATION
    # =========================================================

    def prepare(
        self,
        quarantine: bool = False,
        detect_duplicates: bool = True,
        detect_perceptual_duplicates: bool = True,
        check_quality: bool = True,
        detect_leakage: bool = True,
        validate_annotations: bool = True,
        create_manifest: bool = True,
        **split_kwargs
    ) -> Dict[str, Any]:

        self.load()

        self.validate(validate_annotations=validate_annotations)

        if detect_duplicates:
            self.detect_duplicates()

        if detect_perceptual_duplicates:
            self.detect_perceptual_duplicates()

        if check_quality:
            self.check_quality()

        if detect_leakage:
            self.detect_leakage()

        self.clean(quarantine=quarantine)

        self.split(**split_kwargs)

        if create_manifest:
            self.create_manifest()

        self.preparation_result = self.report()

        self.preparation_result["status"] = "prepared"

        return self.preparation_result