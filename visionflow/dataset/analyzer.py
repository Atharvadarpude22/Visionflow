from pathlib import Path
from typing import Optional, Any, Dict, List
from collections import Counter, defaultdict
import hashlib
import math

try:
    from PIL import Image, ImageStat, ImageFilter
except ImportError:
    Image = None
    ImageStat = None
    ImageFilter = None


class DatasetAnalyzer:
    """
    Analyzes what a computer vision dataset actually contains.

    profile()            General dataset statistics.
    analyze_classes()    Class distribution.
    analyze_quality()    Image quality problems.
    analyze_coverage()   Coverage across visual conditions.
    detect_redundancy()  Exact, near, and temporal redundancy.
    detect_leakage()     Potential train/validation/test leakage.
    report()             Complete dataset analysis report.
    """

    SUPPORTED_IMAGE_FORMATS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    }

    SUPPORTED_ANNOTATION_FORMATS = {
        ".txt", ".xml", ".json", ".yaml", ".yml",
    }

    def __init__(
        self,
        path: str,
        config: Optional[dict] = None
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

        # Cached data
        self.images: List[Path] = []
        self.annotations: List[Path] = []

        # Analysis results
        self.profile_result = None
        self.class_result = None
        self.quality_result = None
        self.coverage_result = None
        self.redundancy_result = None
        self.leakage_result = None

        self.report_result = None

        # Internal caches (avoid re-reading files across analyses)
        self._class_cache = None
        self._brightness_cache = {}

    # =========================================================
    # INTERNAL DISCOVERY
    # =========================================================

    def _discover_images(self) -> List[Path]:
        """Find all supported image files."""

        return sorted(
            file
            for file in self.path.rglob("*")
            if file.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS
            and file.is_file()
            and "_quarantine" not in file.parts
        )

    def _discover_annotations(self) -> List[Path]:
        """Find all supported annotation files."""

        return sorted(
            file
            for file in self.path.rglob("*")
            if file.suffix.lower() in self.SUPPORTED_ANNOTATION_FORMATS
            and file.is_file()
            and "_quarantine" not in file.parts
        )

    def _ensure_discovered(self):
        """Discover files once (single directory walk for both types)."""

        need_images = not self.images
        need_annotations = not self.annotations

        if not (need_images or need_annotations):
            return

        images = []
        annotations = []

        image_formats = self.SUPPORTED_IMAGE_FORMATS
        annotation_formats = self.SUPPORTED_ANNOTATION_FORMATS

        for file in self.path.rglob("*"):

            suffix = file.suffix.lower()

            if suffix in image_formats:
                if not need_images:
                    continue
                target = images

            elif suffix in annotation_formats:
                if not need_annotations:
                    continue
                target = annotations

            else:
                continue

            if file.is_file() and "_quarantine" not in file.parts:
                target.append(file)

        if need_images:
            self.images = sorted(images)

        if need_annotations:
            self.annotations = sorted(annotations)
            self._class_cache = None

    # =========================================================
    # PROFILE
    # =========================================================

    def profile(self) -> Dict[str, Any]:
        """
        Generate a general dataset profile.

        Calculates: images, resolutions, file sizes, formats,
        annotations, classes.
        """

        self._ensure_discovered()

        resolutions = Counter()
        formats = Counter()

        # Running statistics instead of storing every file size
        size_total = 0
        size_count = 0
        size_min = 0
        size_max = 0

        valid_images = 0
        corrupted_images = 0

        for image_path in self.images:

            formats[image_path.suffix.lower()] += 1

            try:

                size = image_path.stat().st_size

                size_total += size

                if size_count == 0:
                    size_min = size_max = size
                else:
                    if size < size_min:
                        size_min = size
                    if size > size_max:
                        size_max = size

                size_count += 1

                if Image is not None:

                    with Image.open(image_path) as image:
                        resolutions[image.size] += 1

                valid_images += 1

            except Exception:

                corrupted_images += 1

        result = {
            "dataset_path": str(self.path),

            "number_of_images": len(self.images),

            "number_of_annotations": len(self.annotations),

            "valid_images": valid_images,

            "corrupted_images": corrupted_images,

            "formats": dict(formats),

            "resolutions": {
                f"{width}x{height}": count
                for (width, height), count in resolutions.items()
            },

            "file_size": {
                "total_bytes": size_total,
                "average_bytes": (
                    size_total / size_count if size_count else 0
                ),
                "minimum_bytes": size_min if size_count else 0,
                "maximum_bytes": size_max if size_count else 0
            },

            "classes": self._extract_classes()
        }

        self.profile_result = result

        return result

    # =========================================================
    # CLASS ANALYSIS
    # =========================================================

    def analyze_classes(self) -> Dict[str, Any]:
        """Analyze class distribution."""

        self._ensure_discovered()

        class_counts = self._extract_classes()

        total = sum(class_counts.values())

        distribution = {}

        for class_name, count in class_counts.items():

            percentage = (count / total) * 100 if total > 0 else 0

            distribution[class_name] = {
                "count": count,
                "percentage": round(percentage, 2)
            }

        result = {
            "total_annotations": total,
            "number_of_classes": len(class_counts),
            "distribution": distribution
        }

        if class_counts:

            most_common = max(class_counts, key=class_counts.get)
            least_common = min(class_counts, key=class_counts.get)

            result["most_common_class"] = {
                "name": most_common,
                "count": class_counts[most_common]
            }

            result["least_common_class"] = {
                "name": least_common,
                "count": class_counts[least_common]
            }

        else:

            result["most_common_class"] = None
            result["least_common_class"] = None

        self.class_result = result

        return result

    # =========================================================
    # EXTRACT CLASSES
    # =========================================================

    def _extract_classes(self) -> Dict[str, int]:
        """
        Extract class information from annotations.

        Supports simple YOLO-style TXT labels:

            class_id x_center y_center width height
        """

        cache = self._class_cache

        if cache is not None and cache[0] is self.annotations:
            counts = Counter(cache[1])

        else:

            counts = Counter()

            for annotation in self.annotations:

                if annotation.suffix.lower() != ".txt":
                    continue

                try:

                    with open(annotation, "r", encoding="utf-8") as file:

                        for line in file:

                            parts = line.split()

                            if parts:
                                counts[parts[0]] += 1

                except (OSError, UnicodeDecodeError):
                    continue

            self._class_cache = (self.annotations, dict(counts))

        names = self.config.get("class_names", {})

        if names:

            renamed = Counter()

            for class_id, count in counts.items():

                renamed[names.get(str(class_id), str(class_id))] += count

            return dict(renamed)

        return dict(counts)

    # =========================================================
    # QUALITY ANALYSIS
    # =========================================================

    def analyze_quality(self) -> Dict[str, Any]:
        """
        Analyze image quality: blur, dark images, corrupt images,
        extreme resolution.
        """

        self._ensure_discovered()

        if Image is None:

            return {
                "status": "limited",
                "reason": "Pillow is not installed",
                "blurred": [],
                "dark": [],
                "extreme_resolution": [],
                "corrupt": []
            }

        config = self.config

        blur_threshold = config.get("blur_threshold", 20.0)
        dark_threshold = config.get("dark_threshold", 40.0)
        min_width = config.get("min_width", 64)
        min_height = config.get("min_height", 64)
        max_width = config.get("max_width", 8192)
        max_height = config.get("max_height", 8192)

        blurred = []
        dark = []
        extreme_resolution = []
        corrupt = []

        cache = self._brightness_cache

        for image_path in self.images:

            try:

                with Image.open(image_path) as image:

                    # convert() fully decodes the image, so corrupt
                    # files raise here, before anything is recorded.
                    grayscale = image.convert("L")

                    width, height = image.size

                if (
                    width < min_width
                    or height < min_height
                    or width > max_width
                    or height > max_height
                ):
                    extreme_resolution.append(str(image_path))

                brightness = ImageStat.Stat(grayscale).mean[0]

                cache[image_path] = ((width, height), brightness)

                if brightness < dark_threshold:

                    dark.append({
                        "image": str(image_path),
                        "brightness": brightness
                    })

                variance = self._laplacian_variance(grayscale)

                del grayscale

                if variance < blur_threshold:

                    blurred.append({
                        "image": str(image_path),
                        "blur_score": variance
                    })

            except Exception:

                corrupt.append(str(image_path))

        result = {
            "total_images": len(self.images),

            "blur": {
                "count": len(blurred),
                "images": blurred
            },

            "dark_images": {
                "count": len(dark),
                "images": dark
            },

            "corrupt_files": {
                "count": len(corrupt),
                "images": corrupt
            },

            "extreme_resolution": {
                "count": len(extreme_resolution),
                "images": extreme_resolution
            }
        }

        self.quality_result = result

        return result

    # =========================================================
    # BLUR DETECTION
    # =========================================================

    def _laplacian_variance(self, image):
        """
        Estimate image sharpness (edge-variance proxy).

        Higher value generally means sharper image.
        """

        try:

            return ImageStat.Stat(
                image.filter(ImageFilter.FIND_EDGES)
            ).var[0]

        except Exception:

            return 0.0

    # =========================================================
    # COVERAGE ANALYSIS
    # =========================================================

    def analyze_coverage(self) -> Dict[str, Any]:
        """
        Analyze dataset coverage: lighting, environment, object size,
        camera/source, conditions.
        """

        self._ensure_discovered()

        lighting = Counter()
        environments = Counter()
        sources = Counter()
        resolutions = Counter()

        cache = self._brightness_cache

        for image_path in self.images:

            sources[self._infer_source(image_path)] += 1

            if Image is None:
                continue

            # Reuse size/brightness if analyze_quality already
            # decoded this image.
            hit = cache.get(image_path)

            if hit is None:

                try:

                    with Image.open(image_path) as image:

                        size = image.size

                        brightness = ImageStat.Stat(
                            image.convert("L")
                        ).mean[0]

                except Exception:
                    continue

                hit = cache[image_path] = (size, brightness)

            (width, height), brightness = hit

            resolutions[f"{width}x{height}"] += 1

            if brightness < 40:
                lighting["dark"] += 1

            elif brightness < 100:
                lighting["low_light"] += 1

            elif brightness < 180:
                lighting["normal"] += 1

            else:
                lighting["bright"] += 1

        result = {
            "lighting": dict(lighting),

            "environment": dict(environments),

            "camera_source": dict(sources),

            "resolution": dict(resolutions),

            "object_size": {
                "status": (
                    "requires_annotations"
                    if self.annotations
                    else "unavailable"
                )
            },

            "conditions": {}
        }

        self.coverage_result = result

        return result

    # =========================================================
    # REDUNDANCY
    # =========================================================

    def detect_redundancy(
        self,
        near_duplicate_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Detect redundant samples: exact duplicates, near duplicates,
        temporal redundancy.
        """

        self._ensure_discovered()

        # -----------------------------------------------------
        # Exact duplicates
        # Only files sharing a size with another file can be
        # identical, so unique-size files are never hashed.
        # -----------------------------------------------------

        by_size = defaultdict(list)

        for image_path in self.images:

            try:
                by_size[image_path.stat().st_size].append(image_path)

            except OSError:
                continue

        candidates = {
            path
            for paths in by_size.values()
            if len(paths) > 1
            for path in paths
        }

        del by_size

        hashes = defaultdict(list)

        for image_path in self.images:

            if image_path not in candidates:
                continue

            try:

                hashes[self._file_hash(image_path)].append(
                    str(image_path)
                )

            except OSError:
                continue

        exact_duplicates = [
            paths for paths in hashes.values() if len(paths) > 1
        ]

        del hashes, candidates

        # -----------------------------------------------------
        # Near duplicates (bounded pairwise comparison)
        # -----------------------------------------------------

        near_duplicates = []

        max_comparisons = self.config.get("max_duplicate_comparisons", 10000)

        if Image is not None:

            comparisons = 0

            image_hashes = {}

            for image_path in self.images:

                try:
                    image_hashes[image_path] = self._average_hash(
                        image_path
                    )

                except Exception:
                    continue

                # Beyond max_comparisons + 1 hashes, extra images
                # can never be reached by the bounded comparison.
                if len(image_hashes) > max_comparisons:
                    break

            image_list = list(image_hashes)
            count = len(image_list)
            similarity_of = self._hash_similarity

            for i in range(count):

                if comparisons >= max_comparisons:
                    break

                first = image_list[i]
                first_hash = image_hashes[first]

                for j in range(i + 1, count):

                    if comparisons >= max_comparisons:
                        break

                    comparisons += 1

                    second = image_list[j]

                    similarity = similarity_of(
                        first_hash,
                        image_hashes[second]
                    )

                    if similarity >= near_duplicate_threshold:

                        near_duplicates.append({
                            "image_1": str(first),
                            "image_2": str(second),
                            "similarity": similarity
                        })

        # -----------------------------------------------------
        # Temporal redundancy
        # -----------------------------------------------------

        temporal_groups = self._detect_temporal_groups()

        result = {
            "exact_duplicates": {
                "count": len(exact_duplicates),
                "groups": exact_duplicates
            },

            "near_duplicates": {
                "count": len(near_duplicates),
                "pairs": near_duplicates
            },

            "temporal_redundancy": {
                "count": len(temporal_groups),
                "groups": temporal_groups
            }
        }

        self.redundancy_result = result

        return result

    # =========================================================
    # FILE HASH
    # =========================================================

    def _file_hash(
        self,
        path: Path
    ) -> str:

        hasher = hashlib.sha256()

        # One reusable buffer instead of allocating a new
        # 1 MB bytes object per chunk.
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
    # IMAGE HASH
    # =========================================================

    def _average_hash(
        self,
        image_path: Path,
        size: int = 8
    ) -> str:

        with Image.open(image_path) as image:

            pixels = image.convert("L").resize((size, size)).tobytes()

        average = sum(pixels) / len(pixels)

        return "".join(
            "1" if pixel >= average else "0"
            for pixel in pixels
        )

    # =========================================================
    # HASH SIMILARITY
    # =========================================================

    def _hash_similarity(
        self,
        first: str,
        second: str
    ) -> float:

        if len(first) != len(second):
            return 0.0

        if not first:
            return 1.0

        same = sum(a == b for a, b in zip(first, second))

        return same / len(first)

    # =========================================================
    # TEMPORAL REDUNDANCY
    # =========================================================

    def _detect_temporal_groups(self):
        """
        Detect likely video/frame groups (conservative).

        Images in the same source directory are grouped together.
        """

        groups = defaultdict(list)

        # self.images is already sorted, so every group stays sorted.
        for image_path in self.images:
            groups[self._infer_source(image_path)].append(image_path)

        return [
            {
                "source": source,
                "count": len(images),
                "images": [str(path) for path in images]
            }
            for source, images in groups.items()
            if len(images) >= 2
        ]

    # =========================================================
    # LEAKAGE DETECTION
    # =========================================================

    def detect_leakage(self) -> Dict[str, Any]:
        """
        Detect possible dataset leakage: duplicate, scene, video,
        temporal.
        """

        self._ensure_discovered()

        split_images = self._discover_split_images()

        # -----------------------------------------------------
        # Duplicate leakage (only same-size files can match)
        # -----------------------------------------------------

        duplicate_leakage = []

        if split_images:

            entries = []
            size_counts = Counter()

            for split_name, images in split_images.items():

                for image_path in images:

                    try:
                        size = image_path.stat().st_size

                    except OSError:
                        continue

                    entries.append((split_name, image_path, size))
                    size_counts[size] += 1

            split_hashes = {}

            for split_name, image_path, size in entries:

                if size_counts[size] < 2:
                    continue

                try:

                    image_hash = self._file_hash(image_path)

                    previous = split_hashes.get(image_hash)

                    if (
                        previous is not None
                        and previous["split"] != split_name
                    ):

                        duplicate_leakage.append({
                            "image": str(image_path),
                            "split_1": previous["split"],
                            "split_2": split_name
                        })

                    else:

                        split_hashes[image_hash] = {
                            "split": split_name,
                            "image": str(image_path)
                        }

                except OSError:
                    continue

            del entries, split_hashes

        # -----------------------------------------------------
        # Video/source leakage
        # -----------------------------------------------------

        source_splits = defaultdict(set)

        for split_name, images in split_images.items():

            for image_path in images:

                source_splits[
                    self._infer_source(image_path)
                ].add(split_name)

        video_leakage = [
            {"source": source, "splits": sorted(splits)}
            for source, splits in source_splits.items()
            if len(splits) > 1
        ]

        del source_splits

        scene_leakage = self._detect_scene_leakage(split_images)

        temporal_leakage = self._detect_temporal_leakage(split_images)

        result = {
            "duplicate_leakage": {
                "count": len(duplicate_leakage),
                "cases": duplicate_leakage
            },

            "scene_leakage": {
                "count": len(scene_leakage),
                "cases": scene_leakage
            },

            "video_leakage": {
                "count": len(video_leakage),
                "cases": video_leakage
            },

            "temporal_leakage": {
                "count": len(temporal_leakage),
                "cases": temporal_leakage
            }
        }

        self.leakage_result = result

        return result

    # =========================================================
    # DISCOVER SPLITS
    # =========================================================

    def _discover_split_images(self):

        result = {}

        for split in ("train", "validation", "val", "test"):

            for directory in (
                self.path / split,
                self.path / "prepared" / split
            ):

                if not directory.exists():
                    continue

                images = [
                    file
                    for file in directory.rglob("*")
                    if file.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS
                    and file.is_file()
                ]

                if images:

                    result["validation" if split == "val" else split] = images

                    break

        return result

    # =========================================================
    # SCENE LEAKAGE
    # =========================================================

    def _detect_scene_leakage(
        self,
        split_images
    ):
        """
        Approximate scene leakage detection using perceptual
        hashes across different splits.
        """

        if Image is None or not split_images:
            return []

        threshold = self.config.get("scene_similarity_threshold", 0.95)

        # One block per split; hashes stored as ints so similarity
        # is a single XOR + bit count.
        blocks = []

        for split_name, images in split_images.items():

            block = []

            for image_path in images:

                try:

                    block.append((
                        image_path,
                        self._average_hash(image_path)
                    ))

                except Exception:
                    continue

            blocks.append((
                split_name,
                [(path, int(h, 2), len(h)) for path, h in block]
            ))

        leakage = []

        # Same pair order as a flat i < j scan, skipping same-split pairs.
        for a in range(len(blocks)):

            split_a, items_a = blocks[a]

            for image_a, hash_a, length_a in items_a:

                for b in range(a + 1, len(blocks)):

                    split_b, items_b = blocks[b]

                    for image_b, hash_b, length_b in items_b:

                        if length_a != length_b:
                            similarity = 0.0

                        elif not length_a:
                            similarity = 1.0

                        else:
                            similarity = (
                                length_a - (hash_a ^ hash_b).bit_count()
                            ) / length_a

                        if similarity >= threshold:

                            leakage.append({
                                "image_1": str(image_a),
                                "image_2": str(image_b),
                                "split_1": split_a,
                                "split_2": split_b,
                                "similarity": similarity
                            })

        return leakage

    # =========================================================
    # TEMPORAL LEAKAGE
    # =========================================================

    def _detect_temporal_leakage(
        self,
        split_images
    ):
        """
        Detect source/frame leakage across splits.
        """

        source_splits = defaultdict(set)
        source_images = defaultdict(list)

        for split_name, images in split_images.items():

            for image_path in images:

                source = self._infer_source(image_path)

                source_splits[source].add(split_name)
                source_images[source].append(str(image_path))

        return [
            {
                "source": source,
                "splits": sorted(splits),
                "images": source_images[source]
            }
            for source, splits in source_splits.items()
            if len(splits) > 1
        ]

    # =========================================================
    # SOURCE
    # =========================================================

    def _infer_source(
        self,
        image_path: Path
    ) -> str:

        try:

            parts = image_path.relative_to(self.path).parts

            if len(parts) > 1:

                # Ignore standard split directory
                if parts[0] in {
                    "train", "validation", "val", "test", "prepared"
                }:

                    if len(parts) > 2:
                        return parts[1]

                return parts[0]

        except ValueError:
            pass

        return "default_source"

    # =========================================================
    # REPORT
    # =========================================================

    def report(
        self,
        run_redundancy: bool = True,
        run_leakage: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete dataset analysis report.
        """

        profile = self.profile()
        classes = self.analyze_classes()
        quality = self.analyze_quality()
        coverage = self.analyze_coverage()

        redundancy = self.detect_redundancy() if run_redundancy else None
        leakage = self.detect_leakage() if run_leakage else None

        self.report_result = {
            "dataset": str(self.path),
            "profile": profile,
            "classes": classes,
            "quality": quality,
            "coverage": coverage,
            "redundancy": redundancy,
            "leakage": leakage
        }

        return self.report_result