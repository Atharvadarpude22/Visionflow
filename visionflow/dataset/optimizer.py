from pathlib import Path
from typing import Optional, Any, Dict, List
from collections import Counter, defaultdict
import hashlib
import math
import random
import shutil

try:
    from PIL import Image
except ImportError:
    Image = None


class DatasetOptimizer:
    """
    Optimize a computer-vision dataset without blindly deleting
    potentially valuable information.

    Goals: remove exact duplicates, reduce redundancy, filter clearly
    irrelevant samples, select representative samples, preserve rare
    cases / class distribution / visual coverage / source diversity,
    reduce temporal redundancy, and build a smaller useful subset.

    Prefers KEEP / REVIEW / QUARANTINE over silently deleting
    uncertain samples.
    """

    SUPPORTED_IMAGE_FORMATS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
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

        # Dataset samples
        self.samples: List[Dict[str, Any]] = []

        # Intermediate results
        self.duplicate_result = None
        self.redundancy_result = None
        self.relevance_result = None
        self.representative_result = None
        self.rare_case_result = None
        self.coverage_result = None
        self.distribution_result = None
        self.source_result = None
        self.temporal_result = None
        self.value_result = None
        self.validation_result = None

        # Final result
        self.optimization_result = None

        # Internal cache: image size per path (reset by load())
        self._size_cache = {}

    # =========================================================
    # LOAD
    # =========================================================

    def load(self) -> List[Dict[str, Any]]:
        """
        Discover dataset images and create internal sample
        representations.
        """

        formats = self.SUPPORTED_IMAGE_FORMATS

        images = sorted(
            file
            for file in self.path.rglob("*")
            if file.suffix.lower() in formats
            and file.is_file()
            and "_quarantine" not in file.parts
            and "optimized" not in file.parts
        )

        self._size_cache = {}

        self.samples = [
            {
                "path": image_path,
                "status": "KEEP",
                "score": 0.0,
                "issues": [],
                "source": self._infer_source(image_path),
                "classes": self._extract_sample_classes(image_path),
            }
            for image_path in images
        ]

        return self.samples

    # =========================================================
    # DUPLICATES
    # =========================================================

    def remove_duplicates(
        self,
        quarantine: bool = False
    ) -> Dict[str, Any]:
        """
        Detect exact duplicate files.

        Duplicates are marked rather than physically deleted. Only
        files that share a size with another file are hashed, since
        files of different sizes cannot be identical.
        """

        if not self.samples:
            self.load()

        samples = self.samples

        sizes = []
        size_counts = Counter()

        for sample in samples:

            try:
                size = sample["path"].stat().st_size

            except OSError:

                sample["status"] = "REVIEW"
                sample["issues"].append("hash_failed")
                size = None

            sizes.append(size)

            if size is not None:
                size_counts[size] += 1

        hash_groups = defaultdict(list)

        for sample, size in zip(samples, sizes):

            if size is None or size_counts[size] < 2:
                continue

            try:
                hash_groups[self._file_hash(sample["path"])].append(sample)

            except OSError:

                sample["status"] = "REVIEW"
                sample["issues"].append("hash_failed")

        duplicate_groups = []
        removed = []

        for group in hash_groups.values():

            if len(group) <= 1:
                continue

            duplicate_groups.append([str(s["path"]) for s in group])

            # Keep the first copy.
            for duplicate in group[1:]:

                duplicate["status"] = "QUARANTINE"
                duplicate["issues"].append("exact_duplicate")

                removed.append(str(duplicate["path"]))

        self.duplicate_result = {
            "duplicate_groups": duplicate_groups,
            "duplicate_count": len(removed),
            "removed": removed,
            "quarantine": quarantine,
        }

        return self.duplicate_result

    # =========================================================
    # REDUNDANCY
    # =========================================================

    def remove_redundancy(
        self,
        similarity_threshold: float = 0.95,
        max_comparisons: int = 10000
    ) -> Dict[str, Any]:
        """
        Detect highly similar images.

        Keeps one representative from each group rather than
        deleting all similar samples.
        """

        if not self.samples:
            self.load()

        if Image is None:

            self.redundancy_result = {
                "status": "limited",
                "reason": "Pillow is not installed",
                "groups": [],
            }

            return self.redundancy_result

        # Hashes are stored as integers: similarity is one XOR plus
        # a bit count instead of a character-by-character compare.
        hashes = {}
        bits = 0

        for sample in self.samples:

            if sample["status"] != "KEEP":
                continue

            try:

                hash_text = self._average_hash(sample["path"])

                hashes[sample["path"]] = int(hash_text, 2)

                bits = len(hash_text)

            except Exception:

                sample["status"] = "REVIEW"

        by_path = {}

        for sample in self.samples:
            by_path.setdefault(sample["path"], sample)

        paths = list(hashes)
        count = len(paths)

        groups = []
        marked = set()

        comparisons = 0

        for i in range(count):

            if comparisons >= max_comparisons:
                break

            first = paths[i]

            if first in marked:
                continue

            first_hash = hashes[first]

            group = [first]

            for j in range(i + 1, count):

                if comparisons >= max_comparisons:
                    break

                comparisons += 1

                second = paths[j]

                similarity = (
                    bits - (first_hash ^ hashes[second]).bit_count()
                ) / bits

                if similarity >= similarity_threshold:

                    group.append(second)
                    marked.add(second)

            if len(group) > 1:

                groups.append(group)

                # Preserve first representative
                for redundant_path in group[1:]:

                    sample = by_path.get(redundant_path)

                    if sample is not None:

                        sample["status"] = "QUARANTINE"
                        sample["issues"].append("high_redundancy")

        self.redundancy_result = {
            "groups": [
                [str(path) for path in group]
                for group in groups
            ],
            "group_count": len(groups),
            "comparisons": comparisons,
            "threshold": similarity_threshold,
        }

        return self.redundancy_result

    # =========================================================
    # RELEVANCE
    # =========================================================

    def filter_irrelevant(
        self,
        relevance_function=None
    ) -> Dict[str, Any]:
        """
        Filter samples that are clearly irrelevant.

        relevance_function(sample) -> bool (True = relevant).
        Without one, no semantic assumptions are made.
        """

        if not self.samples:
            self.load()

        irrelevant = []

        if relevance_function is not None:

            if not callable(relevance_function):
                raise TypeError("relevance_function must be callable")

            for sample in self.samples:

                if sample["status"] != "KEEP":
                    continue

                try:

                    if not relevance_function(sample):

                        sample["status"] = "QUARANTINE"
                        sample["issues"].append("irrelevant")

                        irrelevant.append(str(sample["path"]))

                except Exception:

                    sample["status"] = "REVIEW"
                    sample["issues"].append("relevance_check_failed")

        self.relevance_result = {
            "irrelevant_count": len(irrelevant),
            "irrelevant": irrelevant,
            "custom_filter_used": relevance_function is not None,
        }

        return self.relevance_result

    # =========================================================
    # REPRESENTATIVE SELECTION
    # =========================================================

    def select_representative(
        self,
        groups: Optional[List[List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        Select representative samples from redundant groups using
        a score based on rarity, coverage, quality and source
        diversity.
        """

        if not self.samples:
            self.load()

        if groups is None:

            groups = []

            if self.redundancy_result:

                groups = [
                    [Path(path) for path in group]
                    for group in self.redundancy_result.get("groups", [])
                ]

        # One lookup table instead of a linear scan per path.
        by_path = {}

        for sample in self.samples:
            by_path.setdefault(sample["path"], sample)

        representatives = []

        for group in groups:

            best_score = None
            best_sample = None

            for path in group:

                sample = by_path.get(path)

                if sample is None:
                    continue

                score = self.estimate_sample_value(sample)

                sample["score"] = score

                # strict ">" keeps the first of equal scores,
                # same as a stable descending sort
                if best_score is None or score > best_score:
                    best_score = score
                    best_sample = sample

            if best_sample is None:
                continue

            representatives.append(str(best_sample["path"]))

        self.representative_result = {
            "representatives": representatives,
            "count": len(representatives),
        }

        return self.representative_result

    # =========================================================
    # RARE CASES
    # =========================================================

    def preserve_rare_cases(
        self,
        minimum_frequency: int = 2
    ) -> Dict[str, Any]:
        """
        Protect rare classes/cases. A rare class is never removed
        merely because it is uncommon.
        """

        if not self.samples:
            self.load()

        class_counts = Counter()

        for sample in self.samples:
            class_counts.update(sample.get("classes", []))

        rare_classes = {
            class_name
            for class_name, count in class_counts.items()
            if count <= minimum_frequency
        }

        protected = []

        for sample in self.samples:

            if any(
                class_name in rare_classes
                for class_name in sample.get("classes", [])
            ):

                sample["rare_case"] = True
                sample["status"] = "KEEP"
                sample["issues"].append("rare_case_protected")

                protected.append(str(sample["path"]))

            else:

                sample["rare_case"] = False

        self.rare_case_result = {
            "rare_classes": sorted(rare_classes),
            "protected_samples": protected,
            "protected_count": len(protected),
        }

        return self.rare_case_result

    # =========================================================
    # COVERAGE
    # =========================================================

    def preserve_coverage(self) -> Dict[str, Any]:
        """
        Preserve visual/source coverage using source, resolution
        and lighting.
        """

        if not self.samples:
            self.load()

        coverage = defaultdict(list)

        cache = self._size_cache

        for sample in self.samples:

            path = sample["path"]
            path_text = str(path)

            coverage[
                f"source:{sample.get('source', 'unknown')}"
            ].append(path_text)

            # Single image open for both lighting and resolution.
            lighting = "unknown"
            size = None

            if Image is not None:

                try:

                    with Image.open(path) as image:

                        size = image.size

                        try:

                            lighting = self._lighting_label(
                                image.convert("L").resize((1, 1)).tobytes()[0]
                            )

                        except Exception:
                            pass

                except Exception:
                    pass

            coverage[f"lighting:{lighting}"].append(path_text)

            if size is not None:

                cache[path] = size

                coverage[f"resolution:{size[0]}x{size[1]}"].append(path_text)

        self.coverage_result = {
            "groups": {
                key: {
                    "count": len(paths),
                    "samples": paths
                }
                for key, paths in coverage.items()
            }
        }

        return self.coverage_result

    # =========================================================
    # DISTRIBUTION
    # =========================================================

    def preserve_distribution(self) -> Dict[str, Any]:
        """
        Analyze class distribution and identify classes that should
        be preserved during subset creation.
        """

        if not self.samples:
            self.load()

        counts = Counter()

        for sample in self.samples:
            counts.update(sample.get("classes", []))

        total = sum(counts.values())

        distribution = {
            class_name: {
                "count": count,
                "ratio": count / total if total > 0 else 0.0
            }
            for class_name, count in counts.items()
        }

        self.distribution_result = {
            "counts": dict(counts),
            "distribution": distribution
        }

        return self.distribution_result

    # =========================================================
    # SOURCE OPTIMIZATION
    # =========================================================

    def optimize_sources(self) -> Dict[str, Any]:
        """
        Analyze source/camera distribution and protect source
        diversity.
        """

        if not self.samples:
            self.load()

        sources = defaultdict(list)

        for sample in self.samples:

            sources[sample.get("source", "unknown")].append(
                str(sample["path"])
            )

        self.source_result = {
            "sources": {
                source: {
                    "count": len(paths),
                    "samples": paths
                }
                for source, paths in sources.items()
            },
            "number_of_sources": len(sources)
        }

        return self.source_result

    # =========================================================
    # TEMPORAL OPTIMIZATION
    # =========================================================

    def optimize_temporal_redundancy(
        self,
        keep_every: int = 5
    ) -> Dict[str, Any]:
        """
        Reduce sequential frame redundancy (conservative; per source;
        rare cases stay protected).
        """

        if not self.samples:
            self.load()

        if keep_every < 1:
            raise ValueError("keep_every must be >= 1")

        source_groups = defaultdict(list)

        for sample in self.samples:

            source_groups[sample.get("source", "unknown")].append(sample)

        marked = []

        for samples in source_groups.values():

            samples.sort(key=lambda sample: str(sample["path"]))

            for index, sample in enumerate(samples):

                # Never remove protected rare cases
                if sample.get("rare_case", False):
                    continue

                if index % keep_every != 0:

                    if sample["status"] == "KEEP":

                        sample["status"] = "QUARANTINE"
                        sample["issues"].append("temporal_redundancy")

                        marked.append(str(sample["path"]))

        self.temporal_result = {
            "marked_count": len(marked),
            "marked": marked,
            "keep_every": keep_every,
        }

        return self.temporal_result

    # =========================================================
    # SAMPLE VALUE
    # =========================================================

    def estimate_sample_value(
        self,
        sample: Optional[Dict[str, Any]] = None
    ):
        """
        Estimate how valuable a sample is (rare cases, class
        diversity, source diversity, quality, useful resolution).
        """

        if sample is None:
            raise ValueError("sample cannot be None")

        score = 0.0

        # Rare case
        if sample.get("rare_case", False):
            score += 5.0

        # Class contribution
        score += min(len(sample.get("classes", [])) * 0.5, 2.0)

        # Source diversity
        if sample.get("source") not in (None, "unknown", "default_source"):
            score += 1.0

        # Quality
        if sample["status"] == "KEEP":
            score += 1.0

        # Resolution (image size is cached, header is read once)
        if Image is not None:

            path = sample["path"]
            cache = self._size_cache

            if path in cache:

                size = cache[path]

            else:

                try:

                    with Image.open(path) as image:
                        size = image.size

                except Exception:
                    size = None

                cache[path] = size

            if size is None:
                score -= 2.0

            elif size[0] * size[1] >= 640 * 480:
                score += 1.0

        return score

    # =========================================================
    # PRIORITIZE
    # =========================================================

    def prioritize_samples(self) -> List[Dict[str, Any]]:
        """Calculate sample value and rank samples."""

        if not self.samples:
            self.load()

        for sample in self.samples:
            sample["score"] = self.estimate_sample_value(sample)

        ranked = sorted(
            self.samples,
            key=lambda sample: sample["score"],
            reverse=True
        )

        self.value_result = {"ranked_samples": ranked}

        return ranked

    # =========================================================
    # BUILD SUBSET
    # =========================================================

    def build_subset(
        self,
        target_size: Optional[int] = None,
        target_ratio: Optional[float] = None,
        output_dir: Optional[str] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Build a smaller dataset while attempting to preserve rare
        cases, class distribution, coverage, source diversity and
        sample value.
        """

        if not self.samples:
            self.load()

        candidates = [
            sample for sample in self.samples if sample["status"] == "KEEP"
        ]

        if not candidates:
            raise ValueError(
                "No valid samples available for subset creation"
            )

        if target_size is None:

            if target_ratio is None:
                target_ratio = 1.0

            if not (0 < target_ratio <= 1):
                raise ValueError("target_ratio must be between 0 and 1")

            target_size = max(
                1,
                math.ceil(len(candidates) * target_ratio)
            )

        if target_size <= 0:
            raise ValueError("target_size must be greater than 0")

        target_size = min(target_size, len(candidates))

        # Rank candidates
        self.prioritize_samples()

        ranked = sorted(
            candidates,
            key=lambda sample: sample["score"],
            reverse=True
        )

        # Always preserve rare cases first
        selected = [
            sample for sample in ranked if sample.get("rare_case", False)
        ][:target_size]

        selected_paths = {sample["path"] for sample in selected}

        # Preserve source diversity
        sources = defaultdict(list)

        for sample in ranked:

            if sample["path"] in selected_paths:
                continue

            sources[sample.get("source", "unknown")].append(sample)

        for source_samples in sources.values():

            if len(selected) >= target_size:
                break

            best = max(source_samples, key=lambda sample: sample["score"])

            selected.append(best)
            selected_paths.add(best["path"])

        # Fill remaining slots
        remaining = [
            sample for sample in ranked
            if sample["path"] not in selected_paths
        ]

        rng = random.Random(seed)

        # Small controlled shuffle among equally useful candidates.
        rng.shuffle(remaining)

        remaining.sort(key=lambda sample: sample["score"], reverse=True)

        for sample in remaining:

            if len(selected) >= target_size:
                break

            selected.append(sample)

        # Create output directory
        output_path = (
            self.path / "optimized"
            if output_dir is None
            else Path(output_dir)
        )

        output_path.mkdir(parents=True, exist_ok=True)

        copied = []

        for sample in selected:

            source = sample["path"]

            destination = self._unique_destination(
                output_path / source.name
            )

            try:

                shutil.copy2(source, destination)

                copied.append(str(destination))

            except OSError:

                continue

        return {
            "target_size": target_size,
            "selected_count": len(selected),
            "selected": [str(sample["path"]) for sample in selected],
            "output_dir": str(output_path),
            "copied_count": len(copied),
            "copied": copied
        }

    # =========================================================
    # VALIDATE SUBSET
    # =========================================================

    def validate_subset(
        self,
        selected: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate the resulting subset: checks whether important
        information was retained.
        """

        if not selected:
            return {
                "valid": False,
                "reason": "Subset is empty"
            }

        class_counts = Counter()
        sources = Counter()

        rare_count = 0

        for sample in selected:

            class_counts.update(sample.get("classes", []))

            sources[sample.get("source", "unknown")] += 1

            if sample.get("rare_case", False):
                rare_count += 1

        result = {
            "valid": True,
            "sample_count": len(selected),
            "classes": dict(class_counts),
            "sources": dict(sources),
            "rare_cases": rare_count,
        }

        self.validation_result = result

        return result

    # =========================================================
    # SIMULATE SUBSET
    # =========================================================

    def simulate_subset(
        self,
        target_size: int
    ) -> Dict[str, Any]:
        """Simulate subset construction without copying files."""

        if not self.samples:
            self.load()

        if target_size <= 0:
            raise ValueError("target_size must be positive")

        self.prioritize_samples()

        ranked = sorted(
            (s for s in self.samples if s["status"] == "KEEP"),
            key=lambda sample: sample["score"],
            reverse=True
        )

        selected = ranked[:min(target_size, len(ranked))]

        validation = self.validate_subset(selected)

        return {
            "selected_count": len(selected),
            "validation": validation,
            "samples": [str(sample["path"]) for sample in selected]
        }

    # =========================================================
    # COMPARE SUBSETS
    # =========================================================

    def compare_subsets(
        self,
        first: List[Dict[str, Any]],
        second: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare two candidate subsets."""

        first_result = self.validate_subset(first)
        second_result = self.validate_subset(second)

        return {
            "first": first_result,
            "second": second_result,
            "size_difference": len(first) - len(second),
            "rare_case_difference": (
                first_result.get("rare_cases", 0)
                - second_result.get("rare_cases", 0)
            )
        }

    # =========================================================
    # OPTIMIZE
    # =========================================================

    def optimize(
        self,
        target_size: Optional[int] = None,
        target_ratio: Optional[float] = None,
        output_dir: Optional[str] = None,
        remove_duplicates: bool = True,
        remove_redundancy: bool = True,
        preserve_rare: bool = True,
        preserve_coverage: bool = True,
        preserve_distribution: bool = True,
        optimize_sources: bool = True,
        optimize_temporal: bool = False,
        temporal_keep_every: int = 5,
        relevance_function=None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run the complete optimization workflow:
        load -> duplicates -> redundancy -> relevance -> rare cases ->
        coverage -> distribution -> sources -> temporal -> scoring ->
        build subset -> validate.
        """

        self.load()

        original_count = len(self.samples)

        duplicate_result = (
            self.remove_duplicates() if remove_duplicates else None
        )

        redundancy_result = (
            self.remove_redundancy() if remove_redundancy else None
        )

        relevance_result = self.filter_irrelevant(relevance_function)

        rare_result = self.preserve_rare_cases() if preserve_rare else None

        coverage_result = (
            self.preserve_coverage() if preserve_coverage else None
        )

        distribution_result = (
            self.preserve_distribution() if preserve_distribution else None
        )

        source_result = self.optimize_sources() if optimize_sources else None

        temporal_result = (
            self.optimize_temporal_redundancy(keep_every=temporal_keep_every)
            if optimize_temporal
            else None
        )

        ranked = self.prioritize_samples()

        available_count = sum(
            1 for sample in self.samples if sample["status"] == "KEEP"
        )

        if target_size is None:

            if target_ratio is None:
                target_ratio = 1.0

            target_size = max(
                1,
                math.ceil(available_count * target_ratio)
            )

        subset_result = self.build_subset(
            target_size=target_size,
            output_dir=output_dir,
            seed=seed
        )

        selected_paths = {Path(path) for path in subset_result["selected"]}

        selected_samples = [
            sample for sample in self.samples
            if sample["path"] in selected_paths
        ]

        validation = self.validate_subset(selected_samples)

        selected_count = subset_result["selected_count"]

        self.optimization_result = {
            "status": "optimized",
            "original_count": original_count,
            "available_after_filtering": available_count,
            "target_size": target_size,
            "selected_count": selected_count,
            "reduction": original_count - selected_count,
            "reduction_ratio": (
                (original_count - selected_count) / original_count
                if original_count > 0
                else 0.0
            ),
            "duplicates": duplicate_result,
            "redundancy": redundancy_result,
            "relevance": relevance_result,
            "rare_cases": rare_result,
            "coverage": coverage_result,
            "distribution": distribution_result,
            "sources": source_result,
            "temporal": temporal_result,
            "sample_ranking": ranked,
            "subset": subset_result,
            "validation": validation
        }

        return self.optimization_result

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _find_sample(
        self,
        path: Path
    ) -> Optional[Dict[str, Any]]:

        for sample in self.samples:

            if sample["path"] == path:
                return sample

        return None

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

    def _average_hash(
        self,
        path: Path,
        size: int = 8
    ) -> str:

        if Image is None:
            raise RuntimeError("Pillow is required for image similarity")

        with Image.open(path) as image:

            pixels = image.convert("L").resize((size, size)).tobytes()

        average = sum(pixels) / len(pixels)

        return "".join(
            "1" if pixel >= average else "0"
            for pixel in pixels
        )

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

    def _extract_sample_classes(
        self,
        image_path: Path
    ) -> List[str]:
        """
        Extract YOLO-style classes from a matching .txt label,
        e.g. "0 0.5 0.5 0.2 0.3" -> ["0"].
        """

        classes = []

        try:

            # Opening directly avoids a separate exists() call;
            # a missing file raises OSError and returns [].
            with open(
                image_path.with_suffix(".txt"),
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    parts = line.split()

                    if parts:
                        classes.append(parts[0])

        except (OSError, UnicodeDecodeError):

            return []

        return classes

    def _infer_source(
        self,
        image_path: Path
    ) -> str:

        try:

            parts = image_path.relative_to(self.path).parts

            if len(parts) > 1:

                if parts[0] in {
                    "train", "validation", "val", "test",
                    "prepared", "optimized"
                }:

                    if len(parts) > 2:
                        return parts[1]

                return parts[0]

        except ValueError:
            pass

        return "default_source"

    def _lighting_label(
        self,
        brightness: float
    ) -> str:

        if brightness < 40:
            return "dark"

        if brightness < 100:
            return "low_light"

        if brightness < 180:
            return "normal"

        return "bright"

    def _estimate_lighting(
        self,
        image_path: Path
    ) -> str:

        if Image is None:
            return "unknown"

        try:

            with Image.open(image_path) as image:

                return self._lighting_label(
                    image.convert("L").resize((1, 1)).tobytes()[0]
                )

        except Exception:

            return "unknown"

    def _unique_destination(
        self,
        destination: Path
    ) -> Path:

        if not destination.exists():
            return destination

        counter = 1

        while True:

            candidate = (
                destination.parent
                / f"{destination.stem}_{counter}{destination.suffix}"
            )

            if not candidate.exists():
                return candidate

            counter += 1