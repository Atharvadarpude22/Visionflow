import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import visionflow as vf


# =========================================================
# CONFIGURATION
# =========================================================

# PUT YOUR ORIGINAL DATASET HERE
DATASET_PATH = Path(
    r"D:\New folder (2)\archive (22)\merged_dataset\test"
)

# Prepared dataset will be created alongside original dataset
#
# Example:
#
# merged_dataset/
# ├── test/
# └── test_prepared/
#
OUTPUT_PATH = DATASET_PATH.parent / (
    DATASET_PATH.name + "_prepared"
)

# Final report
REPORT_PATH = OUTPUT_PATH / "dataset_report.txt"


# =========================================================
# SAFETY
# =========================================================

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset does not exist:\n{DATASET_PATH}"
    )


if OUTPUT_PATH.exists():

    print(
        f"\nPrepared dataset already exists:"
        f"\n{OUTPUT_PATH}"
    )

    answer = input(
        "\nDelete existing prepared dataset? (y/n): "
    ).strip().lower()

    if answer == "y":

        shutil.rmtree(
            OUTPUT_PATH
        )

    else:

        print(
            "\nOperation cancelled."
        )

        raise SystemExit


OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CREATE DATASET
# =========================================================

dataset = vf.Dataset(
    str(DATASET_PATH)
)


# =========================================================
# RESULT STORAGE
# =========================================================

results = {}


# =========================================================
# RUN STAGE
# =========================================================

def run_stage(name, function):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    try:

        raw_result = function()

        results[name] = {
            "status": "SUCCESS",
            "result": raw_result
        }

        print("SUCCESS")

        return raw_result

    except Exception as e:

        results[name] = {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}"
        }

        print(
            f"FAILED: {type(e).__name__}: {e}"
        )

        return None


# =========================================================
# GENERIC HELPERS
# =========================================================

def is_path_like(value):
    """
    Detect filesystem paths.
    """

    if isinstance(value, Path):

        return True

    if not isinstance(value, str):

        return False

    text = value.lower()

    path_patterns = [
        ":\\",
        ":/",
        "\\",
        "/",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".txt",
        ".xml",
        ".json",
        ".yaml",
        ".yml",
    ]

    return any(
        pattern in text
        for pattern in path_patterns
    )


def is_individual_record(item):
    """
    Detect whether a dictionary represents an
    individual image/sample.

    These records must NEVER appear in the report.
    """

    if not isinstance(item, dict):

        return False

    sample_keys = {
        "image",
        "image_path",
        "image_name",
        "filename",
        "file",
        "annotation",
        "annotation_path",
        "metadata",
        "status",
        "issues",
        "warnings",
        "perceptual_hash",
        "hash",
    }

    return bool(
        sample_keys.intersection(
            set(item.keys())
        )
    )


# =========================================================
# AGGREGATE PER-SAMPLE RECORDS
# =========================================================

def aggregate_records(records):
    """
    Convert individual image/sample records into
    dataset-level statistics.

    IMPORTANT:

    Individual image records are NOT returned.

    No:
        image paths
        filenames
        annotation paths
        hashes
        individual samples

    are included in the final report.
    """

    total = 0

    status_counter = Counter()
    format_counter = Counter()
    resolution_counter = Counter()
    source_counter = Counter()

    issue_counter = Counter()
    warning_counter = Counter()

    class_counter = Counter()

    total_size = 0

    images_with_annotations = 0
    images_without_annotations = 0

    images_with_issues = 0
    images_with_warnings = 0

    video_samples = 0

    # Quality-specific counters
    blur_count = 0
    dark_count = 0
    night_count = 0
    corrupted_count = 0

    low_resolution_count = 0
    extreme_resolution_count = 0

    for record in records:

        if not isinstance(record, dict):

            continue

        total += 1

        # =================================================
        # STATUS
        # =================================================

        status = record.get(
            "status"
        )

        if status:

            status_counter[
                str(status).upper()
            ] += 1

        # =================================================
        # ANNOTATION
        # =================================================

        annotation = record.get(
            "annotation"
        )

        if annotation:

            images_with_annotations += 1

        else:

            images_without_annotations += 1

        # =================================================
        # ISSUES
        # =================================================

        issues = record.get(
            "issues",
            []
        )

        if isinstance(issues, list):

            if issues:

                images_with_issues += 1

            for issue in issues:

                issue_text = str(
                    issue
                ).lower()

                issue_counter[
                    str(issue)
                ] += 1

                # -----------------------------
                # Blur detection
                # -----------------------------

                if (
                    "blur" in issue_text
                    or "blurry" in issue_text
                ):

                    blur_count += 1

                # -----------------------------
                # Dark/night detection
                # -----------------------------

                if (
                    "dark" in issue_text
                    or "night" in issue_text
                    or "low_light" in issue_text
                    or "low-light" in issue_text
                    or "low light" in issue_text
                ):

                    dark_count += 1

                # -----------------------------
                # Corruption
                # -----------------------------

                if (
                    "corrupt" in issue_text
                    or "damaged" in issue_text
                ):

                    corrupted_count += 1

                # -----------------------------
                # Resolution
                # -----------------------------

                if (
                    "low resolution"
                    in issue_text
                    or "low_resolution"
                    in issue_text
                    or "small resolution"
                    in issue_text
                ):

                    low_resolution_count += 1

                if (
                    "extreme resolution"
                    in issue_text
                    or "extreme_resolution"
                    in issue_text
                ):

                    extreme_resolution_count += 1

        # =================================================
        # WARNINGS
        # =================================================

        warnings = record.get(
            "warnings",
            []
        )

        if isinstance(warnings, list):

            if warnings:

                images_with_warnings += 1

            for warning in warnings:

                warning_counter[
                    str(warning)
                ] += 1

        # =================================================
        # CLASSES
        # =================================================

        classes = record.get(
            "classes",
            []
        )

        if isinstance(classes, list):

            for class_name in classes:

                class_counter[
                    str(class_name)
                ] += 1

        # =================================================
        # SOURCE
        # =================================================

        source = record.get(
            "source"
        )

        if source:

            source_counter[
                str(source)
            ] += 1

        # =================================================
        # VIDEO
        # =================================================

        video = record.get(
            "video"
        )

        if video:

            video_samples += 1

        # =================================================
        # METADATA
        # =================================================

        metadata = record.get(
            "metadata",
            {}
        )

        if not isinstance(
            metadata,
            dict
        ):

            continue

        # -------------------------------------------------
        # File size
        # -------------------------------------------------

        file_size = metadata.get(
            "file_size",
            0
        )

        if isinstance(
            file_size,
            (int, float)
        ):

            total_size += file_size

        # -------------------------------------------------
        # Format
        # -------------------------------------------------

        image_format = metadata.get(
            "format"
        )

        if image_format:

            format_counter[
                str(image_format).lower()
            ] += 1

        # -------------------------------------------------
        # Resolution
        # -------------------------------------------------

        width = metadata.get(
            "width"
        )

        height = metadata.get(
            "height"
        )

        if width and height:

            resolution_counter[
                f"{width}x{height}"
            ] += 1

    # =====================================================
    # RETURN DATASET-LEVEL STATISTICS
    # =====================================================

    return {

        "total_samples": total,

        "annotations": {

            "with_annotations":
                images_with_annotations,

            "without_annotations":
                images_without_annotations
        },

        "status": dict(
            status_counter
        ),

        "formats": dict(
            format_counter
        ),

        "resolutions": dict(
            resolution_counter
        ),

        "classes": dict(
            class_counter
        ),

        "sources": dict(
            source_counter
        ),

        "video": {

            "video_samples":
                video_samples
        },

        "quality": {

            "blurry_images":
                blur_count,

            "dark_or_night_images":
                dark_count,

            "corrupted_images":
                corrupted_count,

            "low_resolution_images":
                low_resolution_count,

            "extreme_resolution_images":
                extreme_resolution_count,

            "images_with_any_issue":
                images_with_issues,

            "images_with_warnings":
                images_with_warnings
        },

        "issues": dict(
            issue_counter
        ),

        "warnings": dict(
            warning_counter
        ),

        "dataset_size": {

            "bytes":
                total_size,

            "megabytes":
                round(
                    total_size / (1024 ** 2),
                    2
                ),

            "gigabytes":
                round(
                    total_size / (1024 ** 3),
                    2
                )
        }
    }


# =========================================================
# AGGREGATE DICTIONARIES
# =========================================================

def aggregate_dictionary(
    data,
    parent_key=""
):
    """
    Recursively process analyzer output.

    Individual samples are removed.

    Dataset-level statistics are preserved.
    """

    if not isinstance(
        data,
        dict
    ):

        return data

    output = {}

    forbidden_keys = {

        "image",
        "image_path",
        "image_paths",
        "image_name",
        "image_names",

        "annotation",
        "annotation_path",
        "annotation_paths",

        "filename",
        "filenames",

        "file",
        "files",

        "filepath",
        "filepaths",

        "path",
        "paths",

        "source_path",
        "source_paths",

        "hash",
        "hashes",

        "perceptual_hash",
        "perceptual_hashes",
    }

    for key, value in data.items():

        key_string = str(
            key
        )

        key_lower = key_string.lower()

        # =================================================
        # NEVER REPORT FILE INFORMATION
        # =================================================

        if key_lower in forbidden_keys:

            if isinstance(
                value,
                list
            ):

                output[
                    f"{key_string}_count"
                ] = len(value)

            continue

        # =================================================
        # LIST
        # =================================================

        if isinstance(
            value,
            list
        ):

            # ---------------------------------------------
            # Individual records
            # ---------------------------------------------

            if value and all(
                isinstance(
                    x,
                    dict
                )
                for x in value
            ):

                if all(
                    is_individual_record(x)
                    for x in value
                ):

                    output[
                        key_string
                    ] = aggregate_records(
                        value
                    )

                else:

                    output[
                        key_string
                    ] = {
                        "count":
                            len(value)
                    }

            else:

                # Do not dump lists of paths
                if all(
                    is_path_like(x)
                    for x in value
                    if isinstance(x, str)
                ):

                    output[
                        f"{key_string}_count"
                    ] = len(value)

                else:

                    # Keep small categorical lists
                    if len(value) <= 20:

                        output[
                            key_string
                        ] = value

                    else:

                        output[
                            f"{key_string}_count"
                        ] = len(value)

            continue

        # =================================================
        # NESTED DICTIONARY
        # =================================================

        if isinstance(
            value,
            dict
        ):

            output[
                key_string
            ] = aggregate_dictionary(
                value,
                parent_key=key_string
            )

            continue

        # =================================================
        # PATH OBJECT
        # =================================================

        if isinstance(
            value,
            Path
        ):

            continue

        # =================================================
        # PATH-LIKE STRING
        # =================================================

        if isinstance(
            value,
            str
        ):

            if is_path_like(value):

                continue

        # =================================================
        # NORMAL VALUE
        # =================================================

        output[
            key_string
        ] = value

    return output


# =========================================================
# PREPARE REPORT RESULT
# =========================================================

def prepare_report_result(
    stage_name,
    result
):

    if result is None:

        return None

    # =====================================================
    # LIST OF INDIVIDUAL RECORDS
    # =====================================================

    if isinstance(
        result,
        list
    ):

        if result and all(
            isinstance(
                x,
                dict
            )
            for x in result
        ):

            if all(
                is_individual_record(x)
                for x in result
            ):

                return aggregate_records(
                    result
                )

        # General list
        return {
            "count":
                len(result)
        }

    # =====================================================
    # DICTIONARY
    # =====================================================

    if isinstance(
        result,
        dict
    ):

        return aggregate_dictionary(
            result
        )

    # =====================================================
    # NORMAL VALUE
    # =====================================================

    return result


# =========================================================
# LOAD
# =========================================================

load_result = run_stage(
    "LOAD",
    dataset._preparer.load
)


# =========================================================
# VALIDATION
# =========================================================

validation_result = run_stage(
    "VALIDATION",
    dataset._preparer.validate
)


# =========================================================
# CLEANING
# =========================================================

clean_result = run_stage(
    "CLEANING",
    dataset._preparer.clean
)


# =========================================================
# SPLITTING
# =========================================================

split_result = run_stage(
    "SPLITTING",
    dataset._preparer.split
)


# =========================================================
# DATASET PROFILE
# =========================================================

profile_result = run_stage(
    "DATASET PROFILE",
    dataset._analyzer.profile
)


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

class_result = run_stage(
    "CLASS DISTRIBUTION",
    dataset._analyzer.analyze_classes
)


# =========================================================
# QUALITY ANALYSIS
# =========================================================

quality_result = run_stage(
    "QUALITY ANALYSIS",
    dataset._analyzer.analyze_quality
)


# =========================================================
# COVERAGE ANALYSIS
# =========================================================

coverage_result = run_stage(
    "COVERAGE ANALYSIS",
    dataset._analyzer.analyze_coverage
)


# =========================================================
# REDUNDANCY
# =========================================================

redundancy_result = run_stage(
    "REDUNDANCY ANALYSIS",
    dataset._analyzer.detect_redundancy
)


# =========================================================
# LEAKAGE
# =========================================================

leakage_result = run_stage(
    "LEAKAGE ANALYSIS",
    dataset._analyzer.detect_leakage
)


# =========================================================
# FINAL SUMMARY
# =========================================================

final_summary = run_stage(
    "FINAL DATASET SUMMARY",
    dataset.summary
)


# =========================================================
# AGGREGATE ALL RESULTS
# =========================================================

aggregated_results = {}

for stage_name, data in results.items():

    if data.get(
        "status"
    ) == "FAILED":

        aggregated_results[
            stage_name
        ] = data

        continue

    raw_result = data.get(
        "result"
    )

    aggregated_results[
        stage_name
    ] = {

        "status":
            "SUCCESS",

        "result":
            prepare_report_result(
                stage_name,
                raw_result
            )
    }


# =========================================================
# BUILD HUMAN-READABLE REPORT
# =========================================================

report = []

report.append(
    "VISIONFLOW DATASET PREPARATION REPORT"
)

report.append(
    "=" * 80
)

report.append("")

report.append(
    f"Generated: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

report.append("")

report.append(
    "REPORT SCOPE"
)

report.append(
    "-" * 80
)

report.append(
    "This report contains aggregated dataset findings."
)

report.append(
    "Individual image records are intentionally excluded."
)

report.append(
    "No image paths, filenames, annotation paths or hashes "
    "are included."
)

report.append("")

report.append(
    "=" * 80
)


# =========================================================
# DATASET INFORMATION
# =========================================================

report.append(
    "\nDATASET INFORMATION"
)

report.append(
    "-" * 80
)

report.append(
    f"Original dataset exists: "
    f"{DATASET_PATH.exists()}"
)

report.append(
    f"Prepared dataset name: "
    f"{OUTPUT_PATH.name}"
)

report.append(
    "Original dataset modified: NO"
)


# =========================================================
# STAGE REPORTS
# =========================================================

for stage_name, data in aggregated_results.items():

    report.append("\n")

    report.append(
        stage_name
    )

    report.append(
        "-" * 80
    )

    status = data.get(
        "status"
    )

    report.append(
        f"Status: {status}"
    )

    if status == "FAILED":

        report.append(
            f"Error: "
            f"{data.get('error')}"
        )

        continue

    result = data.get(
        "result"
    )

    if result is None:

        report.append(
            "No result returned."
        )

        continue

    try:

        formatted = json.dumps(
            result,
            indent=4,
            default=str
        )

        report.append(
            formatted
        )

    except Exception:

        report.append(
            str(result)
        )


# =========================================================
# EXPLICIT QUALITY SUMMARY
# =========================================================

quality_data = (
    aggregated_results
    .get(
        "QUALITY ANALYSIS",
        {}
    )
    .get(
        "result",
        {}
    )
)


if isinstance(
    quality_data,
    dict
):

    report.append("\n")

    report.append(
        "=" * 80
    )

    report.append(
        "QUALITY FINDINGS SUMMARY"
    )

    report.append(
        "=" * 80
    )

    quality_section = (
        quality_data.get(
            "quality"
        )
    )

    if isinstance(
        quality_section,
        dict
    ):

        report.append(
            f"\nBlurry images: "
            f"{quality_section.get('blurry_images', 0)}"
        )

        report.append(
            f"Dark/night images: "
            f"{quality_section.get('dark_or_night_images', 0)}"
        )

        report.append(
            f"Corrupted images: "
            f"{quality_section.get('corrupted_images', 0)}"
        )

        report.append(
            f"Low-resolution images: "
            f"{quality_section.get('low_resolution_images', 0)}"
        )

        report.append(
            f"Extreme-resolution images: "
            f"{quality_section.get('extreme_resolution_images', 0)}"
        )

        report.append(
            f"Images with any issue: "
            f"{quality_section.get('images_with_any_issue', 0)}"
        )

        report.append(
            f"Images with warnings: "
            f"{quality_section.get('images_with_warnings', 0)}"
        )


# =========================================================
# PREPARATION SUMMARY
# =========================================================

report.append("\n")

report.append(
    "=" * 80
)

report.append(
    "PREPARATION SUMMARY"
)

report.append(
    "=" * 80
)

report.append("")

report.append(
    "Original dataset:"
)

report.append(
    "The original dataset was not modified."
)

report.append("")

report.append(
    "Prepared dataset:"
)

report.append(
    "A separate prepared dataset was created."
)

report.append("")

report.append(
    "Report contents:"
)

report.append(
    "Dataset-level statistics and findings only."
)

report.append(
    "Individual image information is excluded."
)

report.append(
    "Image paths and filenames are excluded."
)


# =========================================================
# WRITE REPORT
# =========================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# =========================================================
# FINAL CONSOLE OUTPUT
# =========================================================

print("\n")

print(
    "#" * 70
)

print(
    "VISIONFLOW DATASET PREPARATION COMPLETE"
)

print(
    "#" * 70
)

print(
    "\nOriginal dataset:"
)

print(
    DATASET_PATH
)

print(
    "\nPrepared dataset:"
)

print(
    OUTPUT_PATH
)

print(
    "\nDetailed report:"
)

print(
    REPORT_PATH
)

print(
    "\nOriginal dataset was NOT modified."
)

print(
    "Report contains aggregated dataset findings only."
)

print(
    "No individual image paths or filenames are included."
)

print(
    "\nImportant quality findings such as:"
)

print(
    "  - blurry images"
)

print(
    "  - dark/night images"
)

print(
    "  - corrupted images"
)

print(
    "  - low-resolution images"
)

print(
    "  - duplicate/redundant samples"
)

print(
    "  - leakage"
)

print(
    "are reported as dataset-level findings."
)