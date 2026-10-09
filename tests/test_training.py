import json
import traceback
import platform
import time
from pathlib import Path

import visionflow as vf


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = Path(
    r"D:\New folder (2)\archive (22)\merged_dataset\test_prepared"
)

CONFIG_PATH = None

MODEL_PATH = Path(
    r"yolov8n.pt"
)

EPOCHS = 1
BATCH_SIZE = 4
IMAGE_SIZE = 416
DEVICE = "auto"

REPORT_PATH = DATASET_PATH / "visionflow_training_report.txt"


# =========================================================
# REPORT STORAGE
# =========================================================

REPORT = {
    "system": {},
    "dataset": {},
    "model": {},
    "training": {},
    "validation": {},
    "pipeline": {},
    "errors": [],
    "timestamps": {}
}


# =========================================================
# HELPERS
# =========================================================

def make_serializable(value):

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): make_serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_serializable(v)
            for v in value
        ]

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def run_test(title, function):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    start = time.perf_counter()

    try:

        result = function()

        elapsed = (
            time.perf_counter()
            - start
        )

        print("STATUS: SUCCESS")

        if result is not None:

            try:
                print(
                    json.dumps(
                        result,
                        indent=4,
                        default=str
                    )
                )
            except Exception:
                print(result)

        return result

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - start
        )

        print("STATUS: FAILED")

        print(
            f"{type(error).__name__}: {error}"
        )

        traceback.print_exc()

        REPORT["errors"].append({
            "stage": title,
            "type": type(error).__name__,
            "message": str(error),
            "duration_seconds": elapsed
        })

        return None


# =========================================================
# SYSTEM INFORMATION
# =========================================================

def collect_system_information():

    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "hostname": platform.node()
    }

    try:

        import torch

        info["pytorch_version"] = (
            torch.__version__
        )

        info["cuda_available"] = bool(
            torch.cuda.is_available()
        )

        if torch.cuda.is_available():

            info["cuda_version"] = (
                torch.version.cuda
            )

            info["gpu_count"] = (
                torch.cuda.device_count()
            )

            info["gpu_devices"] = []

            for index in range(
                torch.cuda.device_count()
            ):

                info["gpu_devices"].append(
                    torch.cuda.get_device_name(
                        index
                    )
                )

        else:

            info["cuda_version"] = None
            info["gpu_count"] = 0
            info["gpu_devices"] = []

    except Exception as error:

        info["pytorch_error"] = str(error)

    try:

        import torchvision

        info["torchvision_version"] = (
            torchvision.__version__
        )

    except Exception as error:

        info["torchvision_version"] = (
            f"ERROR: {error}"
        )

    try:

        import ultralytics

        info["ultralytics_version"] = (
            ultralytics.__version__
        )

    except Exception as error:

        info["ultralytics_version"] = (
            f"ERROR: {error}"
        )

    return info


# =========================================================
# DATASET INFORMATION
# =========================================================

def collect_dataset_information():

    result = {
        "path": str(DATASET_PATH),
        "exists": DATASET_PATH.exists(),
        "is_directory": DATASET_PATH.is_dir(),
        "files": {},
        "yaml": None
    }

    if not DATASET_PATH.exists():
        return result

    extensions = {
        "images": [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ],
        "labels": [
            ".txt"
        ],
        "yaml": [
            ".yaml",
            ".yml"
        ]
    }

    for category, suffixes in extensions.items():

        count = 0

        for suffix in suffixes:

            count += len(
                list(
                    DATASET_PATH.rglob(
                        f"*{suffix}"
                    )
                )
            )

        result["files"][
            category
        ] = count

    yaml_files = list(
        DATASET_PATH.rglob(
            "*.yaml"
        )
    )

    yaml_files += list(
        DATASET_PATH.rglob(
            "*.yml"
        )
    )

    if yaml_files:

        result["yaml"] = str(
            yaml_files[0]
        )

    return result


# =========================================================
# START
# =========================================================

REPORT["timestamps"]["started"] = (
    time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)

print("\n")
print("#" * 70)
print("VISIONFLOW TRAINING TEST")
print("#" * 70)


# =========================================================
# SYSTEM
# =========================================================

REPORT["system"] = (
    collect_system_information()
)


# =========================================================
# CHECK DATASET
# =========================================================

print("\nDataset:")
print(DATASET_PATH)

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset does not exist:\n"
        f"{DATASET_PATH}"
    )

if not DATASET_PATH.is_dir():

    raise NotADirectoryError(
        f"Dataset path is not a directory:\n"
        f"{DATASET_PATH}"
    )

print("\nDataset exists: YES")

REPORT["dataset"] = (
    collect_dataset_information()
)


# =========================================================
# CHECK YAML
# =========================================================

if CONFIG_PATH is not None:

    CONFIG_PATH = Path(
        CONFIG_PATH
    )

    print("\nConfiguration:")
    print(CONFIG_PATH)

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Configuration file does not exist:\n"
            f"{CONFIG_PATH}"
        )

    print(
        "Configuration exists: YES"
    )

else:

    print("\nConfiguration: NONE")
    print(
        "Using VisionFlow defaults."
    )


REPORT["dataset"][
    "configuration"
] = (
    str(CONFIG_PATH)
    if CONFIG_PATH is not None
    else "VisionFlow defaults"
)


# =========================================================
# CHECK MODEL
# =========================================================

print("\nModel:")
print(MODEL_PATH)

if MODEL_PATH is None:

    raise ValueError(
        "MODEL_PATH must point to "
        "a valid model."
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model does not exist:\n"
        f"{MODEL_PATH}"
    )

if MODEL_PATH.suffix.lower() != ".pt":

    raise ValueError(
        "MODEL_PATH must be a .pt model."
    )

print("Model exists: YES")

REPORT["model"] = {
    "path": str(MODEL_PATH),
    "exists": True,
    "format": MODEL_PATH.suffix,
    "name": MODEL_PATH.name
}


# =========================================================
# CREATE DATASET
# =========================================================

def create_dataset():

    if CONFIG_PATH is not None:

        return vf.Dataset(
            str(DATASET_PATH),
            config=str(CONFIG_PATH)
        )

    return vf.Dataset(
        str(DATASET_PATH)
    )


dataset = run_test(
    "1. CREATE DATASET",
    create_dataset
)

if dataset is None:

    raise SystemExit(
        "\nDataset creation failed."
    )


# =========================================================
# DATASET SUMMARY
# =========================================================

dataset_summary = run_test(
    "2. DATASET SUMMARY",
    dataset.summary
)

REPORT["dataset"][
    "visionflow_summary"
] = make_serializable(
    dataset_summary
)


# =========================================================
# CREATE MODEL
# =========================================================

def create_model():

    return vf.Model(
        str(MODEL_PATH)
    )


model = run_test(
    "3. CREATE MODEL",
    create_model
)

if model is None:

    raise SystemExit(
        "\nModel creation failed."
    )

print(
    f"\nModel object: "
    f"{type(model).__name__}"
)


# =========================================================
# CREATE PIPELINE
# =========================================================

pipeline = vf.Pipeline(
    name="VisionFlow Training Test"
)


# =========================================================
# REGISTER DATASET
# =========================================================

dataset_result = run_test(
    "4. ADD DATASET TO PIPELINE",
    lambda: pipeline.add_dataset(
        dataset
    )
)

if dataset_result is None:

    raise SystemExit(
        "\nCould not add dataset."
    )


# =========================================================
# REGISTER MODEL
# =========================================================

model_result = run_test(
    "5. ADD MODEL TO PIPELINE",
    lambda: pipeline.set_model(
        model
    )
)

if model_result is None:

    raise SystemExit(
        "\nCould not add model."
    )


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

print("\n")
print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(
    f"Epochs      : {EPOCHS}"
)

print(
    f"Batch size  : {BATCH_SIZE}"
)

print(
    f"Image size  : {IMAGE_SIZE}"
)

print(
    f"Device      : {DEVICE}"
)


REPORT["training"] = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "requested_device": DEVICE
}


# =========================================================
# TRAIN
# =========================================================

training_result = run_test(
    "6. MODEL TRAINING",
    lambda: pipeline.train(
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE
    )
)

REPORT["training"][
    "result"
] = make_serializable(
    training_result
)


# =========================================================
# VALIDATION
# =========================================================

if hasattr(model, "validate"):

    validation_result = run_test(
        "7. MODEL VALIDATION",
        lambda: model.validate(
            dataset=dataset
        )
    )

    REPORT["validation"] = (
        make_serializable(
            validation_result
        )
    )

else:

    print(
        "\nValidation method is not available."
    )

    REPORT["validation"] = {
        "status": "NOT_AVAILABLE"
    }


# =========================================================
# PIPELINE SUMMARY
# =========================================================

pipeline_summary = run_test(
    "8. PIPELINE SUMMARY",
    pipeline.summary
)

REPORT["pipeline"] = (
    make_serializable(
        pipeline_summary
    )
)


# =========================================================
# FINAL STATUS
# =========================================================

if training_result is not None:

    REPORT["training"][
        "status"
    ] = "SUCCESS"

else:

    REPORT["training"][
        "status"
    ] = "FAILED"


REPORT["timestamps"]["finished"] = (
    time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


# =========================================================
# GENERATE DETAILED TXT REPORT
# =========================================================

def write_section(
    file,
    title,
    data
):

    file.write("\n")
    file.write("=" * 80)
    file.write("\n")
    file.write(title)
    file.write("\n")
    file.write("=" * 80)
    file.write("\n")

    if isinstance(data, dict):

        for key, value in data.items():

            if isinstance(
                value,
                (dict, list)
            ):

                file.write(
                    f"\n{key}:\n"
                )

                file.write(
                    json.dumps(
                        value,
                        indent=4,
                        default=str
                    )
                )

                file.write("\n")

            else:

                file.write(
                    f"{key}: {value}\n"
                )

    elif data is None:

        file.write(
            "None\n"
        )

    else:

        file.write(
            str(data)
        )

        file.write("\n")


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        "VISIONFLOW TRAINING REPORT\n"
    )

    report_file.write(
        "=" * 80
    )

    report_file.write("\n")

    report_file.write(
        "Generated by VisionFlow\n"
    )

    report_file.write(
        f"Generated at: "
        f"{REPORT['timestamps']['finished']}\n"
    )

    write_section(
        report_file,
        "1. SYSTEM INFORMATION",
        REPORT["system"]
    )

    write_section(
        report_file,
        "2. DATASET INFORMATION",
        REPORT["dataset"]
    )

    write_section(
        report_file,
        "3. MODEL INFORMATION",
        REPORT["model"]
    )

    write_section(
        report_file,
        "4. TRAINING CONFIGURATION AND RESULT",
        REPORT["training"]
    )

    write_section(
        report_file,
        "5. VALIDATION RESULTS",
        REPORT["validation"]
    )

    write_section(
        report_file,
        "6. PIPELINE SUMMARY",
        REPORT["pipeline"]
    )

    write_section(
        report_file,
        "7. ERRORS",
        REPORT["errors"]
    )

    report_file.write("\n")
    report_file.write(
        "=" * 80
    )
    report_file.write("\n")
    report_file.write(
        "FINAL STATUS\n"
    )
    report_file.write(
        "=" * 80
    )
    report_file.write("\n")

    report_file.write(
        f"Training status: "
        f"{REPORT['training']['status']}\n"
    )

    report_file.write(
        f"Dataset: "
        f"{DATASET_PATH}\n"
    )

    report_file.write(
        f"Model: "
        f"{MODEL_PATH}\n"
    )

    report_file.write(
        f"Epochs: "
        f"{EPOCHS}\n"
    )

    report_file.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )

    report_file.write(
        f"Image size: "
        f"{IMAGE_SIZE}\n"
    )

    report_file.write(
        f"Requested device: "
        f"{DEVICE}\n"
    )

    report_file.write(
        f"Report path: "
        f"{REPORT_PATH}\n"
    )


# =========================================================
# COMPLETE
# =========================================================

print("\n")
print("#" * 70)
print("VISIONFLOW TRAINING TEST COMPLETE")
print("#" * 70)

print(
    f"\nDetailed report created at:\n"
    f"{REPORT_PATH}"
)

print(
    "\nTraining status: "
    f"{REPORT['training']['status']}"
)