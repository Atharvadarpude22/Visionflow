from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime
import hashlib
import json
import copy


class DatasetVersioner:
    """
    Dataset version management for VisionFlow.

    - Create dataset versions
    - Store version metadata
    - Compare versions
    - Restore previous versions
    - Track dataset lineage
    - Track model association
    """

    def __init__(
        self,
        path=None,
        dataset=None,
        config: Optional[dict] = None
    ):
        self.path = Path(path) if path is not None else None

        self.dataset = dataset
        self.config = config or {}

        # Version storage
        self.version_directory = (
            self.path / ".visionflow_versions"
            if self.path is not None
            else None
        )

        if self.version_directory is not None:

            self.version_directory.mkdir(parents=True, exist_ok=True)

        # Internal state
        self.versions = []

        # (mtime, size) of versions.json at the last load/save, so the
        # file is only re-parsed when it actually changed.
        self._version_signature = None

        self._load_versions()

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _version_file(self):
        """Location of the version metadata file."""

        if self.version_directory is None:
            return None

        return self.version_directory / "versions.json"

    def _load_versions(self):
        """Load previously stored version metadata (only if changed)."""

        version_file = self._version_file()

        if version_file is None:
            return

        try:

            stat = version_file.stat()

        except OSError:

            self.versions = []
            self._version_signature = None

            return

        signature = (stat.st_mtime_ns, stat.st_size)

        if signature == self._version_signature:
            return

        try:

            with open(version_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.versions = data if isinstance(data, list) else []

        except Exception:

            self.versions = []

        self._version_signature = signature

    def _save_versions(self):
        """Save version metadata."""

        version_file = self._version_file()

        if version_file is None:
            return

        version_file.parent.mkdir(parents=True, exist_ok=True)

        with open(version_file, "w", encoding="utf-8") as file:

            json.dump(self.versions, file, indent=4, default=str)

        try:

            stat = version_file.stat()

            self._version_signature = (stat.st_mtime_ns, stat.st_size)

        except OSError:

            self._version_signature = None

    def _find_version(
        self,
        version
    ):
        """First stored version whose name or id matches."""

        for item in self.versions:

            if (
                item.get("version") == version
                or item.get("version_id") == version
            ):
                return item

        return None

    def _dataset_statistics(self) -> Dict[str, Any]:
        """
        Generate lightweight dataset statistics.

        No image contents are copied here.
        """

        if self.path is None or not self.path.exists():

            return {"exists": False}

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        label_extensions = {".txt", ".xml", ".json"}

        image_count = 0
        label_count = 0

        try:

            for file in self.path.rglob("*"):

                suffix = file.suffix.lower()

                # Extension first: avoids a disk call for other files.
                if suffix in image_extensions:

                    if file.is_file():
                        image_count += 1

                elif suffix in label_extensions:

                    if file.is_file():
                        label_count += 1

        except Exception:

            pass

        return {
            "exists": True,
            "image_count": image_count,
            "label_count": label_count
        }

    def _generate_id(
        self,
        metadata
    ) -> str:
        """Generate a deterministic version identifier."""

        raw = json.dumps(metadata, sort_keys=True, default=str)

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    # =========================================================
    # CREATE
    # =========================================================

    def create(
        self,
        description: str = "",
        changes=None,
        model=None,
        metadata=None
    ) -> Dict[str, Any]:
        """
        Create a new dataset version.

        description: human-readable description.
        changes:     list describing changes.
        model:       model associated with this dataset version.
        metadata:    additional metadata.
        """

        if changes is None:
            changes = []

        if not isinstance(changes, list):
            changes = [changes]

        model_name = None

        if model is not None:

            model_name = getattr(model, "name", None)

            if model_name is None:
                model_name = getattr(model, "path", None)

            if model_name is not None:
                model_name = str(model_name)

        version_metadata = {
            "version": f"V{len(self.versions) + 1}",
            "created_at": datetime.now().isoformat(),
            "description": description,
            "changes": changes,
            "dataset_statistics": self._dataset_statistics(),
            "model": model_name,
            "metadata": metadata or {}
        }

        version_metadata["version_id"] = self._generate_id(version_metadata)

        self.versions.append(version_metadata)

        self._save_versions()

        return version_metadata

    # =========================================================
    # LIST VERSIONS
    # =========================================================

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all dataset versions."""

        self._load_versions()

        return copy.deepcopy(self.versions)

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> Dict[str, Any]:
        """Return versioning summary."""

        self._load_versions()

        return {
            "dataset_exists": (
                self.path.exists() if self.path is not None else False
            ),
            "version_count": len(self.versions),
            "latest_version": (
                self.versions[-1]["version"] if self.versions else None
            ),
            "version_directory": (
                str(self.version_directory)
                if self.version_directory
                else None
            )
        }

    # =========================================================
    # COMPARE
    # =========================================================

    def compare(
        self,
        version_a,
        version_b
    ) -> Dict[str, Any]:
        """
        Compare two dataset versions.

        version_a/version_b can be "V1", "V2", a version id,
        or version dictionaries.
        """

        first = (
            version_a
            if isinstance(version_a, dict)
            else self._find_version(version_a)
        )

        second = (
            version_b
            if isinstance(version_b, dict)
            else self._find_version(version_b)
        )

        if first is None:
            raise ValueError(f"Version not found: {version_a}")

        if second is None:
            raise ValueError(f"Version not found: {version_b}")

        first_stats = first.get("dataset_statistics", {})
        second_stats = second.get("dataset_statistics", {})

        differences = {}

        for key in set(first_stats).union(second_stats):

            old_value = first_stats.get(key)
            new_value = second_stats.get(key)

            if old_value != new_value:

                differences[key] = {
                    "previous": old_value,
                    "current": new_value
                }

        return {
            "version_a": first.get("version"),
            "version_b": second.get("version"),
            "changes_a": first.get("changes", []),
            "changes_b": second.get("changes", []),
            "dataset_differences": differences
        }

    # =========================================================
    # RESTORE
    # =========================================================

    def restore(
        self,
        version
    ) -> Dict[str, Any]:
        """
        Restore metadata/state information from a previous version.

        This method intentionally does not silently delete or
        overwrite the user's actual dataset.
        """

        target = self._find_version(version)

        if target is None:
            raise ValueError(f"Version not found: {version}")

        return {
            "status": "RESTORE_REFERENCE",
            "version": target.get("version"),
            "version_id": target.get("version_id"),
            "message": (
                "Version metadata located. "
                "Original dataset was not modified."
            ),
            "dataset_statistics": target.get("dataset_statistics", {})
        }

    # =========================================================
    # LINEAGE
    # =========================================================

    def lineage(self) -> Dict[str, Any]:
        """Show dataset lineage: V1 -> V2 -> V3."""

        self._load_versions()

        lineage = []

        parent = None

        for version in self.versions:

            name = version.get("version")

            lineage.append({
                "version": name,
                "created_at": version.get("created_at"),
                "description": version.get("description"),
                "changes": version.get("changes", []),
                "model": version.get("model"),
                "version_id": version.get("version_id"),
                "parent_version": parent
            })

            parent = name

        return {
            "dataset": str(self.path) if self.path is not None else None,
            "versions": lineage
        }

    # =========================================================
    # LATEST VERSION
    # =========================================================

    def latest(self):
        """Return the latest version."""

        if not self.versions:
            return None

        return copy.deepcopy(self.versions[-1])

    # =========================================================
    # GET VERSION
    # =========================================================

    def get(
        self,
        version
    ):
        """Get a specific version."""

        item = self._find_version(version)

        return copy.deepcopy(item) if item is not None else None

    # =========================================================
    # CLEAR METADATA
    # =========================================================

    def clear(self):
        """
        Clear VisionFlow version metadata.

        This does NOT delete the actual dataset.
        """

        self.versions = []

        self._save_versions()

        return {
            "status": "CLEARED",
            "message": "Version metadata cleared."
        }

    def __repr__(self):

        return (
            "DatasetVersioner("
            f"path={self.path!r}, "
            f"versions={len(self.versions)}"
            ")"
        )