import json
import time
import shutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any


class TrainingLogger:
    """
    A simple interface for logging RL training data compatible with the monitoring system.

    This class handles creating and managing log files in the required format for the
    RL training monitoring dashboard.
    """

    def __init__(self,
                 experiment_name: str = "training",
                 log_dir: str = ".",
                 number_of_steps: int = 100):
        """
        Initialize training logger.

        Args:
            experiment_name: Base name for log files (default: "training")
            log_dir: Directory to store files (default: current directory)
            number_of_steps: Total number of training steps
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.number_of_steps = number_of_steps

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.metadata_file = self.log_dir / f"{experiment_name}.metadata"

        print(str(self.log_dir), str(self.log_file), str(self.metadata_file))

        # Thread lock for safe concurrent access
        self._lock = threading.Lock()

        # Initialize metadata
        self._create_metadata()

        # Clear existing log file
        self.clean_logs()

    def _create_metadata(self):
        """Create or update the metadata file."""
        metadata = {
            "number_of_steps": self.number_of_steps,
            "experiment_name": self.experiment_name,
            "created_at": time.time()
        }

        with self._lock:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

    def log_step(self, step: int, error: float, timestamp: Optional[float] = None):
        """
        Log a training step with.

        Args:
            step: Current training step number
            error: Error/loss value for this step
            timestamp: Optional timestamp (defaults to current time)

        Format: "step: N; error: X.X"
        """
        if timestamp is None:
            timestamp = time.time()

        # Format the log entry to match expected format
        log_entry = f"step: {step}; error: {error:.6f}"

        with self._lock:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\n")
                f.flush()  # Ensure immediate write for real-time monitoring

    def update_metadata(self, **kwargs):
        """
        Update training metadata.

        Args:
            **kwargs: Key-value pairs to update in metadata
                     Common keys: number_of_steps, experiment_name
        """
        with self._lock:
            # Load existing metadata
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                metadata = {}

            # Update with new values
            metadata.update(kwargs)
            metadata["updated_at"] = time.time()

            # Update number_of_steps if provided
            if "number_of_steps" in kwargs:
                self.number_of_steps = kwargs["number_of_steps"]

            # Write back to file
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

    def backup_logs(self, suffix: Optional[str] = None):
        """
        Create backup of current log files.

        Args:
            suffix: Optional suffix for backup files (defaults to timestamp)

        Returns:
            tuple: (backup_log_path, backup_metadata_path)
        """
        if suffix is None:
            suffix = str(int(time.time()))

        backup_log = self.log_dir / f"{self.experiment_name}_{suffix}.log"
        backup_metadata = self.log_dir / f"{self.experiment_name}_{suffix}.metadata"

        with self._lock:
            if self.log_file.exists():
                shutil.copy2(self.log_file, backup_log)

            if self.metadata_file.exists():
                shutil.copy2(self.metadata_file, backup_metadata)

        return backup_log, backup_metadata

    def clean_logs(self):
        """Remove current log files (keeps metadata)."""
        with self._lock:
            if self.log_file.exists():
                self.log_file.unlink()

    def reset(self):
        """Reset the logger by cleaning logs and recreating metadata."""
        self.clean_logs()
        self._create_metadata()

    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current log file.

        Returns:
            dict: Statistics including line count, file size, etc.
        """
        stats = {
            "log_file": str(self.log_file),
            "metadata_file": str(self.metadata_file),
            "exists": self.log_file.exists(),
            "line_count": 0,
            "file_size": 0,
            "number_of_steps": self.number_of_steps
        }

        if self.log_file.exists():
            stats["file_size"] = self.log_file.stat().st_size
            with open(self.log_file, 'r') as f:
                stats["line_count"] = sum(1 for _ in f)

        return stats

    def close(self):
        """Close logger and finalize files."""
        # with self._lock:
        self.update_metadata(completed_at=time.time())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
