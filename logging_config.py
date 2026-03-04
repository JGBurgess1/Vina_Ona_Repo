"""
Shared logging configuration for the docking pipeline.

Provides:
  - configure_logging(): sets up console + rotating file handlers
  - ProgressTracker: logs at milestone percentages (not per-item)
  - Log files with timestamps, rotation, and per-rank support for MPI
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Optional


def configure_logging(
    log_dir: str,
    log_name: str = "pipeline",
    rank: int = 0,
    mpi_size: int = 1,
    verbose: bool = False,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configure logging with console and file handlers.

    Creates a main log file that all ranks write to (rank 0 at INFO,
    workers at WARNING). In verbose MPI mode, each rank also gets its
    own log file at DEBUG level.

    Args:
        log_dir: directory for log files
        log_name: base name for the log file (e.g., "docking_campaign")
        rank: MPI rank (0 for serial)
        mpi_size: total MPI ranks
        verbose: enable DEBUG level and per-rank log files
        console: enable console output
        max_bytes: max log file size before rotation
        backup_count: number of rotated backups to keep

    Returns:
        The root logger.
    """
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    # Clear existing handlers to avoid duplicates on re-init
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [Rank %(rank)03d] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Inject rank into all log records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)

    # --- Main log file (all ranks write, rank 0 at INFO, workers at WARNING) ---
    main_log_path = os.path.join(log_dir, f"{log_name}.log")
    file_handler = RotatingFileHandler(
        main_log_path, maxBytes=max_bytes, backupCount=backup_count,
    )
    file_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # --- Console handler ---
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        if rank != 0:
            console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

    # --- Per-rank log file (MPI verbose mode only) ---
    if verbose and mpi_size > 1:
        rank_log_path = os.path.join(log_dir, f"rank_{rank:04d}.log")
        rank_handler = RotatingFileHandler(
            rank_log_path, maxBytes=max_bytes, backupCount=backup_count,
        )
        rank_handler.setLevel(logging.DEBUG)
        rank_handler.setFormatter(fmt)
        root.addHandler(rank_handler)

    if rank == 0:
        logging.getLogger(__name__).info(
            "Logging configured: %s (ranks=%d, verbose=%s)",
            main_log_path, mpi_size, verbose,
        )

    return root


class ProgressTracker:
    """
    Tracks progress and logs at milestone percentages.

    Instead of logging every item, logs at configurable milestones
    (default: 10%, 25%, 50%, 75%, 90%, 100%). Also logs elapsed time
    and estimated time remaining at each milestone.

    Usage:
        tracker = ProgressTracker(total=50000, label="Docking")
        for i, ligand in enumerate(ligands):
            dock(ligand)
            tracker.update(i + 1)
        tracker.finish()
    """

    DEFAULT_MILESTONES = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        logger_name: Optional[str] = None,
        milestones: Optional[tuple] = None,
        rank: int = 0,
    ):
        self.total = total
        self.label = label
        self.rank = rank
        self.logger = logging.getLogger(logger_name or __name__)
        self.milestones = milestones or self.DEFAULT_MILESTONES
        self._start_time = time.time()
        self._last_milestone_idx = -1
        self._logged_start = False

    def update(self, current: int) -> None:
        """
        Update progress. Logs if a new milestone has been reached.
        """
        if not self._logged_start and current > 0:
            self.logger.info(
                "[%s] Rank %d: started (%d items)",
                self.label, self.rank, self.total,
            )
            self._logged_start = True

        if self.total <= 0:
            return

        fraction = current / self.total
        for idx, milestone in enumerate(self.milestones):
            if idx > self._last_milestone_idx and fraction >= milestone:
                elapsed = time.time() - self._start_time
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (self.total - current) / rate if rate > 0 else 0

                self.logger.info(
                    "[%s] Rank %d: %d/%d (%.0f%%) — %.1fs elapsed, "
                    "%.1f items/s, ~%.0fs remaining",
                    self.label, self.rank,
                    current, self.total,
                    fraction * 100,
                    elapsed, rate, remaining,
                )
                self._last_milestone_idx = idx

    def finish(self, n_success: int = None, n_failed: int = None) -> None:
        """Log completion with final statistics."""
        elapsed = time.time() - self._start_time
        rate = self.total / elapsed if elapsed > 0 else 0

        parts = [
            f"[{self.label}] Rank {self.rank}: complete — "
            f"{self.total} items in {elapsed:.1f}s ({rate:.1f} items/s)"
        ]
        if n_success is not None:
            parts.append(f"success={n_success}")
        if n_failed is not None:
            parts.append(f"failed={n_failed}")

        self.logger.info(", ".join(parts))


def log_phase(logger: logging.Logger, phase: int, description: str) -> None:
    """Log a pipeline phase transition with a visible separator."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE %d: %s", phase, description)
    logger.info("=" * 60)


def log_config_summary(logger: logging.Logger, **kwargs) -> None:
    """Log key configuration parameters."""
    logger.info("Configuration:")
    for key, value in kwargs.items():
        logger.info("  %-25s %s", key, value)


def log_final_summary(
    logger: logging.Logger,
    program: str,
    wall_time: float,
    **kwargs,
) -> None:
    """Log final run summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("%s COMPLETE", program.upper())
    logger.info("=" * 60)
    logger.info("  Wall time: %.1fs", wall_time)
    for key, value in kwargs.items():
        logger.info("  %-25s %s", key, value)
    logger.info("=" * 60)
