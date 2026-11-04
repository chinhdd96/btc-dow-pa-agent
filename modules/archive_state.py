"""Archive legacy learn-state artifacts on Champion agent startup."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_ARCHIVE_TARGETS = (
    "PAPER_ACCOUNT",
    "DECISION_HISTORY",
    "ADAPTIVE_RULES",
    "HYPOTHESES",
    "STATE",
)


def archive_learn_state(*, force: bool | None = None) -> Path | None:
    """
    One-time migrate: move legacy learn files into generated/archive/<timestamp>/.
    Skips if ARCHIVE_LEARN_STATE_ON_START is false (unless force=True),
    or if champion_migrated marker already exists.
    """
    if force is None:
        force = False
    if not force and not config.ARCHIVE_LEARN_STATE_ON_START:
        return None

    marker = Path(config.ARCHIVE_DIR) / "champion_migrated.txt"
    if marker.exists() and not force:
        logger.info("Skip archive: %s exists", marker)
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(config.ARCHIVE_DIR) / stamp
    moved = 0
    for key in _ARCHIVE_TARGETS:
        src = Path(config.PATHS[key])
        if not src.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / src.name
        try:
            shutil.move(str(src), str(target))
            moved += 1
            logger.info("Archived %s → %s", src, target)
        except OSError as exc:
            logger.warning("Failed to archive %s: %s", src, exc)

    mem = Path(config.PATHS["MEMORY"])
    if mem.exists():
        dest.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(mem), str(dest / mem.name))
            moved += 1
            logger.info("Archived memory.json → %s", dest / mem.name)
        except OSError as exc:
            logger.warning("Failed to archive memory: %s", exc)

    # Always write migration marker + empty memory skeleton once
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        from modules.memory_manager import DEFAULT_MEMORY
        import json as _json

        mem.write_text(
            _json.dumps(DEFAULT_MEMORY, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    marker.write_text(stamp, encoding="utf-8")
    if moved:
        logger.info("Archived %d learn-state file(s) to %s", moved, dest)
        return dest
    logger.info("Champion migrate marker set (no legacy files found)")
    return None
