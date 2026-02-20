"""
解析文件名工具：从 MMDD_HHMM_FOCUS.jpg 格式提取时间和焦距。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"(\d{4})_(\d{4})_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_dt_focus(filepath: str, year: int = 2024) -> Tuple[pd.Timestamp, Optional[int]]:
    """解析文件名 MMDD_HHMM_FOCUS.jpg，返回 (capture_datetime, focus)。

    失败时 logger.warning() 提示（非静默），返回 (pd.NaT, None)。
    """
    name = Path(filepath).name
    m = _FILENAME_RE.search(name)
    if not m:
        logger.warning(f"文件名不符合 MMDD_HHMM_FOCUS 格式，跳过解析: {name}")
        return pd.NaT, None

    mmdd, hhmm, focus_str = m.group(1), m.group(2), m.group(3)
    try:
        dt = pd.Timestamp(
            year,
            int(mmdd[:2]),
            int(mmdd[2:]),
            int(hhmm[:2]),
            int(hhmm[2:]),
        )
        return dt, int(focus_str)
    except (ValueError, OverflowError) as e:
        logger.warning(f"时间解析失败 {name}: {e}")
        return pd.NaT, None
