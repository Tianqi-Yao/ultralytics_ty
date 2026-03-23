"""
文件名解析工具：将图像文件名解析为 capture_datetime 和 focus。
命名格式：MMDD_HHMM_FOCUS.jpg  例如 0729_0606_620.jpg
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"(\d{4})_(\d{4})_(\d+)\.(jpg|png)$", re.IGNORECASE)


def parse_dt_focus(
    filepath: str,
    year: int = 2024,
) -> Tuple[pd.Timestamp, Optional[int]]:
    """
    从文件名解析 (capture_datetime, focus)。
    解析失败时记录 warning 并返回 (NaT, None)。
    """
    filename = Path(filepath).name
    m = _FILENAME_RE.search(filename)

    if not m:
        logger.warning(f"文件名不符合 MMDD_HHMM_FOCUS 格式，跳过解析: {filename}")
        return pd.NaT, None

    mmdd, hhmm, focus_str = m.group(1), m.group(2), m.group(3)
    try:
        ts = pd.Timestamp(
            year,
            int(mmdd[:2]),
            int(mmdd[2:]),
            int(hhmm[:2]),
            int(hhmm[2:]),
        )
        return ts, int(focus_str)
    except Exception as e:
        logger.warning(f"解析时间失败 {filename}: {e}")
        return pd.NaT, None
