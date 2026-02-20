"""
传感器数据可视化辅助函数：时序统计聚合（供 Plotly 绘图使用）。
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def _to_time_of_day_minutes(series: pd.Series) -> pd.Series:
    return series.dt.hour * 60 + series.dt.minute


def prepare_sensor_plot_data(df_source: pd.DataFrame, bin_min: int = 5) -> pd.DataFrame:
    """清洗传感器数据并计算绘图所需的时间特征列。"""
    plot_df = df_source.copy()
    plot_df["datetime"]    = pd.to_datetime(plot_df["datetime"],    errors="coerce")
    plot_df["temperature"] = pd.to_numeric(plot_df["temperature"],  errors="coerce")
    plot_df["humidity"]    = pd.to_numeric(plot_df["humidity"],     errors="coerce")
    plot_df = plot_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    plot_df["date"]    = plot_df["datetime"].dt.date
    plot_df["tod"]     = _to_time_of_day_minutes(plot_df["datetime"])
    plot_df["tod_bin"] = (plot_df["tod"] // bin_min) * bin_min
    plot_df["month"]   = plot_df["datetime"].dt.to_period("M").astype(str)
    return plot_df


def build_daily_profile(df_source: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """按 (date, tod) 分组取均值，供每日折线图使用。"""
    return (
        df_source.dropna(subset=[value_col])
        .groupby(["date", "tod"], as_index=False)[value_col]
        .mean()
        .sort_values(["date", "tod"])
        .reset_index(drop=True)
    )


def build_band_stats(df_source: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """按 tod_bin 聚合 min/max/mean，供带状图（min-max band）使用。"""
    daily = (
        df_source.dropna(subset=[value_col])
        .groupby(["date", "tod_bin"], as_index=False)[value_col]
        .mean()
    )
    return (
        daily.groupby("tod_bin")[value_col]
        .agg(min="min", max="max", mean="mean")
        .reset_index()
        .sort_values("tod_bin")
        .reset_index(drop=True)
    )


def build_monthly_band_stats(df_source: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """按 (month, tod_bin) 聚合 min/max/mean，供月度带状图使用。"""
    daily = (
        df_source.dropna(subset=[value_col])
        .groupby(["month", "date", "tod_bin"], as_index=False)[value_col]
        .mean()
    )
    return (
        daily.groupby(["month", "tod_bin"])[value_col]
        .agg(min="min", max="max", mean="mean")
        .reset_index()
        .sort_values(["tod_bin", "month"])
        .reset_index(drop=True)
    )


def get_time_ticks(step_min: int = 120) -> Tuple[List[int], List[str]]:
    """生成时间轴刻度（tickvals 和 ticktext），step_min 为间隔分钟数。"""
    tickvals = list(range(0, 1440, step_min))
    ticktext = [f"{m // 60:02d}:{m % 60:02d}" for m in tickvals]
    return tickvals, ticktext
