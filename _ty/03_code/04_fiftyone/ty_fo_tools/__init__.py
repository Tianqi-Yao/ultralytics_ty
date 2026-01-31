# ty_fo_tools/__init__.py

# =====plotly utilities=====
# ty_fo_tools/__init__.py
from .plotly.data_prep import (
    add_fo_url,
    ensure_event_cols,
    ensure_time_cols,
    find_per_image_csvs,
    infer_event_label,
    safe_name,
)

from .plotly.html_components import (
    make_day_summary_html,
    make_top_table_html,
)

from .plotly.plotting_strategies import (
    PlottingStrategy,
    SchemeAGtTpStrategy,
    # 如果以后有更多策略，也可以在这里导出
    # AllPredictionsStrategy,
    # TpOnlyStrategy,
)


__all__ = [
    # =====plotly utilities=====
    "add_fo_url",
    "ensure_event_cols",
    "ensure_time_cols",
    "find_per_image_csvs",
    "infer_event_label",
    "safe_name",
    "make_day_summary_html",
    "make_top_table_html",
    "PlottingStrategy",
    "SchemeAGtTpStrategy",
]