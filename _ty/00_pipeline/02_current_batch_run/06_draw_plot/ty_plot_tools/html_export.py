"""
HTML 导出工具：将 Plotly 图表 + 数据表格组合成完整的交互式 HTML 页面。
"""
from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .figure_builder import (
    add_fo_url,
    make_day_summary_html,
    make_top_tables_blocks_html,
    build_day_figures,
    FOCUS_COL,
    TOP_N,
)

logger = logging.getLogger(__name__)


def _build_focus_bar_html(focus_values: List[str]) -> str:
    opt_lines = ["<option value='TOTAL'>TOTAL</option>"]
    for f in focus_values:
        opt_lines.append(f"<option value='focus={html.escape(f)}'>focus={html.escape(f)}</option>")
    options_html = "\n".join(opt_lines)
    return f"""
<div style="border:1px solid #ddd; background:#f0f8ff; padding:10px; margin:10px 0 12px 0; font-family:Arial;">
  <div style="display:flex; gap:10px; align-items:center;">
    <div style="font-weight:700; color:#003366;">Focus filter</div>
    <select id="global_focus_select" style="padding:4px 10px; font-size:12px;">
      {options_html}
    </select>
    <div style="font-size:12px; color:#333;">(controls all charts + top table)</div>
  </div>
</div>
"""


def _build_global_js() -> str:
    return """
<script>
(function() {
  function showTable(scope) {
    var blocks = document.getElementsByClassName('top_table_block');
    for (var i=0;i<blocks.length;i++) blocks[i].style.display='none';
    for (var j=0;j<blocks.length;j++) {
      if ((blocks[j].getAttribute('data-scope')||'')===scope) { blocks[j].style.display='block'; return; }
    }
  }
  function set_focus(scope) {
    ["plot_fig0","plot_fig1","plot_fig2"].forEach(function(pid){
      var plot=document.getElementById(pid);
      if(!plot||!plot.data) return;
      var vis=[];
      for(var i=0;i<plot.data.length;i++) vis.push((plot.data[i].name||"").startsWith(scope+" | "));
      Plotly.restyle(plot,{visible:vis});
    });
    showTable(scope);
  }
  var sel=document.getElementById("global_focus_select");
  if(sel) sel.addEventListener("change",function(){ set_focus(this.value); });
  set_focus("TOTAL");
  ["plot_fig0","plot_fig1","plot_fig2"].forEach(function(pid){
    var plot=document.getElementById(pid);
    if(!plot) return;
    plot.on('plotly_click',function(e){
      var url=e?.points?.[0]?.customdata?.[0];
      if(url) window.open(url,'_blank');
    });
  });
})();
</script>
"""


def write_image_level_day_html(
    df_day: pd.DataFrame,
    out_html: Path,
    title: str,
    top_n: int = TOP_N,
    focus_col: str = FOCUS_COL,
) -> None:
    """
    生成单日 per-image 交互式 HTML 报告。
    包含：focus 过滤器 + 日汇总 + 三张时序图 + Top N 错误表格。
    """
    df_day = df_day.copy()
    df_day = add_fo_url(df_day)
    if focus_col not in df_day.columns:
        df_day[focus_col] = "unknown"
    df_day[focus_col] = df_day[focus_col].astype(str).fillna("unknown")

    try:
        fig0, fig1, fig2, focus_values, x_title = build_day_figures(df_day, focus_col)
    except Exception as e:
        logger.error(f"build_day_figures 失败 {out_html}: {e}")
        return

    html0 = fig0.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot_fig0")
    html1 = fig1.to_html(include_plotlyjs=False, full_html=False, div_id="plot_fig1")
    html2 = fig2.to_html(include_plotlyjs=False, full_html=False, div_id="plot_fig2")

    summary_html     = make_day_summary_html(df_day)
    top_tables_html  = make_top_tables_blocks_html(df_day, top_n=top_n, focus_col=focus_col)
    focus_bar_html   = _build_focus_bar_html(focus_values)
    global_js        = _build_global_js()

    full = [
        "<html><head><meta charset='utf-8'><title>Image-level</title></head><body>",
        focus_bar_html, summary_html, html0, html1, html2, top_tables_html, global_js,
        "</body></html>",
    ]

    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_html.write_text("\n".join(full), encoding="utf-8")
        logger.info(f"[SAVE][image-day] {out_html}")
    except Exception as e:
        logger.error(f"写入 HTML 失败 {out_html}: {e}")
