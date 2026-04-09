"""
图表构建工具：生成 Plotly 图表和 HTML 汇总表格。
"""
from __future__ import annotations

import html
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

FO_BASE = "https://fiftyone.tianqiyao.men"
TOP_N   = 20
FOCUS_COL = "focus"


# ── 公共辅助函数 ──────────────────────────────────────────


def safe_name(s: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))
    return s[:200]


def make_fo_url(dataset_name: str, sample_id: str) -> str:
    return f"{FO_BASE}/datasets/{dataset_name}?id={sample_id}"


def add_fo_url(df: pd.DataFrame) -> pd.DataFrame:
    if {"dataset_name", "sample_id"}.issubset(df.columns):
        df["sample_id"] = df["sample_id"].astype(str)
        df["url"] = df.apply(lambda r: make_fo_url(r["dataset_name"], r["sample_id"]), axis=1)
    else:
        df["url"] = ""
    return df


def infer_event_label(df: pd.DataFrame) -> pd.Series:
    has_gt = "gt_count_img" in df.columns
    if has_gt:
        out = pd.Series(["Unknown"] * len(df), index=df.index)
        out[df.get("hit_img", 0).astype(int) == 1]           = "Hit"
        out[df.get("miss_img", 0).astype(int) == 1]          = "Miss"
        out[df.get("false_alarm_img", 0).astype(int) == 1]   = "False Alarm"
        out[df.get("correct_reject_img", 0).astype(int) == 1] = "Correct Reject"
        return out
    pred_present = pd.to_numeric(df.get("pred_count_img", 0), errors="coerce").fillna(0) > 0
    return pd.Series(["Hit" if b else "No-Hit" for b in pred_present], index=df.index)


def _make_hover_cols(df: pd.DataFrame) -> List[str]:
    candidates = ["focus", "filepath", "capture_datetime", "gt_count_img", "pred_count_img",
                  "tp_img", "fp_img", "fn_img", "avg_confidence", "median_confidence",
                  "confidence_threshold", "iou_threshold", "model_tag", "event", "err_obj", "tp_ratio"]
    return [c for c in candidates if c in df.columns]


def _build_hovertemplate(y_label: str, hover_cols: List[str]) -> str:
    lines = [f"<b>{y_label}</b>: %{{y}}<br>"]
    for i, c in enumerate(hover_cols, start=1):
        lines.append(f"{c}: %{{customdata[{i}]}}<br>")
    lines.append("<b>Click</b> to open FiftyOne<br>")
    return "".join(lines)


# ── 日汇总 HTML 文字块 ───────────────────────────────────


def make_day_summary_html(df_day: pd.DataFrame) -> str:
    def fnum(x, nd=3):
        try:
            return "-" if pd.isna(x) else f"{float(x):.{nd}f}"
        except Exception:
            return "-"

    n = len(df_day)
    gt_total   = df_day["gt_count_img"].sum()   if "gt_count_img"   in df_day.columns else pd.NA
    pred_total = df_day["pred_count_img"].sum() if "pred_count_img" in df_day.columns else pd.NA
    tp_total   = df_day["tp_img"].sum()         if "tp_img"         in df_day.columns else pd.NA
    fp_total   = df_day["fp_img"].sum()         if "fp_img"         in df_day.columns else pd.NA
    fn_total   = df_day["fn_img"].sum()         if "fn_img"         in df_day.columns else pd.NA

    block = [
        "<div style='border:1px solid #ddd; background:#fafafa; padding:12px; margin:10px 0 14px 0; font-family:Arial;'>",
        "<div style='font-size:14px; font-weight:700; margin-bottom:6px;'>Day summary</div>",
        "<div style='font-size:12px; line-height:1.6;'>",
        f"Images: <b>{n}</b><br>",
    ]
    if "gt_count_img"   in df_day.columns: block.append(f"GT total / avg: <b>{int(gt_total)}</b> / <b>{fnum(gt_total/max(n,1))}</b><br>")
    if "pred_count_img" in df_day.columns: block.append(f"Pred total / avg: <b>{int(pred_total)}</b> / <b>{fnum(pred_total/max(n,1))}</b><br>")
    if "tp_img"  in df_day.columns: block.append(f"TP total / avg: <b>{int(tp_total)}</b> / <b>{fnum(tp_total/max(n,1))}</b><br>")
    if "fp_img"  in df_day.columns: block.append(f"FP total / avg: <b>{int(fp_total)}</b> / <b>{fnum(fp_total/max(n,1))}</b><br>")
    if "fn_img"  in df_day.columns: block.append(f"FN total / avg: <b>{int(fn_total)}</b> / <b>{fnum(fn_total/max(n,1))}</b><br>")
    block += ["</div></div>"]
    return "\n".join(block)


# ── Top N 错误表格 ────────────────────────────────────────


def make_top_tables_blocks_html(df_day: pd.DataFrame, top_n: int = 20, focus_col: str = FOCUS_COL) -> str:
    d0 = df_day.copy()
    d0 = add_fo_url(d0)
    if focus_col not in d0.columns:
        d0[focus_col] = "unknown"
    d0[focus_col] = d0[focus_col].astype(str).fillna("unknown")
    for c in ["pred_count_img", "gt_count_img", "avg_confidence", "tp_img", "fp_img", "fn_img", "tp_ratio"]:
        if c in d0.columns:
            d0[c] = pd.to_numeric(d0[c], errors="coerce")
    d0["event"]   = infer_event_label(d0)
    d0["err_obj"] = pd.to_numeric(d0.get("fp_img", 0), errors="coerce").fillna(0) + \
                    pd.to_numeric(d0.get("fn_img", 0), errors="coerce").fillna(0)

    def esc(x): return html.escape("" if pd.isna(x) else str(x))
    def short_path(p, maxlen=70):
        p = "" if pd.isna(p) else str(p)
        return p if len(p) <= maxlen else "…" + p[-(maxlen - 1):]

    headers = ["Event", focus_col, "capture_datetime", "filepath", "gt", "pred", "tp", "fp", "fn", "fp+fn", "tp/gt", "avg_conf", "Link"]

    def build_table(df_sub, scope, title):
        dd = df_sub.copy()
        dd["_err"] = pd.to_numeric(dd.get("err_obj", 0), errors="coerce").fillna(0)
        dd["_fn"]  = pd.to_numeric(dd.get("fn_img",  0), errors="coerce").fillna(0)
        dd["_fp"]  = pd.to_numeric(dd.get("fp_img",  0), errors="coerce").fillna(0)
        dd = dd.sort_values(["_err", "_fn", "_fp"], ascending=False).head(top_n)
        rows = []
        for _, r in dd.iterrows():
            url  = r.get("url", "")
            link = f'<a href="{esc(url)}" target="_blank">Open</a>' if url else ""
            cap  = "" if pd.isna(r.get("capture_datetime", "")) else str(r.get("capture_datetime", ""))
            tp_r = r.get("tp_ratio", "")
            tp_r = "" if pd.isna(tp_r) else f"{float(tp_r):.3f}"
            rows.append([esc(r.get("event","")), esc(r.get(focus_col,"")), esc(cap),
                         esc(short_path(r.get("filepath",""))),
                         esc(r.get("gt_count_img","")), esc(r.get("pred_count_img","")),
                         esc(r.get("tp_img","")), esc(r.get("fp_img","")), esc(r.get("fn_img","")),
                         esc(r.get("err_obj","")), esc(tp_r), esc(r.get("avg_confidence","")), link])
        buf = [f"<div class='top_table_block' data-scope='{esc(scope)}' style='display:none;'>",
               f"<h3 style='margin-top:18px;'>{esc(title)}</h3>",
               "<div style='overflow:auto; max-height:460px; border:1px solid #ddd;'>",
               "<table style='border-collapse:collapse; width:100%; font-family:Arial; font-size:12px;'>",
               "<thead><tr>"]
        for h in headers:
            buf.append(f"<th style='position:sticky; top:0; background:#f7f7f7; border:1px solid #ddd; padding:6px; text-align:left;'>{esc(h)}</th>")
        buf.append("</tr></thead><tbody>")
        for row in rows:
            buf.append("<tr>" + "".join(f"<td style='border:1px solid #ddd; padding:6px; vertical-align:top;'>{cell}</td>" for cell in row) + "</tr>")
        buf.append("</tbody></table></div></div>")
        return "\n".join(buf)

    focuses = sorted(d0[focus_col].unique().tolist())
    blocks = [build_table(d0, "TOTAL", f"TOTAL (all focus) | Top {top_n} by (fp+fn)")]
    for f in focuses:
        blocks.append(build_table(d0[d0[focus_col] == f], f"focus={f}", f"focus={f} | Top {top_n} by (fp+fn)"))
    return "\n".join(blocks)


# ── 三张日图（GT/Pred, TP/GT, FP/FN）────────────────────


def build_day_figures(df_day: pd.DataFrame, focus_col: str = FOCUS_COL) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    返回三张 Plotly Figure（fig0: GT vs Pred, fig1: GT vs TP, fig2: FP/FN）。
    每张图按 scope（TOTAL + 各 focus）添加 traces，默认只显示 TOTAL。
    """
    def _x_for_df(df):
        if "capture_datetime" in df.columns:
            col = pd.to_datetime(df["capture_datetime"], errors="coerce")
            if col.notna().any():
                return col, "capture_datetime", df.copy()
        df = df.copy()
        df["_index"] = range(len(df))
        return df["_index"], "index", df

    def _numeric(df):
        for c in ["gt_count_img", "pred_count_img", "tp_img", "fp_img", "fn_img", "avg_confidence"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df

    d = df_day.copy()
    d = add_fo_url(d)
    if focus_col not in d.columns:
        d[focus_col] = "unknown"
    d[focus_col] = d[focus_col].astype(str).fillna("unknown")
    d = _numeric(d)
    d["event"]   = infer_event_label(d)
    d["err_obj"] = d.get("fp_img", 0).fillna(0) + d.get("fn_img", 0).fillna(0)
    if "gt_count_img" in d.columns and "tp_img" in d.columns:
        gt = pd.to_numeric(d["gt_count_img"], errors="coerce").fillna(0)
        tp = pd.to_numeric(d["tp_img"], errors="coerce").fillna(0)
        d["tp_ratio"] = np.where(gt > 0, tp / gt, np.nan)
    else:
        d["tp_ratio"] = pd.NA

    hover_cols = _make_hover_cols(d)
    focus_values = sorted(d[focus_col].unique().tolist())
    x_all, x_title, d = _x_for_df(d)

    fig0, fig1, fig2 = go.Figure(), go.Figure(), go.Figure()

    def add_traces(fig, scope, dsub, xsub, which):
        custom  = dsub[["url"] + hover_cols].values
        prefix  = f"{scope} | "
        visible = (scope == "TOTAL")

        if which == "fig0":
            for col, nm in [("gt_count_img", "GT count"), ("pred_count_img", "Pred count")]:
                if col in dsub.columns:
                    fig.add_trace(go.Scatter(x=xsub, y=dsub[col], mode="lines+markers",
                                             name=prefix+nm, customdata=custom,
                                             hovertemplate=_build_hovertemplate(prefix+nm, hover_cols),
                                             visible=visible))
        elif which == "fig1":
            for col, nm in [("gt_count_img","GT count"), ("tp_img","TP"), ("tp_ratio","TP/GT")]:
                if col in dsub.columns:
                    kw = {"yaxis": "y2"} if col == "tp_ratio" else {}
                    fig.add_trace(go.Scatter(x=xsub, y=dsub[col], mode="lines+markers",
                                             name=prefix+nm, customdata=custom,
                                             hovertemplate=_build_hovertemplate(prefix+nm, hover_cols),
                                             visible=visible, **kw))
        elif which == "fig2":
            for col, nm in [("fp_img","FP"), ("fn_img","FN"), ("err_obj","FP+FN")]:
                if col in dsub.columns:
                    fig.add_trace(go.Scatter(x=xsub, y=dsub[col], mode="lines+markers",
                                             name=prefix+nm, customdata=custom,
                                             hovertemplate=_build_hovertemplate(prefix+nm, hover_cols),
                                             visible=visible))

    for which, fig in [("fig0", fig0), ("fig1", fig1), ("fig2", fig2)]:
        add_traces(fig, "TOTAL", d, x_all, which)
        for f in focus_values:
            sub = d[d[focus_col] == f].copy()
            xsub, _, sub = _x_for_df(sub)
            sub = _numeric(sub)
            sub["event"]   = infer_event_label(sub)
            sub["err_obj"] = sub.get("fp_img", 0).fillna(0) + sub.get("fn_img", 0).fillna(0)
            add_traces(fig, f"focus={f}", sub, xsub, which)

    title_base = ""
    fig0.update_layout(title=f"{title_base}GT vs Pred", xaxis=dict(title=x_title, rangeslider=dict(visible=True)),
                       yaxis=dict(title="Count"), template="plotly_white")
    fig1.update_layout(title=f"{title_base}GT vs TP (+ TP/GT right axis)",
                       xaxis=dict(title=x_title, rangeslider=dict(visible=True)),
                       yaxis=dict(title="Count"), yaxis2=dict(title="TP/GT", overlaying="y", side="right"),
                       template="plotly_white")
    fig2.update_layout(title=f"{title_base}FP / FN / (FP+FN)",
                       xaxis=dict(title=x_title, rangeslider=dict(visible=True)),
                       yaxis=dict(title="Count"), template="plotly_white")

    return fig0, fig1, fig2, focus_values, x_title


# ── 日概览（Daily Overview）────────────────────────────────


def write_daily_overview_html(
    df: pd.DataFrame,
    out_html: Path,
    day_to_file: Dict[str, str],
    title: str,
) -> None:
    daily = (
        df.groupby("capture_date", dropna=False)
        .agg(
            images=(("sample_id", "count") if "sample_id" in df.columns else ("filepath", "count")),
            hit=("hit_img", "sum"), miss=("miss_img", "sum"),
            false_alarm=("false_alarm_img", "sum"), correct_reject=("correct_reject_img", "sum"),
            gt_total=(("gt_count_img", "sum") if "gt_count_img" in df.columns else ("hit_img", "sum")),
            pred_total=(("pred_count_img", "sum") if "pred_count_img" in df.columns else ("hit_img", "sum")),
            tp_total=(("tp_img", "sum") if "tp_img" in df.columns else ("hit_img", "sum")),
            fp_total=(("fp_img", "sum") if "fp_img" in df.columns else ("hit_img", "sum")),
            fn_total=(("fn_img", "sum") if "fn_img" in df.columns else ("hit_img", "sum")),
        ).reset_index()
    ).sort_values("capture_date")

    daily["hit_rate"]   = daily["hit"] / daily["images"].clip(lower=1)
    daily["error_rate"] = (daily["miss"] + daily["false_alarm"]) / daily["images"].clip(lower=1)
    denom = daily["tp_total"] + daily["fp_total"] + daily["fn_total"]
    daily["obj_error_rate"] = np.where(denom > 0, (daily["fp_total"] + daily["fn_total"]) / denom, np.nan)
    daily["avg_gt_per_img"]   = daily["gt_total"]   / daily["images"].clip(lower=1)
    daily["avg_pred_per_img"] = daily["pred_total"] / daily["images"].clip(lower=1)
    daily["drill_html"] = daily["capture_date"].map(day_to_file).fillna("")
    x = daily["capture_date"]

    hover_pack = daily[["drill_html","images","hit","miss","false_alarm","correct_reject",
                         "hit_rate","error_rate","gt_total","pred_total","tp_total","fp_total",
                         "fn_total","avg_gt_per_img","avg_pred_per_img","obj_error_rate"]].values

    def daily_ht(nm): return (
        "Date: %{x}<br><b>" + nm + "</b>: %{y}<br>"
        "Images: %{customdata[1]}<br>"
        "Hit/Miss/FA/CR: %{customdata[2]}/%{customdata[3]}/%{customdata[4]}/%{customdata[5]}<br>"
        "Hit rate: %{customdata[6]:.3f} | Error rate: %{customdata[7]:.3f}<br>"
        "TP/FP/FN: %{customdata[10]}/%{customdata[11]}/%{customdata[12]}<br>"
        "<b>Click</b> to drill down<br><extra></extra>"
    )

    fig = go.Figure()
    for col, nm in [("hit","Hit"),("miss","Miss"),("false_alarm","False Alarm"),("correct_reject","Correct Reject")]:
        fig.add_trace(go.Bar(x=x, y=daily[col], name=nm, customdata=hover_pack, hovertemplate=daily_ht(nm)))
    fig.add_trace(go.Scatter(x=x, y=daily["hit_rate"], name="Hit rate", mode="lines+markers",
                             yaxis="y2", customdata=hover_pack, hovertemplate=daily_ht("Hit rate")))
    fig.add_trace(go.Scatter(x=x, y=daily["error_rate"], name="Error rate", mode="lines+markers",
                             yaxis="y2", customdata=hover_pack, hovertemplate=daily_ht("Error rate")))
    fig.add_trace(go.Scatter(x=x, y=daily["images"], mode="text",
                             text=[f"{int(v)}" for v in daily["images"].fillna(0)],
                             textposition="top center", showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=title, barmode="stack",
                      xaxis=dict(title="capture_date", rangeslider=dict(visible=True)),
                      yaxis=dict(title="Images (stacked)"),
                      yaxis2=dict(title="Rate", overlaying="y", side="right", range=[0,1]),
                      template="plotly_white")

    post_script = """
    document.addEventListener("DOMContentLoaded", function() {
        var plot = document.getElementsByClassName('js-plotly-plot')[0];
        if (!plot) return;
        plot.on('plotly_click', function(e) {
            var rel = e?.points?.[0]?.customdata?.[0];
            if (rel) window.open(rel, '_blank');
        });
    });
    """
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True, post_script=post_script)
    logger.info(f"[SAVE][daily] {out_html}")


# ── 分钟聚合概览 ──────────────────────────────────────────


def write_minutely_swd_overview_html(
    df: pd.DataFrame,
    out_html: Path,
    title: str,
    focus_col: str = FOCUS_COL,
    pred_col: str = "pred_count_img",
    gt_col: str = "gt_count_img",
) -> None:
    d = df.copy()
    if "capture_datetime" not in d.columns:
        logger.warning("缺少 capture_datetime 列，跳过分钟图生成")
        return
    d["capture_datetime"] = pd.to_datetime(d["capture_datetime"], errors="coerce")
    d = d[d["capture_datetime"].notna()].copy()
    if focus_col not in d.columns:
        d[focus_col] = "unknown"
    d[focus_col] = d[focus_col].astype(str).fillna("unknown")
    d[pred_col] = pd.to_numeric(d.get(pred_col, 0), errors="coerce").fillna(0)
    d[gt_col]   = pd.to_numeric(d.get(gt_col,   0), errors="coerce").fillna(0)
    d["minute"] = d["capture_datetime"].dt.floor("min")

    def agg(dd):
        return (dd.groupby("minute").agg(
            pred=(pred_col, "sum"), gt=(gt_col, "sum"), images=("filepath","count")
        ).reset_index().sort_values("minute"))

    total_m      = agg(d)
    focus_values = sorted(d[focus_col].unique().tolist())
    fig = go.Figure()

    ht = "Minute: %{x}<br><b>%{fullData.name}</b>: %{y}<br>Images: %{customdata[0]}<extra></extra>"
    fig.add_trace(go.Scatter(x=total_m["minute"], y=total_m["pred"], mode="lines+markers",
                             name="TOTAL | Pred/min", customdata=total_m[["images"]].values,
                             hovertemplate=ht, visible=True))
    fig.add_trace(go.Scatter(x=total_m["minute"], y=total_m["gt"], mode="lines",
                             line=dict(dash="dash"), name="TOTAL | GT/min",
                             customdata=total_m[["images"]].values, hovertemplate=ht, visible=True))

    for f in focus_values:
        m = agg(d[d[focus_col] == f])
        fig.add_trace(go.Scatter(x=m["minute"], y=m["pred"], mode="lines+markers",
                                 name=f"focus={f} | Pred/min", customdata=m[["images"]].values,
                                 hovertemplate=ht, visible=False))
        fig.add_trace(go.Scatter(x=m["minute"], y=m["gt"], mode="lines", line=dict(dash="dash"),
                                 name=f"focus={f} | GT/min", customdata=m[["images"]].values,
                                 hovertemplate=ht, visible=False))

    n = len(fig.data)
    vis_total = [False]*n; vis_total[0] = True; vis_total[1] = True
    buttons = [dict(label="TOTAL", method="update", args=[{"visible": vis_total}])]
    for idx, f in enumerate(focus_values):
        vis = [False]*n
        vis[2 + idx*2] = True; vis[2 + idx*2 + 1] = True
        buttons.append(dict(label=f"focus={f}", method="update", args=[{"visible": vis}]))

    fig.update_layout(title=title, updatemenus=[dict(buttons=buttons, x=0.01, y=1.15, active=0)],
                      xaxis=dict(title="minute", rangeslider=dict(visible=True)),
                      yaxis=dict(title="SWD count per minute"), template="plotly_white")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    logger.info(f"[SAVE][minutely] {out_html}")
