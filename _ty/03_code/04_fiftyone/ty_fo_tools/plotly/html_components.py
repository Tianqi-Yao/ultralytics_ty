# html_components.py
import html
from typing import Optional

import pandas as pd


def make_top_table_html(df_day: pd.DataFrame, top_n: int = 20) -> str:
    d = df_day.copy()

    for c in ["pred_count_img", "gt_count_img", "avg_confidence", "tp_img", "fp_img", "fn_img", "err_obj", "tp_ratio"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["event"] = infer_event_label(d)  # Assuming infer_event_label is imported
    d["err_obj"] = pd.to_numeric(d.get("fp_img", 0), errors="coerce").fillna(0) + pd.to_numeric(d.get("fn_img", 0), errors="coerce").fillna(0)

    d["_err"] = d["err_obj"].fillna(0)
    d["_fn"] = pd.to_numeric(d.get("fn_img", 0), errors="coerce").fillna(0)
    d["_fp"] = pd.to_numeric(d.get("fp_img", 0), errors="coerce").fillna(0)

    d = d.sort_values(["_err", "_fn", "_fp"], ascending=[False, False, False]).head(top_n)

    def esc(x: object) -> str:
        return html.escape("" if pd.isna(x) else str(x))

    def short_path(p: str, maxlen: int = 70) -> str:
        p = "" if pd.isna(p) else str(p)
        if len(p) <= maxlen:
            return p
        return "…" + p[-(maxlen - 1):]

    headers = [
        "Event", "capture_datetime", "filepath",
        "gt", "pred", "tp", "fp", "fn", "fp+fn",
        "tp/gt", "avg_conf", "Link"
    ]

    rows = []
    for _, r in d.iterrows():
        url = r.get("url", "")
        link = f'<a href="{esc(url)}" target="_blank">Open</a>' if url else ""
        fp = short_path(r.get("filepath", ""), 70)

        cap = r.get("capture_datetime", "")
        cap = "" if pd.isna(cap) else str(cap)

        tp_ratio = r.get("tp_ratio", "")
        tp_ratio = "" if pd.isna(tp_ratio) else f"{float(tp_ratio):.3f}"

        rows.append([
            esc(r.get("event", "")),
            esc(cap),
            esc(fp),
            esc(r.get("gt_count_img", "")) if "gt_count_img" in d.columns else "",
            esc(r.get("pred_count_img", "")) if "pred_count_img" in d.columns else "",
            esc(r.get("tp_img", "")) if "tp_img" in d.columns else "",
            esc(r.get("fp_img", "")) if "fp_img" in d.columns else "",
            esc(r.get("fn_img", "")) if "fn_img" in d.columns else "",
            esc(r.get("err_obj", "")),
            esc(tp_ratio),
            esc(r.get("avg_confidence", "")),
            link,
        ])

    table_html = []
    table_html.append("<h3 style='margin-top:18px;'>Top images by (fp+fn) (click Open)</h3>")
    table_html.append("<div style='overflow:auto; max-height:460px; border:1px solid #ddd;'>")
    table_html.append("<table style='border-collapse:collapse; width:100%; font-family:Arial; font-size:12px;'>")
    table_html.append("<thead><tr>")
    for h in headers:
        table_html.append(
            f"<th style='position:sticky; top:0; background:#f7f7f7; border:1px solid #ddd; padding:6px; text-align:left;'>{esc(h)}</th>"
        )
    table_html.append("</tr></thead><tbody>")

    for row in rows:
        table_html.append("<tr>")
        for cell in row:
            table_html.append(f"<td style='border:1px solid #ddd; padding:6px; vertical-align:top;'>{cell}</td>")
        table_html.append("</tr>")

    table_html.append("</tbody></table></div>")
    return "\n".join(table_html)


def make_day_summary_html(df_day: pd.DataFrame, corr_gt_tp: Optional[float]) -> str:
    def fnum(x: object, nd: int = 3) -> str:
        try:
            if pd.isna(x):
                return "-"
            return f"{float(x):.{nd}f}"
        except Exception:
            return "-"

    n = len(df_day)
    gt_total = df_day.get("gt_count_img", pd.Series([pd.NA]*n)).sum() if "gt_count_img" in df_day.columns else pd.NA
    tp_total = df_day.get("tp_img", pd.Series([pd.NA]*n)).sum() if "tp_img" in df_day.columns else pd.NA
    fp_total = df_day.get("fp_img", pd.Series([pd.NA]*n)).sum() if "fp_img" in df_day.columns else pd.NA
    fn_total = df_day.get("fn_img", pd.Series([pd.NA]*n)).sum() if "fn_img" in df_day.columns else pd.NA

    avg_gt = (gt_total / max(n, 1)) if gt_total is not pd.NA else pd.NA
    avg_tp = (tp_total / max(n, 1)) if tp_total is not pd.NA else pd.NA
    avg_fp = (fp_total / max(n, 1)) if fp_total is not pd.NA else pd.NA
    avg_fn = (fn_total / max(n, 1)) if fn_total is not pd.NA else pd.NA

    # tp_ratio summary (only where gt>0)
    if "tp_ratio" in df_day.columns:
        tp_ratio_mean = pd.to_numeric(df_day["tp_ratio"], errors="coerce").dropna().mean()
        tp_ratio_median = pd.to_numeric(df_day["tp_ratio"], errors="coerce").dropna().median()
    else:
        tp_ratio_mean = pd.NA
        tp_ratio_median = pd.NA

    corr_txt = "-" if corr_gt_tp is None or pd.isna(corr_gt_tp) else f"{float(corr_gt_tp):.3f}"

    block = []
    block.append("<div style='border:1px solid #ddd; background:#fafafa; padding:12px; margin:10px 0 18px 0; font-family:Arial;'>")
    block.append("<div style='font-size:14px; font-weight:700; margin-bottom:6px;'>Day summary</div>")
    block.append("<div style='font-size:12px; line-height:1.6;'>")
    block.append(f"Images: <b>{n}</b><br>")
    if "gt_count_img" in df_day.columns:
        block.append(f"GT total / avg per img: <b>{int(gt_total)}</b> / <b>{fnum(avg_gt)}</b><br>")
    if "tp_img" in df_day.columns:
        block.append(f"TP total / avg per img: <b>{int(tp_total)}</b> / <b>{fnum(avg_tp)}</b><br>")
    if "fp_img" in df_day.columns:
        block.append(f"FP total / avg per img: <b>{int(fp_total)}</b> / <b>{fnum(avg_fp)}</b><br>")
    if "fn_img" in df_day.columns:
        block.append(f"FN total / avg per img: <b>{int(fn_total)}</b> / <b>{fnum(avg_fn)}</b><br>")

    if "gt_count_img" in df_day.columns and "tp_img" in df_day.columns:
        block.append(f"corr(GT, TP) across images: <b>{corr_txt}</b><br>")
    if "tp_ratio" in df_day.columns:
        block.append(f"TP/GT (gt>0) mean / median: <b>{fnum(tp_ratio_mean)}</b> / <b>{fnum(tp_ratio_median)}</b><br>")

    block.append("</div></div>")
    return "\n".join(block)