# plotting_strategies.py
from typing import List, Tuple, Protocol

import plotly.graph_objects as go
import pandas as pd


class PlottingStrategy(Protocol):
    """Interface for image-level day plotting strategies."""

    def generate_plots(
        self,
        df_day: pd.DataFrame,
        x: pd.Series,
        x_title: str,
        hover_cols: List[str],
        title: str,
    ) -> Tuple[str, str]:
        """Generate two plot HTML snippets (or empty string for the second if not needed)."""
        ...


class SchemeAGtTpStrategy(PlottingStrategy):
    def generate_plots(
        self,
        df_day: pd.DataFrame,
        x: pd.Series,
        x_title: str,
        hover_cols: List[str],
        title: str,
    ) -> Tuple[str, str]:
        # Chart 1: GT vs TP (+ TP/GT on y2)
        fig1 = go.Figure()

        if "gt_count_img" in df_day.columns:
            fig1.add_trace(
                go.Scatter(
                    x=x,
                    y=df_day["gt_count_img"],
                    mode="lines+markers",
                    name="GT count (per image)",
                    customdata=df_day[["url"] + hover_cols].values,
                    hovertemplate=self._build_hovertemplate("GT count", hover_cols),
                )
            )
        if "tp_img" in df_day.columns:
            fig1.add_trace(
                go.Scatter(
                    x=x,
                    y=df_day["tp_img"],
                    mode="lines+markers",
                    name="TP (per image)",
                    customdata=df_day[["url"] + hover_cols].values,
                    hovertemplate=self._build_hovertemplate("TP", hover_cols),
                )
            )
        # TP/GT ratio on y2
        if "tp_ratio" in df_day.columns and df_day["tp_ratio"].notna().any():
            fig1.add_trace(
                go.Scatter(
                    x=x,
                    y=df_day["tp_ratio"],
                    mode="lines+markers",
                    name="TP/GT (gt>0)",
                    yaxis="y2",
                    customdata=df_day[["url"] + hover_cols].values,
                    hovertemplate=self._build_hovertemplate("TP/GT", hover_cols),
                )
            )

        fig1.update_layout(
            title=f"{title} | A1) GT vs TP (+ TP/GT on right axis) (click points)",
            xaxis=dict(title=x_title, rangeslider=dict(visible=True)),
            yaxis=dict(title="Count"),
            yaxis2=dict(title="TP/GT", overlaying="y", side="right"),
            template="plotly_white",
            margin=dict(l=70, r=70, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Chart 2: FP / FN / (FP+FN)
        fig2 = go.Figure()
        for col, nm, ylab in [
            ("fp_img", "FP (per image)", "FP"),
            ("fn_img", "FN (per image)", "FN"),
            ("err_obj", "FP+FN (per image)", "FP+FN"),
        ]:
            if col in df_day.columns:
                fig2.add_trace(
                    go.Scatter(
                        x=x,
                        y=df_day[col],
                        mode="lines+markers",
                        name=nm,
                        customdata=df_day[["url"] + hover_cols].values,
                        hovertemplate=self._build_hovertemplate(ylab, hover_cols),
                    )
                )

        fig2.update_layout(
            title=f"{title} | A2) FP / FN / (FP+FN) (click points)",
            xaxis=dict(title=x_title, rangeslider=dict(visible=True)),
            yaxis=dict(title="Count"),
            template="plotly_white",
            margin=dict(l=70, r=70, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        html1 = fig1.to_html(include_plotlyjs="cdn", full_html=False)
        html2 = fig2.to_html(include_plotlyjs=False, full_html=False)
        return html1, html2

    def _build_hovertemplate(self, y_label: str, hover_cols: List[str]) -> str:
        lines = [f"<b>{y_label}</b>: %{{y}}<br>"]
        for i, c in enumerate(hover_cols, start=1):
            lines.append(f"{c}: %{{customdata[{i}]}}<br>")
        lines.append("<b>Click</b> to open FiftyOne<br>")
        return "".join(lines)


# Example alternative strategy: Plot all predictions
class AllPredictionsStrategy(PlottingStrategy):
    def generate_plots(
        self,
        df_day: pd.DataFrame,
        x: pd.Series,
        x_title: str,
        hover_cols: List[str],
        title: str,
    ) -> Tuple[str, str]:
        fig = go.Figure()

        if "pred_count_img" in df_day.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df_day["pred_count_img"],
                    mode="lines+markers",
                    name="Pred count (per image)",
                    customdata=df_day[["url"] + hover_cols].values,
                    hovertemplate=self._build_hovertemplate("Pred count", hover_cols),
                )
            )

        fig.update_layout(
            title=f"{title} | All Predictions",
            xaxis=dict(title=x_title, rangeslider=dict(visible=True)),
            yaxis=dict(title="Count"),
            template="plotly_white",
            margin=dict(l=70, r=70, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        html1 = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return html1, ""  # Only one plot

    def _build_hovertemplate(self, y_label: str, hover_cols: List[str]) -> str:
        lines = [f"<b>{y_label}</b>: %{{y}}<br>"]
        for i, c in enumerate(hover_cols, start=1):
            lines.append(f"{c}: %{{customdata[{i}]}}<br>")
        lines.append("<b>Click</b> to open FiftyOne<br>")
        return "".join(lines)


# Example: TP only strategy
class TpOnlyStrategy(PlottingStrategy):
    def generate_plots(
        self,
        df_day: pd.DataFrame,
        x: pd.Series,
        x_title: str,
        hover_cols: List[str],
        title: str,
    ) -> Tuple[str, str]:
        fig = go.Figure()

        if "tp_img" in df_day.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df_day["tp_img"],
                    mode="lines+markers",
                    name="TP (per image)",
                    customdata=df_day[["url"] + hover_cols].values,
                    hovertemplate=self._build_hovertemplate("TP", hover_cols),
                )
            )

        fig.update_layout(
            title=f"{title} | TP Only",
            xaxis=dict(title=x_title, rangeslider=dict(visible=True)),
            yaxis=dict(title="Count"),
            template="plotly_white",
            margin=dict(l=70, r=70, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        html1 = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return html1, ""  # Only one plot

    def _build_hovertemplate(self, y_label: str, hover_cols: List[str]) -> str:
        lines = [f"<b>{y_label}</b>: %{{y}}<br>"]
        for i, c in enumerate(hover_cols, start=1):
            lines.append(f"{c}: %{{customdata[{i}]}}<br>")
        lines.append("<b>Click</b> to open FiftyOne<br>")
        return "".join(lines)