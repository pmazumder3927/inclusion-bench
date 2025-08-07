from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def save_dashboard(df: pd.DataFrame, out_html: str | Path) -> None:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    figs = []

    # Pass rate by model
    if not df.empty:
        fig_pass = px.bar(
            df.groupby("model", as_index=False)["pass_rate"].mean(),
            x="model",
            y="pass_rate",
            color="model",
            title="Pass rate by model (mean across runs)",
            range_y=[0, 1],
        )
        figs.append(fig_pass)

        # Pass rate by model and language (facet)
        if "language" in df.columns:
            fig_lang = px.bar(
                df,
                x="model",
                y="pass_rate",
                color="model",
                facet_col="language",
                facet_col_wrap=4,
                title="Pass rate by model per language",
                range_y=[0, 1],
            )
            figs.append(fig_lang)

        # OOV vs missing targets scatter
        fig_scatter = px.scatter(
            df,
            x="avg_oov_types",
            y="avg_missing_targets",
            color="model",
            size="trials",
            hover_data=["model", "language", "vocab_size", "trials"],
            title="OOV types vs Missing targets",
        )
        figs.append(fig_scatter)

        # Pass rate vs vocab size per model
        if "vocab_size" in df.columns:
            fig_size = px.line(
                df.sort_values(["model", "language", "vocab_size"]),
                x="vocab_size",
                y="pass_rate",
                color="model",
                line_group="language",
                markers=True,
                title="Pass rate vs Vocab Size",
            )
            figs.append(fig_size)

    # Compose into a single HTML with multiple figures
    html_parts = [f.to_html(full_html=False, include_plotlyjs='cdn') for f in figs]
    html = "\n".join(html_parts)
    out_html.write_text(html, encoding="utf-8")
