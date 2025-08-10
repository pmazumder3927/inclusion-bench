"""Beautiful visualizations for benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()


def _results_to_dataframe(results: List[Any]) -> pd.DataFrame:
    rows = []
    for r in results:
        try:
            if isinstance(r, dict):
                model = r.get("model_label") or r.get("model")
                lang = r.get("language")
                vocab_size = r.get("vocab_size")
                success = r.get("success", False)
                validation = r.get("validation", {}) or {}
                exec_time = r.get("execution_time")
                words_field = r.get("words")
            else:
                model = getattr(r, "model_label", None) or getattr(r, "model", None)
                lang = getattr(r, "language", None)
                vocab_size = getattr(r, "vocab_size", None)
                success = getattr(r, "success", False)
                validation = getattr(r, "validation", {}) or {}
                exec_time = getattr(r, "execution_time", None)
                words_field = getattr(r, "words", None)
            # Exclude entries with empty words arrays from visualization
            if isinstance(words_field, list) and len(words_field) == 0:
                continue
            pass_flag = bool(validation.get("pass") or (validation.get("only_vocab") and validation.get("all_targets_present")))
            coverage = validation.get("vocabulary_coverage")
            words = validation.get("total_words")
            percent_in_vocab = validation.get("percent_in_vocab")
            targets_present_ratio = validation.get("targets_present_ratio")
            rows.append({
                "model": model,
                "language": lang,
                "vocab_size": vocab_size,
                "pass": pass_flag if success else False,
                "vocabulary_coverage": coverage,
                "total_words": words,
                "percent_in_vocab": percent_in_vocab,
                "targets_present_ratio": targets_present_ratio,
                "execution_time": exec_time,
                "success": bool(success),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def create_dashboard_html(results: List[Any], output_file: Path) -> Path:
    df = _results_to_dataframe(results)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figs: Dict[str, Any] = {}
    kpis: Dict[str, Any] = {
        "total_trials": 0,
        "pass_rate": 0.0,
        "avg_percent_in_vocab": 0.0,
        "avg_targets_ratio": 0.0,
        "median_exec_time": 0.0,
    }

    # Define exact colors from pramit.gg
    colors = {
        "void-black": "#000000",
        "charcoal-black": "#0a0a0a", 
        "deep-graphite": "#1a1b22",
        "slate-gray": "#1c1c23",
        "accent-orange": "#ff6b3d",
        "accent-purple": "#7c77c6",
        "accent-blue": "#4a9eff",
        "accent-green": "#30d158",
        "accent-pink": "#ff375f",
        "accent-yellow": "#ffd60a",
        "text-primary": "#ffffff",
        "text-secondary": "#a1a1aa",
        "text-muted": "#71717a",
    }

    # Plotly layout for dark theme
    plotly_layout = {
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "font": {"family": "ui-sans-serif, system-ui, -apple-system, sans-serif", "color": colors["text-secondary"]},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "showlegend": True,
        "legend": {
            "bgcolor": "rgba(26, 27, 34, 0.6)",
            "bordercolor": "rgba(255, 255, 255, 0.05)",
            "borderwidth": 1,
            "font": {"color": colors["text-secondary"]}
        },
        "xaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.05)",
            "zerolinecolor": "rgba(255, 255, 255, 0.1)",
            "color": colors["text-muted"]
        },
        "yaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.05)",
            "zerolinecolor": "rgba(255, 255, 255, 0.1)",
            "color": colors["text-muted"]
        },
    }

    if not df.empty:
        # Basic KPI metrics
        df_k = df.copy()
        df_k["pass_int"] = df_k["pass"].astype(int)
        kpis["total_trials"] = int(len(df_k))
        kpis["pass_rate"] = float(df_k["pass_int"].mean()) if len(df_k) else 0.0
        if "percent_in_vocab" in df_k:
            kpis["avg_percent_in_vocab"] = float(df_k["percent_in_vocab"].fillna(0).mean())
        if "targets_present_ratio" in df_k:
            kpis["avg_targets_ratio"] = float(df_k["targets_present_ratio"].fillna(0).mean())
        if "execution_time" in df_k:
            kpis["median_exec_time"] = float(df_k["execution_time"].fillna(0).median())

        # CAPSTONE: Overall Vocabulary Inclusion Performance Bar Chart
        overall_df = df_k.copy()
        if "percent_in_vocab" in overall_df:
            model_performance = (
                overall_df.groupby("model", as_index=False)
                .agg({
                    "percent_in_vocab": "mean",
                    "targets_present_ratio": "mean",
                    "pass_int": "mean"
                })
                .rename(columns={"pass_int": "pass_rate"})
                .sort_values("percent_in_vocab", ascending=True)
            )
            
            # Create gradient using accent colors
            n_models = len(model_performance)
            # Use purple gradient for bars
            bar_colors = [f"rgba(124, 119, 198, {0.4 + (0.4 * i / max(n_models-1, 1))})" for i in range(n_models)]
            
            figs["overall_performance"] = go.Figure()
            
            # Add vocabulary compliance bars
            figs["overall_performance"].add_trace(go.Bar(
                y=model_performance["model"],
                x=model_performance["percent_in_vocab"],
                name="Vocabulary Compliance",
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(color=colors["accent-purple"], width=1)
                ),
                text=[f"{x:.1%}" for x in model_performance["percent_in_vocab"]],
                textposition="auto",
                textfont=dict(color=colors["text-primary"], size=11),
                hovertemplate="<b>%{y}</b><br>Compliance: %{x:.1%}<extra></extra>"
            ))
            
            # Add target inclusion as orange accent markers
            figs["overall_performance"].add_trace(go.Scatter(
                y=model_performance["model"],
                x=model_performance["targets_present_ratio"],
                name="Target Inclusion",
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors["accent-orange"],
                    symbol="diamond",
                    line=dict(width=0)
                ),
                hovertemplate="<b>%{y}</b><br>Targets: %{x:.1%}<extra></extra>"
            ))
            
            layout_update = plotly_layout.copy()
            layout_update.update({
                "title": dict(
                    text="<b>Vocabulary Inclusion Performance</b>",
                    font=dict(size=18, color=colors["text-primary"]),
                    x=0.5,
                    xanchor="center"
                ),
                "xaxis": dict(
                    title="Performance Score",
                    range=[0, 1],
                    tickformat=".0%",
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    showgrid=True,
                    color=colors["text-muted"]
                ),
                "yaxis": dict(
                    title="",
                    gridcolor="rgba(255, 255, 255, 0.02)",
                    showgrid=False,
                    color=colors["text-secondary"]
                ),
                "height": 400,
                "bargap": 0.2
            })
            figs["overall_performance"].update_layout(layout_update)

        # Compliance vs Vocab Size 
        comp_df = (
            df_k.dropna(subset=["percent_in_vocab"]) if "percent_in_vocab" in df_k else pd.DataFrame()
        )
        if not comp_df.empty:
            comp_agg = (
                comp_df.groupby(["model", "vocab_size"], as_index=False)["percent_in_vocab"].mean()
            )
            
            fig_comp = go.Figure()
            # Use a variety of accent colors
            line_colors = [
                colors["accent-purple"],
                colors["accent-orange"],
                colors["accent-blue"],
                colors["accent-green"],
                colors["accent-pink"],
                colors["accent-yellow"]
            ]
            
            for i, model in enumerate(comp_agg["model"].unique()):
                model_data = comp_agg[comp_agg["model"] == model].sort_values("vocab_size")
                line_color = line_colors[i % len(line_colors)]
                
                fig_comp.add_trace(go.Scatter(
                    x=model_data["vocab_size"],
                    y=model_data["percent_in_vocab"],
                    name=model,
                    mode="lines+markers",
                    line=dict(color=line_color, width=2),
                    marker=dict(size=6, color=line_color),
                    hovertemplate="<b>%{fullData.name}</b><br>Vocab: %{x}<br>Compliance: %{y:.1%}<extra></extra>"
                ))
            
            layout_update = plotly_layout.copy()
            layout_update.update({
                "title": dict(text="Vocabulary Compliance Across Sizes", font=dict(color=colors["text-primary"], size=16)),
                "xaxis_title": "Vocabulary Size",
                "yaxis_title": "Compliance Rate",
                "yaxis": dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255, 255, 255, 0.05)"),
                "hovermode": "x unified"
            })
            fig_comp.update_layout(layout_update)
            figs["compliance_vs_vocab"] = fig_comp

        # Target Inclusion Heatmap
        targ_df = df_k.dropna(subset=["targets_present_ratio"]) if "targets_present_ratio" in df_k else pd.DataFrame()
        if not targ_df.empty:
            heat_data = targ_df.groupby(["model", "language"], as_index=False)["targets_present_ratio"].mean()
            pivot = heat_data.pivot(index="model", columns="language", values="targets_present_ratio").fillna(0)
            
            # Custom colorscale from dark to orange
            figs["targets_heatmap"] = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=[
                    [0, colors["deep-graphite"]],
                    [0.5, colors["accent-purple"]],
                    [1, colors["accent-orange"]]
                ],
                text=[[f"{val:.1%}" for val in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 10, "color": colors["text-primary"]},
                hovertemplate="Model: %{y}<br>Language: %{x}<br>Target Rate: %{z:.1%}<extra></extra>",
                colorbar=dict(
                    title=dict(text="Target Rate", font=dict(color=colors["text-secondary"])),
                    tickformat=".0%",
                    thickness=12,
                    len=0.6,
                    tickfont=dict(color=colors["text-muted"])
                )
            ))
            
            layout_update = plotly_layout.copy()
            layout_update.update({
                "title": dict(text="Target Inclusion by Model and Language", font=dict(color=colors["text-primary"], size=16)),
                "xaxis_title": "Language",
                "yaxis_title": "Model"
            })
            figs["targets_heatmap"].update_layout(layout_update)

        # Performance Distribution (Violin plots)
        if not comp_df.empty:
            perf_models = comp_df["model"].unique()[:8]
            
            fig_violin = go.Figure()
            # Use alternating accent colors
            violin_colors = [colors["accent-purple"], colors["accent-blue"]]
            
            for i, model in enumerate(perf_models):
                model_data = comp_df[comp_df["model"] == model]["percent_in_vocab"]
                violin_color = violin_colors[i % len(violin_colors)]
                
                fig_violin.add_trace(go.Violin(
                    y=model_data,
                    name=model,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=violin_color,
                    opacity=0.3,
                    line_color=violin_color,
                    hovertemplate="<b>%{fullData.name}</b><br>Compliance: %{y:.1%}<extra></extra>"
                ))
            
            layout_update = plotly_layout.copy()
            layout_update.update({
                "title": dict(text="Compliance Distribution by Model", font=dict(color=colors["text-primary"], size=16)),
                "yaxis_title": "Compliance Rate",
                "yaxis": dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255, 255, 255, 0.05)"),
                "showlegend": False
            })
            fig_violin.update_layout(layout_update)
            figs["compliance_distribution"] = fig_violin

        # Execution Time vs Performance Scatter
        time_perf_df = df_k.dropna(subset=["execution_time", "percent_in_vocab"])
        if not time_perf_df.empty:
            scatter_agg = (
                time_perf_df.groupby("model", as_index=False)
                .agg({
                    "execution_time": "median",
                    "percent_in_vocab": "mean",
                    "pass_int": "mean"
                })
            )
            
            figs["time_vs_performance"] = go.Figure()
            
            # Use green to pink gradient for performance
            figs["time_vs_performance"].add_trace(go.Scatter(
                x=scatter_agg["execution_time"],
                y=scatter_agg["percent_in_vocab"],
                mode="markers+text",
                text=scatter_agg["model"],
                textposition="top center",
                textfont=dict(size=9, color=colors["text-muted"]),
                marker=dict(
                    size=scatter_agg["pass_int"] * 40 + 8,
                    color=scatter_agg["percent_in_vocab"],
                    colorscale=[
                        [0, colors["accent-pink"]],
                        [0.5, colors["accent-yellow"]],
                        [1, colors["accent-green"]]
                    ],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Compliance", font=dict(color=colors["text-secondary"])),
                        tickformat=".0%",
                        thickness=12,
                        tickfont=dict(color=colors["text-muted"])
                    ),
                    line=dict(width=1, color="rgba(255, 255, 255, 0.2)")
                ),
                hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}s<br>Compliance: %{y:.1%}<extra></extra>"
            ))
            
            layout_update = plotly_layout.copy()
            layout_update.update({
                "title": dict(text="Efficiency vs Performance", font=dict(color=colors["text-primary"], size=16)),
                "xaxis_title": "Median Execution Time (s)",
                "yaxis_title": "Compliance Rate",
                "yaxis": dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255, 255, 255, 0.05)")
            })
            figs["time_vs_performance"].update_layout(layout_update)

        # Language Comparison Radar
        lang_df = df_k.groupby("language", as_index=False).agg({
            "percent_in_vocab": "mean",
            "targets_present_ratio": "mean",
            "pass_int": "mean",
            "execution_time": "mean"
        })
        
        if len(lang_df) > 2:
            metrics = ["percent_in_vocab", "targets_present_ratio", "pass_int"]
            
            fig_radar = go.Figure()
            radar_colors = [colors["accent-purple"], colors["accent-orange"], colors["accent-blue"]]
            
            for idx, row in lang_df.iterrows():
                values = [row[m] for m in metrics]
                values.append(values[0])
                radar_color = radar_colors[idx % len(radar_colors)]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=["Compliance", "Targets", "Pass Rate", "Compliance"],
                    fill="toself",
                    fillcolor=radar_color,
                    opacity=0.2,
                    line=dict(color=radar_color, width=2),
                    name=row["language"],
                    hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.1%}<extra></extra>"
                ))
            
            layout_update = plotly_layout.copy()
            layout_update.update({
                "title": dict(text="Language Performance Comparison", font=dict(color=colors["text-primary"], size=16)),
                "polar": dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickformat=".0%",
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        tickfont=dict(color=colors["text-muted"])
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        tickfont=dict(color=colors["text-secondary"])
                    ),
                    bgcolor="rgba(0, 0, 0, 0)"
                )
            })
            fig_radar.update_layout(layout_update)
            figs["language_radar"] = fig_radar

    # Serialize figures
    fig_json = {k: v.to_json() for k, v in figs.items()}

    # HTML with refined dark glassmorphism aesthetic
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Vocabulary Inclusion Benchmark</title>
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
      <style>
        :root {{
          --void-black: #000000;
          --charcoal-black: #0a0a0a;
          --deep-graphite: #1a1b22;
          --slate-gray: #1c1c23;
          --accent-orange: #ff6b3d;
          --accent-purple: #7c77c6;
          --accent-blue: #4a9eff;
          --accent-green: #30d158;
          --accent-pink: #ff375f;
          --accent-yellow: #ffd60a;
          --text-primary: #ffffff;
          --text-secondary: #a1a1aa;
          --text-muted: #71717a;
          --border: rgba(255, 255, 255, 0.06);
          --glass-bg: rgba(26, 27, 34, 0.5);
          --glass-border: rgba(255, 255, 255, 0.05);
        }}
        
        * {{
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }}
        
        body {{
          background: linear-gradient(to bottom right, var(--void-black), var(--charcoal-black), var(--void-black));
          color: var(--text-primary);
          font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
          line-height: 1.6;
          min-height: 100vh;
          position: relative;
        }}
        
        body::before {{
          content: '';
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: 
            radial-gradient(circle at 20% 80%, rgba(124, 119, 198, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 107, 61, 0.02) 0%, transparent 50%);
          pointer-events: none;
          z-index: 1;
        }}
        
        .container {{
          max-width: 1400px;
          margin: 0 auto;
          padding: 3rem 2rem;
          position: relative;
          z-index: 2;
        }}
        
        .header {{
          text-align: center;
          margin-bottom: 4rem;
        }}
        
        h1 {{
          font-size: 2.5rem;
          font-weight: 700;
          color: var(--text-primary);
          margin-bottom: 0.5rem;
          letter-spacing: -0.025em;
        }}
        
        .accent-text {{
          background: linear-gradient(135deg, var(--accent-orange), var(--accent-purple));
          -webkit-background-clip: text;
          background-clip: text;
          -webkit-text-fill-color: transparent;
          display: inline-block;
        }}
        
        .subtitle {{
          color: var(--text-muted);
          font-size: 1rem;
          font-weight: 400;
        }}
        
        .kpi-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1.5rem;
          margin-bottom: 3rem;
        }}
        
        .kpi-card {{
          background: var(--glass-bg);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid var(--glass-border);
          border-radius: 12px;
          padding: 1.5rem;
          text-align: center;
          transition: all 0.2s ease;
          position: relative;
          overflow: hidden;
        }}
        
        .kpi-card::before {{
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent, var(--accent-purple), transparent);
          opacity: 0;
          transition: opacity 0.3s ease;
        }}
        
        .kpi-card:hover {{
          transform: translateY(-2px);
          border-color: rgba(255, 255, 255, 0.1);
        }}
        
        .kpi-card:hover::before {{
          opacity: 1;
        }}
        
        .kpi-label {{
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--text-muted);
          margin-bottom: 0.5rem;
          font-weight: 500;
        }}
        
        .kpi-value {{
          font-size: 1.875rem;
          font-weight: 600;
          color: var(--text-primary);
        }}
        
        .kpi-card:nth-child(2) .kpi-value {{
          color: var(--accent-green);
        }}
        
        .kpi-card:nth-child(3) .kpi-value {{
          color: var(--accent-purple);
        }}
        
        .kpi-card:nth-child(4) .kpi-value {{
          color: var(--accent-orange);
        }}
        
        .chart-grid {{
          display: grid;
          gap: 2rem;
        }}
        
        .chart-card {{
          background: var(--glass-bg);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid var(--glass-border);
          border-radius: 12px;
          padding: 1.5rem;
          min-height: 400px;
        }}
        
        .chart-card.featured {{
          border-color: rgba(124, 119, 198, 0.2);
        }}
        
        .chart-title {{
          font-size: 1.125rem;
          font-weight: 600;
          color: var(--text-secondary);
          margin-bottom: 1rem;
        }}
        
        .two-col {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
        }}
        
        @media (max-width: 968px) {{
          h1 {{ font-size: 2rem; }}
          .two-col {{ grid-template-columns: 1fr; }}
          .kpi-value {{ font-size: 1.5rem; }}
        }}
        
        .footer {{
          text-align: center;
          padding: 3rem 2rem 2rem;
          color: var(--text-muted);
          font-size: 0.875rem;
          border-top: 1px solid var(--border);
          margin-top: 4rem;
        }}
      </style>
      <script>
        const FIGS = {json.dumps(fig_json)};
        const KPIS = {json.dumps(kpis)};
        
        function plotFromJson(id, key) {{
          const json = FIGS[key];
          if (!json) return;
          const obj = JSON.parse(json);
          
          // Ensure dark theme
          if (obj.layout) {{
            obj.layout.paper_bgcolor = 'rgba(0, 0, 0, 0)';
            obj.layout.plot_bgcolor = 'rgba(0, 0, 0, 0)';
          }}
          
          Plotly.newPlot(id, obj.data, obj.layout, {{
            displayModeBar: false,
            responsive: true
          }});
        }}
        
        function formatPercent(x) {{
          return (x * 100).toFixed(1) + '%';
        }}
        
        function formatNumber(x) {{
          return new Intl.NumberFormat().format(x);
        }}
        
        window.addEventListener('DOMContentLoaded', () => {{
          // Populate KPIs
          document.getElementById('kpi-trials').textContent = formatNumber(KPIS.total_trials);
          document.getElementById('kpi-pass').textContent = formatPercent(KPIS.pass_rate || 0);
          document.getElementById('kpi-compliance').textContent = formatPercent(KPIS.avg_percent_in_vocab || 0);
          document.getElementById('kpi-targets').textContent = formatPercent(KPIS.avg_targets_ratio || 0);
          document.getElementById('kpi-time').textContent = (KPIS.median_exec_time || 0).toFixed(1) + 's';
          
          // Render charts
          plotFromJson('chart-overall', 'overall_performance');
          plotFromJson('chart-compliance', 'compliance_vs_vocab');
          plotFromJson('chart-heatmap', 'targets_heatmap');
          plotFromJson('chart-distribution', 'compliance_distribution');
          plotFromJson('chart-efficiency', 'time_vs_performance');
          plotFromJson('chart-radar', 'language_radar');
        }});
      </script>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Vocabulary <span class="accent-text">Inclusion</span> Benchmark</h1>
          <p class="subtitle">Comprehensive analysis of model performance on vocabulary-constrained generation</p>
        </div>
        
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Total Trials</div>
            <div class="kpi-value" id="kpi-trials">-</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Pass Rate</div>
            <div class="kpi-value" id="kpi-pass">-</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Avg Compliance</div>
            <div class="kpi-value" id="kpi-compliance">-</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Target Inclusion</div>
            <div class="kpi-value" id="kpi-targets">-</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Median Time</div>
            <div class="kpi-value" id="kpi-time">-</div>
          </div>
        </div>
        
        <div class="chart-grid">
          <div class="chart-card featured">
            <div class="chart-title">Overall Model Performance</div>
            <div id="chart-overall" style="height: 400px;"></div>
          </div>
          
          <div class="two-col">
            <div class="chart-card">
              <div class="chart-title">Compliance Trends</div>
              <div id="chart-compliance" style="height: 350px;"></div>
            </div>
            
            <div class="chart-card">
              <div class="chart-title">Target Achievement Heatmap</div>
              <div id="chart-heatmap" style="height: 350px;"></div>
            </div>
          </div>
          
          <div class="two-col">
            <div class="chart-card">
              <div class="chart-title">Performance Distribution</div>
              <div id="chart-distribution" style="height: 350px;"></div>
            </div>
            
            <div class="chart-card">
              <div class="chart-title">Efficiency Analysis</div>
              <div id="chart-efficiency" style="height: 350px;"></div>
            </div>
          </div>
          
          <div class="chart-card">
            <div class="chart-title">Language Comparison</div>
            <div id="chart-radar" style="height: 400px;"></div>
          </div>
        </div>
        
        <div class="footer">
          <p>Vocabulary Inclusion Benchmark • Generated with inclusion-bench</p>
        </div>
      </div>
    </body>
    </html>
    """
    output_file.write_text(html, encoding="utf-8")
    return output_file


def display_terminal_charts(results: List[Any]):
    console.print("\n[bold cyan]📊 Benchmark Results Summary[/bold cyan]\n")
    
    df = _results_to_dataframe(results)
    if df.empty:
        return

    models_data = {}
    for model, g in df.groupby("model"):
        compliance_rates = (g["percent_in_vocab"].fillna(0) * 100).tolist() if "percent_in_vocab" in g else []
        if compliance_rates:
            models_data[model] = compliance_rates
    
    if models_data:
        panel_content = "[bold]Vocabulary Compliance by Model:[/bold]\n\n"
        max_width = 40
        
        # Sort models by average compliance
        sorted_models = sorted(models_data.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0, reverse=True)
        
        for model, rates in sorted_models:
            avg_rate = sum(rates) / len(rates) if rates else 0
            bar_width = int((avg_rate / 100) * max_width)
            
            # Create gradient bar effect
            if avg_rate >= 90:
                color = "green"
                bar_char = "█"
            elif avg_rate >= 70:
                color = "yellow" 
                bar_char = "▓"
            elif avg_rate >= 50:
                color = "yellow"
                bar_char = "▒"
            else:
                color = "red"
                bar_char = "░"
            
            bar = bar_char * bar_width + "░" * (max_width - bar_width)
            panel_content += f"{model:20s} [{color}]{bar}[/{color}] {avg_rate:.1f}%\n"
            
        console.print(Panel(
            panel_content,
            title="[bold magenta]Performance Overview[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        ))
    
    # Enhanced Language Performance Table
    from rich.table import Table
    table = Table(
        title="[bold cyan]Language Performance Matrix[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        show_lines=True
    )
    
    table.add_column("Language", style="white", no_wrap=True)
    table.add_column("Compliance", justify="center")
    table.add_column("Targets", justify="center")
    table.add_column("Pass Rate", justify="center")
    table.add_column("Trials", justify="center", style="dim")

    for lang, g in df.groupby("language"):
        compliance = g["percent_in_vocab"].mean() * 100 if "percent_in_vocab" in g else 0
        targets = g["targets_present_ratio"].mean() * 100 if "targets_present_ratio" in g else 0
        pass_rate = g["pass"].astype(int).mean() * 100
        trials = len(g)
        
        # Color coding
        comp_color = "green" if compliance >= 90 else "yellow" if compliance >= 70 else "red"
        targ_color = "green" if targets >= 90 else "yellow" if targets >= 70 else "red"
        pass_color = "green" if pass_rate >= 80 else "yellow" if pass_rate >= 50 else "red"
        
        table.add_row(
            lang or "unknown",
            f"[{comp_color}]{compliance:.1f}%[/{comp_color}]",
            f"[{targ_color}]{targets:.1f}%[/{targ_color}]",
            f"[{pass_color}]{pass_rate:.1f}%[/{pass_color}]",
            str(trials)
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n")


def create_dashboard_from_jsonl(details_path: Path | str, output_file: Path | str) -> Path:
    """Load details.jsonl and generate the dashboard HTML."""
    details_path = Path(details_path)
    rows: List[Dict[str, Any]] = []
    with details_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return create_dashboard_html(rows, Path(output_file))