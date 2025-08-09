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
            model = getattr(r, "model_label", None) or getattr(r, "model", None)
            lang = getattr(r, "language", None)
            vocab_size = getattr(r, "vocab_size", None)
            success = getattr(r, "success", False)
            validation = getattr(r, "validation", {}) or {}
            pass_flag = bool(validation.get("pass") or (validation.get("only_vocab") and validation.get("all_targets_present")))
            coverage = validation.get("vocabulary_coverage")
            words = validation.get("total_words")
            exec_time = getattr(r, "execution_time", None)
            rows.append({
                "model": model,
                "language": lang,
                "vocab_size": vocab_size,
                "pass": pass_flag if success else False,
                "vocabulary_coverage": coverage,
                "total_words": words,
                "execution_time": exec_time,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def create_dashboard_html(results: List[Any], output_file: Path) -> Path:
    df = _results_to_dataframe(results)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build simple figures guarded against empty data
    figs = {}
    if not df.empty:
        # Pass rate by model and language
        agg = (
            df.assign(pass_int=df["pass"].astype(int))
              .groupby(["model", "language"], as_index=False)["pass_int"].mean()
              .rename(columns={"pass_int": "pass_rate"})
        )
        figs["pass"] = px.bar(agg, x="model", y="pass_rate", color="language", range_y=[0, 1], title="Pass rate by model and language")

        # Language averages
        lang_agg = agg.groupby("language", as_index=False)["pass_rate"].mean()
        figs["lang"] = px.bar(lang_agg, x="language", y="pass_rate", range_y=[0, 1], title="Average pass rate by language")

        # Execution time by model
        time_df = df.dropna(subset=["execution_time"]) if "execution_time" in df.columns else pd.DataFrame()
        if not time_df.empty:
            figs["time"] = px.box(time_df, x="model", y="execution_time", title="Execution time by model (s)")

    # Serialize figures
    fig_json = {k: v.to_json() for k, v in figs.items()}

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Vocabulary Benchmark Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f7fafc; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    h1 {{ margin: 0 0 12px; }}
  </style>
  <script>
    const FIGS = {json.dumps(fig_json)};
    function plotFromJson(id, key) {{
      if (!FIGS[key]) return;
      const obj = JSON.parse(FIGS[key]);
      Plotly.newPlot(id, obj.data, obj.layout, {{responsive: true}});
    }}
    window.addEventListener('DOMContentLoaded', () => {{
      plotFromJson('chart-pass', 'pass');
      plotFromJson('chart-lang', 'lang');
      plotFromJson('chart-time', 'time');
    }});
  </script>
    </head>
    <body>
        <div class="container">
      <div class="card"><h1>🎯 Vocabulary Inclusion Benchmark</h1></div>
      <div class="card"><div id="chart-pass"></div></div>
      <div class="card"><div id="chart-lang"></div></div>
      <div class="card"><div id="chart-time"></div></div>
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
        rates = (g["pass"].astype(int) * 100).tolist()
        models_data[model] = rates
    
    if models_data:
        panel_content = "[bold]Average Pass Rates by Model:[/bold]\n\n"
        max_width = 40
        for model, rates in models_data.items():
            avg_rate = sum(rates) / len(rates)
            bar_width = int((avg_rate / 100) * max_width)
            bar = "█" * bar_width + "░" * (max_width - bar_width)
            color = "green" if avg_rate >= 80 else "yellow" if avg_rate >= 50 else "red"
            panel_content += f"{model:20s} [{color}]{bar}[/{color}] {avg_rate:.1f}%\n"
        console.print(Panel(panel_content, title="Performance Overview", border_style="blue"))
    
    # Language performance summary
    from rich.table import Table
    table = Table(title="Language Performance Summary", show_header=True, header_style="bold magenta")
    table.add_column("Language", style="cyan", no_wrap=True)
    table.add_column("Avg Pass Rate", justify="right", style="green")

    for lang, g in df.groupby("language"):
        avg_pass = (g["pass"].astype(int).mean() * 100.0)
        pass_color = "green" if avg_pass >= 80 else "yellow" if avg_pass >= 50 else "red"
        table.add_row(lang or "", f"[{pass_color}]{avg_pass:.1f}%[/{pass_color}]")
    
    console.print("\n")
    console.print(table)
    console.print("\n")