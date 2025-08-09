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
            else:
                model = getattr(r, "model_label", None) or getattr(r, "model", None)
                lang = getattr(r, "language", None)
                vocab_size = getattr(r, "vocab_size", None)
                success = getattr(r, "success", False)
                validation = getattr(r, "validation", {}) or {}
                exec_time = getattr(r, "execution_time", None)
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

        # Compliance vs Vocab Size (avg percent_in_vocab), by model
        comp_df = (
            df_k.dropna(subset=["percent_in_vocab"]) if "percent_in_vocab" in df_k else pd.DataFrame()
        )
        if not comp_df.empty:
            comp_agg = (
                comp_df.groupby(["model", "vocab_size"], as_index=False)["percent_in_vocab"].mean()
            )
            figs["compliance_vs_vocab"] = px.line(
                comp_agg, x="vocab_size", y="percent_in_vocab", color="model",
                markers=True, title="Compliance (percent_in_vocab) vs Vocab Size"
            )
            figs["compliance_vs_vocab"].update_yaxes(range=[0, 1])

        # Target Inclusion vs Vocab Size (avg targets_present_ratio), by model
        targ_df = (
            df_k.dropna(subset=["targets_present_ratio"]) if "targets_present_ratio" in df_k else pd.DataFrame()
        )
        if not targ_df.empty:
            targ_agg = (
                targ_df.groupby(["model", "vocab_size"], as_index=False)["targets_present_ratio"].mean()
            )
            figs["targets_vs_vocab"] = px.line(
                targ_agg, x="vocab_size", y="targets_present_ratio", color="model",
                markers=True, title="Target Inclusion vs Vocab Size"
            )
            figs["targets_vs_vocab"].update_yaxes(range=[0, 1])

        # Pass Rate vs Vocab Size, by model
        pass_agg = (
            df_k.assign(pass_int=df_k["pass"].astype(int))
                .groupby(["model", "vocab_size"], as_index=False)["pass_int"].mean()
                .rename(columns={"pass_int": "pass_rate"})
        )
        if not pass_agg.empty:
            figs["pass_vs_vocab"] = px.bar(
                pass_agg, x="vocab_size", y="pass_rate", color="model", barmode="group",
                title="Pass Rate vs Vocab Size"
            )
            figs["pass_vs_vocab"].update_yaxes(range=[0, 1])

        # Execution Time distribution by model
        time_df = df_k.dropna(subset=["execution_time"]) if "execution_time" in df_k else pd.DataFrame()
        if not time_df.empty:
            figs["time_by_model"] = px.box(
                time_df, x="model", y="execution_time", points=False, title="Execution Time by Model (s)"
            )

        # Heatmap: avg compliance by model x language
        comp_ml = (
            df_k.dropna(subset=["percent_in_vocab"]) if "percent_in_vocab" in df_k else pd.DataFrame()
        )
        if not comp_ml.empty:
            heat_df = comp_ml.groupby(["model", "language"], as_index=False)["percent_in_vocab"].mean()
            pivot = heat_df.pivot(index="language", columns="model", values="percent_in_vocab").fillna(0)
            figs["heatmap_compliance"] = px.imshow(
                pivot, aspect="auto", color_continuous_scale="Blues", origin="lower",
                title="Compliance Heatmap (percent_in_vocab) — Language × Model"
            )
            figs["heatmap_compliance"].update_coloraxes(cmin=0, cmax=1)

        # Scatter: percent_in_vocab vs targets_present_ratio, size by vocab_size
        if not comp_df.empty and not targ_df.empty:
            scat_df = df_k.dropna(subset=["percent_in_vocab", "targets_present_ratio"]).copy()
            scat_df["size"] = scat_df["vocab_size"].astype(float)
            figs["scatter_quality"] = px.scatter(
                scat_df, x="percent_in_vocab", y="targets_present_ratio", color="model",
                size="size", hover_data=["language", "vocab_size"],
                title="Compliance vs Target Inclusion (size = vocab size)"
            )
            figs["scatter_quality"].update_xaxes(range=[0, 1])
            figs["scatter_quality"].update_yaxes(range=[0, 1])

    # Serialize figures
    fig_json = {k: v.to_json() for k, v in figs.items()}

    # Minimal aesthetic matching pramit.gg
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Vocabulary Benchmark Results</title>
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
      <style>
        :root {{
          --bg: #ffffff;
          --muted: #f5f5f7;
          --text: #0f172a;
          --subtle: #64748b;
          --card: #ffffff;
          --shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
          --radius: 14px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0; background: var(--bg);
          color: var(--text);
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
          line-height: 1.55;
        }}
        .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 20px 40px; }}
        .title {{ font-size: 28px; font-weight: 700; margin: 0 0 8px; letter-spacing: -0.02em; }}
        .subtitle {{ color: var(--subtle); margin-bottom: 20px; }}
        .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(12, 1fr); }}
        .card {{ background: var(--card); border-radius: var(--radius); padding: 16px; box-shadow: var(--shadow); }}
        .span-12 {{ grid-column: span 12; }}
        .span-6 {{ grid-column: span 6; }}
        .span-4 {{ grid-column: span 4; }}
        .kpis {{ display: grid; gap: 12px; grid-template-columns: repeat(4, 1fr); }}
        .kpi .label {{ color: var(--subtle); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
        .kpi .value {{ font-size: 24px; font-weight: 700; }}
        @media (max-width: 980px) {{
          .span-6, .span-4 {{ grid-column: span 12; }}
          .kpis {{ grid-template-columns: repeat(2, 1fr); }}
        }}
      </style>
      <script>
        const FIGS = {json.dumps(fig_json)};
        const KPIS = {json.dumps(kpis)};
        function plotFromJson(id, key) {{
          const json = FIGS[key]; if (!json) return;
          const obj = JSON.parse(json);
          Plotly.newPlot(id, obj.data, obj.layout, {{displayModeBar: false, responsive: true}});
        }}
        function fmtPct(x) {{ return (x*100).toFixed(1) + '%'; }}
        function fmtNum(x) {{ return new Intl.NumberFormat().format(x); }}
        window.addEventListener('DOMContentLoaded', () => {{
          // KPIs
          document.getElementById('kpi-total').textContent = fmtNum(KPIS.total_trials);
          document.getElementById('kpi-pass').textContent = fmtPct(KPIS.pass_rate || 0);
          document.getElementById('kpi-comp').textContent = fmtPct(KPIS.avg_percent_in_vocab || 0);
          document.getElementById('kpi-targets').textContent = fmtPct(KPIS.avg_targets_ratio || 0);
          document.getElementById('kpi-time').textContent = (KPIS.median_exec_time || 0).toFixed(1) + ' s';

          // Charts
          plotFromJson('chart-compliance-vocab', 'compliance_vs_vocab');
          plotFromJson('chart-targets-vocab', 'targets_vs_vocab');
          plotFromJson('chart-pass-vocab', 'pass_vs_vocab');
          plotFromJson('chart-time-model', 'time_by_model');
          plotFromJson('chart-heatmap-comp', 'heatmap_compliance');
          plotFromJson('chart-scatter', 'scatter_quality');
        }});
      </script>
    </head>
    <body>
      <div class="wrap">
        <div class="card span-12" style="margin-bottom: 16px;">
          <div class="title">Vocabulary Inclusion Benchmark</div>
          <div class="subtitle">Compliance, target inclusion, and performance across models, languages, and vocabulary sizes</div>
          <div class="kpis">
            <div class="kpi"><div class="label">Trials</div><div class="value" id="kpi-total">–</div></div>
            <div class="kpi"><div class="label">Pass rate</div><div class="value" id="kpi-pass">–</div></div>
            <div class="kpi"><div class="label">Avg compliance</div><div class="value" id="kpi-comp">–</div></div>
            <div class="kpi"><div class="label">Target inclusion</div><div class="value" id="kpi-targets">–</div></div>
            <div class="kpi"><div class="label">Median time</div><div class="value" id="kpi-time">–</div></div>
          </div>
        </div>

        <div class="grid">
          <div class="card span-6"><div id="chart-compliance-vocab"></div></div>
          <div class="card span-6"><div id="chart-targets-vocab"></div></div>

          <div class="card span-12"><div id="chart-pass-vocab"></div></div>

          <div class="card span-6"><div id="chart-time-model"></div></div>
          <div class="card span-6"><div id="chart-heatmap-comp"></div></div>

          <div class="card span-12"><div id="chart-scatter"></div></div>
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