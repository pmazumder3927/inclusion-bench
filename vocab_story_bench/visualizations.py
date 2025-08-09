"""Beautiful visualizations for benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()


def create_pass_rate_chart(summary: Dict[str, Any]) -> go.Figure:
    """Create a bar chart showing pass rates by model and language."""
    models = []
    languages = []
    pass_rates = []
    
    for key, stats in summary.items():
        if isinstance(stats, dict) and 'model' in stats:
            models.append(stats['model'])
            languages.append(stats['language'])
            pass_rates.append(stats['pass_rate'] * 100)
    
    df = pd.DataFrame({
        'Model': models,
        'Language': languages,
        'Pass Rate (%)': pass_rates
    })
    
    # Create grouped bar chart
    fig = px.bar(
        df,
        x='Model',
        y='Pass Rate (%)',
        color='Language',
        title='Pass Rate by Model and Language',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=500
    )
    
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Pass Rate (%)',
        yaxis_range=[0, 105],
        showlegend=True,
        font=dict(size=12),
        hovermode='x unified'
    )
    
    # Add threshold line at 80%
    fig.add_hline(y=80, line_dash="dash", line_color="green", 
                  annotation_text="Good (80%)", annotation_position="right")
    
    return fig


def create_vocabulary_coverage_chart(summary: Dict[str, Any]) -> go.Figure:
    """Create a heatmap showing vocabulary coverage."""
    models = []
    languages = []
    coverages = []
    vocab_sizes = []
    
    for key, stats in summary.items():
        if isinstance(stats, dict) and 'model' in stats:
            models.append(stats['model'])
            languages.append(stats['language'])
            coverages.append(stats.get('avg_vocabulary_coverage', 0) * 100)
            vocab_sizes.append(stats.get('vocab_size', 0))
    
    # Get unique models and languages
    unique_models = sorted(list(set(models)))
    unique_languages = sorted(list(set(languages)))
    
    # Create matrix for heatmap
    coverage_matrix = []
    for lang in unique_languages:
        row = []
        for model in unique_models:
            # Find coverage for this model-language pair
            coverage = 0
            for i, (m, l, c) in enumerate(zip(models, languages, coverages)):
                if m == model and l == lang:
                    coverage = c
                    break
            row.append(coverage)
        coverage_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=coverage_matrix,
        x=unique_models,
        y=unique_languages,
        colorscale='RdYlGn',
        text=[[f'{val:.1f}%' for val in row] for row in coverage_matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Coverage (%)")
    ))
    
    fig.update_layout(
        title='Vocabulary Coverage Heatmap',
        xaxis_title='Model',
        yaxis_title='Language',
        height=400 + len(unique_languages) * 30
    )
    
    return fig


def create_performance_comparison(summary: Dict[str, Any]) -> go.Figure:
    """Create a radar chart comparing model performance across metrics."""
    models_data = {}
    
    for key, stats in summary.items():
        if isinstance(stats, dict) and 'model' in stats:
            model = stats['model']
            if model not in models_data:
                models_data[model] = {
                    'pass_rates': [],
                    'coverages': [],
                    'word_counts': [],
                    'languages': []
                }
            
            models_data[model]['pass_rates'].append(stats.get('pass_rate', 0))
            models_data[model]['coverages'].append(stats.get('avg_vocabulary_coverage', 0))
            models_data[model]['word_counts'].append(stats.get('avg_word_count', 0))
            models_data[model]['languages'].append(stats.get('language', ''))
    
    # Calculate averages for each model
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    for i, (model, data) in enumerate(models_data.items()):
        avg_pass_rate = sum(data['pass_rates']) / len(data['pass_rates']) * 100
        avg_coverage = sum(data['coverages']) / len(data['coverages']) * 100
        avg_word_count = sum(data['word_counts']) / len(data['word_counts'])
        
        # Normalize word count to 0-100 scale (assuming target is 150 words)
        normalized_word_count = min(100, (avg_word_count / 150) * 100)
        
        fig.add_trace(go.Scatterpolar(
            r=[avg_pass_rate, avg_coverage, normalized_word_count],
            theta=['Pass Rate', 'Vocabulary Coverage', 'Story Length'],
            fill='toself',
            name=model,
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Model Performance Comparison (Averaged Across Languages)",
        height=500
    )
    
    return fig


def create_execution_time_chart(summary: Dict[str, Any]) -> go.Figure:
    """Create a box plot showing execution times."""
    models = []
    times = []
    
    for key, stats in summary.items():
        if isinstance(stats, dict) and 'model' in stats:
            models.append(stats['model'])
            times.append(stats.get('avg_execution_time', 0))
    
    df = pd.DataFrame({
        'Model': models,
        'Execution Time (s)': times
    })
    
    fig = px.box(
        df,
        x='Model',
        y='Execution Time (s)',
        title='Execution Time Distribution by Model',
        color='Model',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        showlegend=False,
        height=400
    )
    
    return fig


def create_language_comparison_chart(summary: Dict[str, Any]) -> go.Figure:
    """Create a grouped bar chart comparing languages."""
    languages_data = {}
    
    for key, stats in summary.items():
        if isinstance(stats, dict) and 'language' in stats:
            lang = stats['language']
            if lang not in languages_data:
                languages_data[lang] = {
                    'pass_rates': [],
                    'coverages': [],
                    'models': []
                }
            
            languages_data[lang]['pass_rates'].append(stats.get('pass_rate', 0) * 100)
            languages_data[lang]['coverages'].append(stats.get('avg_vocabulary_coverage', 0) * 100)
            languages_data[lang]['models'].append(stats.get('model', ''))
    
    # Calculate averages for each language
    languages = []
    avg_pass_rates = []
    avg_coverages = []
    
    for lang, data in languages_data.items():
        languages.append(lang)
        avg_pass_rates.append(sum(data['pass_rates']) / len(data['pass_rates']))
        avg_coverages.append(sum(data['coverages']) / len(data['coverages']))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Pass Rate (%)',
        x=languages,
        y=avg_pass_rates,
        marker_color='lightblue',
        text=[f'{v:.1f}%' for v in avg_pass_rates],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Vocabulary Coverage (%)',
        x=languages,
        y=avg_coverages,
        marker_color='lightgreen',
        text=[f'{v:.1f}%' for v in avg_coverages],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Performance by Language (Averaged Across Models)',
        xaxis_title='Language',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=500
    )
    
    return fig


def create_dashboard_html(
    summary: Dict[str, Any],
    output_dir: Path,
    details_path: Optional[Path] = None
) -> Path:
    """Create an HTML dashboard with all visualizations."""
    
    # Create all charts
    pass_rate_chart = create_pass_rate_chart(summary)
    coverage_chart = create_vocabulary_coverage_chart(summary)
    performance_chart = create_performance_comparison(summary)
    time_chart = create_execution_time_chart(summary)
    language_chart = create_language_comparison_chart(summary)
    
    # Create summary statistics
    total_trials = sum(1 for k, v in summary.items() if isinstance(v, dict) and 'trials' in v)
    total_models = len(set(v['model'] for v in summary.values() if isinstance(v, dict) and 'model' in v))
    total_languages = len(set(v['language'] for v in summary.values() if isinstance(v, dict) and 'language' in v))
    
    avg_pass_rate = sum(v.get('pass_rate', 0) for v in summary.values() if isinstance(v, dict)) / max(1, len(summary)) * 100
    
    # Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vocabulary Benchmark Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                color: #2d3748;
                font-size: 2.5em;
            }}
            .header .subtitle {{
                color: #718096;
                font-size: 1.2em;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }}
            .stat-label {{
                color: #718096;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .chart-container {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .chart-title {{
                font-size: 1.5em;
                color: #2d3748;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #e2e8f0;
            }}
            .footer {{
                text-align: center;
                padding: 30px;
                color: white;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Vocabulary Inclusion Benchmark Results</h1>
                <div class="subtitle">Comprehensive analysis of language model performance with vocabulary constraints</div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Models</div>
                    <div class="stat-value">{total_models}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Languages Tested</div>
                    <div class="stat-value">{total_languages}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Trials</div>
                    <div class="stat-value">{total_trials}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Pass Rate</div>
                    <div class="stat-value">{avg_pass_rate:.1f}%</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">📊 Pass Rate by Model and Language</div>
                <div id="pass-rate-chart"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🌍 Performance by Language</div>
                <div id="language-chart"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🎭 Model Performance Comparison</div>
                <div id="performance-chart"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🔥 Vocabulary Coverage Heatmap</div>
                <div id="coverage-chart"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">⏱️ Execution Time Distribution</div>
                <div id="time-chart"></div>
            </div>
            
            <div class="footer">
                Generated by Vocabulary Inclusion Benchmark | View raw data in {output_dir}
            </div>
        </div>
        
        <script>
            Plotly.newPlot('pass-rate-chart', {pass_rate_chart.to_json()}.data, {pass_rate_chart.to_json()}.layout, {{responsive: true}});
            Plotly.newPlot('coverage-chart', {coverage_chart.to_json()}.data, {coverage_chart.to_json()}.layout, {{responsive: true}});
            Plotly.newPlot('performance-chart', {performance_chart.to_json()}.data, {performance_chart.to_json()}.layout, {{responsive: true}});
            Plotly.newPlot('time-chart', {time_chart.to_json()}.data, {time_chart.to_json()}.layout, {{responsive: true}});
            Plotly.newPlot('language-chart', {language_chart.to_json()}.data, {language_chart.to_json()}.layout, {{responsive: true}});
        </script>
    </body>
    </html>
    """
    
    # Save HTML
    dashboard_path = output_dir / "dashboard.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return dashboard_path


def display_terminal_charts(results: List[Any]):
    """Display a summary of charts in the terminal."""
    console.print("\n[bold cyan]📊 Benchmark Results Summary[/bold cyan]\n")
    
    # Handle both list of results and dict format
    if isinstance(results, list):
        # Convert list of results to summary format
        models_data = {}
        for result in results:
            if hasattr(result, 'model_label'):
                model = result.model_label
                if model not in models_data:
                    models_data[model] = []
                if result.success and result.validation.get('pass', False):
                    models_data[model].append(100.0)
                else:
                    models_data[model].append(0.0)
    else:
        # Original dict processing
        models_data = {}
        for key, stats in results.items():
            if isinstance(stats, dict) and 'model' in stats:
                model = stats['model']
                if model not in models_data:
                    models_data[model] = []
            models_data[model].append(stats.get('pass_rate', 0) * 100)
    
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
    language_data = {}
    for key, stats in summary.items():
        if isinstance(stats, dict) and 'language' in stats:
            lang = stats['language']
            if lang not in language_data:
                language_data[lang] = {'pass_rate': [], 'coverage': []}
            language_data[lang]['pass_rate'].append(stats.get('pass_rate', 0) * 100)
            language_data[lang]['coverage'].append(stats.get('avg_vocabulary_coverage', 0) * 100)
    
    if language_data:
        from rich.table import Table
        
        table = Table(title="Language Performance Summary", show_header=True, header_style="bold magenta")
        table.add_column("Language", style="cyan", no_wrap=True)
        table.add_column("Avg Pass Rate", justify="right", style="green")
        table.add_column("Avg Coverage", justify="right", style="yellow")
        
        for lang, data in sorted(language_data.items()):
            avg_pass = sum(data['pass_rate']) / len(data['pass_rate'])
            avg_cov = sum(data['coverage']) / len(data['coverage'])
            
            pass_color = "green" if avg_pass >= 80 else "yellow" if avg_pass >= 50 else "red"
            cov_color = "green" if avg_cov >= 10 else "yellow" if avg_cov >= 5 else "red"
            
            table.add_row(
                lang,
                f"[{pass_color}]{avg_pass:.1f}%[/{pass_color}]",
                f"[{cov_color}]{avg_cov:.1f}%[/{cov_color}]"
            )
        
        console.print("\n")
        console.print(table)
        console.print("\n")