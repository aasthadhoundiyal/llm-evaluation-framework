"""
eval_framework.reporting.html_generator
=======================================
Generates rich, modern, dark-theme HTML evaluation reports.
Includes interactive filters, metric scorecards, accordion details, and responsive design.
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
GitHub: https://github.com/aasthadhoundiyal/llm-evaluation-framework
"""

import json
import os
import time
from typing import Any, Dict, List
from eval_framework.evaluators.base import EvaluationResult


class HTMLReportGenerator:
    """Generates standalone visual HTML reports from evaluation runs."""

    def __init__(self, title: str = "LLM Evaluation Report", subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle

    def generate_report(
        self,
        results: List[EvaluationResult],
        output_path: str = "results/report.html",
        metadata: Dict[str, Any] = None,
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        meta = metadata or {}

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = (passed_count / total * 100) if total else 0
        avg_latency = (sum(r.latency_ms for r in results) / total) if total else 0.0

        # Extract all distinct categories
        categories = sorted(
            list(set(r.metadata.get("category", "General") for r in results if r.metadata))
        )
        if not categories:
            categories = ["General"]

        rows_json = json.dumps([r.to_dict() for r in results])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.title} — Aastha Dhoundiyal</title>
  <style>
    :root {{
      --bg-base: #0a0d14;
      --bg-card: rgba(22, 27, 34, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --accent: #6366f1;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
      --pass-color: #10b981;
      --fail-color: #ef4444;
      --card-radius: 14px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif; }}
    body {{
      background-color: var(--bg-base);
      color: var(--text-main);
      padding: 30px 40px;
      line-height: 1.5;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 30px;
    }}
    .header-title h1 {{
      font-size: 26px;
      font-weight: 700;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .header-title p {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 4px;
    }}
    .author-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 20px;
      font-size: 13px;
      color: #c7d2fe;
    }}
    /* Scorecards Grid */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 18px;
      margin-bottom: 32px;
    }}
    .metric-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--card-radius);
      padding: 20px;
      backdrop-filter: blur(12px);
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card:hover {{
      transform: translateY(-2px);
      border-color: rgba(99, 102, 241, 0.4);
    }}
    .metric-label {{
      font-size: 13px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
    }}
    .metric-value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 8px;
      color: #fff;
    }}
    .metric-sub {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    .pass-val {{ color: var(--pass-color); }}
    .fail-val {{ color: var(--fail-color); }}

    /* Controls Bar */
    .controls-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 24px;
      align-items: center;
      justify-content: space-between;
    }}
    .search-box {{
      padding: 10px 16px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      color: #fff;
      font-size: 14px;
      min-width: 280px;
      outline: none;
    }}
    .search-box:focus {{ border-color: var(--accent); }}
    .filter-pills {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 6px 14px;
      border-radius: 20px;
      border: 1px solid var(--border-color);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-muted);
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .pill.active, .pill:hover {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}

    /* Test Case Rows */
    .cases-container {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .case-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--card-radius);
      padding: 18px 22px;
      backdrop-filter: blur(10px);
      cursor: pointer;
      transition: border-color 0.2s;
    }}
    .case-card:hover {{
      border-color: rgba(255, 255, 255, 0.2);
    }}
    .case-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .case-id-group {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .status-chip {{
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .status-pass {{ background: rgba(16, 185, 129, 0.15); color: var(--pass-color); border: 1px solid rgba(16, 185, 129, 0.3); }}
    .status-fail {{ background: rgba(239, 68, 68, 0.15); color: var(--fail-color); border: 1px solid rgba(239, 68, 68, 0.3); }}
    .case-id {{ font-size: 14px; font-weight: 600; color: #fff; }}
    .case-category {{ font-size: 12px; color: var(--text-muted); background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 4px; }}
    .case-prompt {{
      margin-top: 10px;
      font-size: 14px;
      color: #d1d5db;
    }}
    .case-body {{
      display: none;
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--border-color);
      font-size: 13px;
    }}
    .case-body.open {{ display: block; }}
    .section-title {{
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 6px;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }}
    .code-box {{
      background: #0d1117;
      padding: 12px;
      border-radius: 8px;
      font-family: monospace;
      white-space: pre-wrap;
      color: #c9d1d9;
      margin-bottom: 12px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    .metrics-table th, .metrics-table td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 13px;
    }}
    .metrics-table th {{ color: var(--text-muted); font-weight: 600; }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-title">
      <h1>{self.title}</h1>
      <p>{self.subtitle or f"Run at {time.strftime('%Y-%m-%d %H:%M:%S')} • Generated by llm-evaluation-framework"}</p>
    </div>
    <div class="author-badge">
      <span>Framework Author:</span>
      <strong>Aastha Dhoundiyal</strong>
    </div>
  </div>

  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Total Test Cases</div>
      <div class="metric-value">{total}</div>
      <div class="metric-sub">Evaluated Prompts</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Pass Rate</div>
      <div class="metric-value {'pass-val' if pass_rate >= 80 else 'fail-val'}">{pass_rate:.1f}%</div>
      <div class="metric-sub">{passed_count} Passed / {total - passed_count} Failed</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Average Latency</div>
      <div class="metric-value">{avg_latency:.1f} ms</div>
      <div class="metric-sub">Execution speed</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Primary Engine</div>
      <div class="metric-value" style="font-size: 22px;">{meta.get('provider', 'Mock / Universal')}</div>
      <div class="metric-sub">{meta.get('model', 'llama3.1:8b')}</div>
    </div>
  </div>

  <div class="controls-bar">
    <input type="text" id="searchInput" class="search-box" placeholder="Search prompts, responses, or test IDs..." onkeyup="filterCases()">
    <div class="filter-pills">
      <button class="pill active" onclick="setFilter('all', this)">All ({total})</button>
      <button class="pill" onclick="setFilter('pass', this)">Passed ({passed_count})</button>
      <button class="pill" onclick="setFilter('fail', this)">Failed ({total - passed_count})</button>
    </div>
  </div>

  <div class="cases-container" id="casesContainer"></div>

  <script>
    const data = {rows_json};
    let currentFilter = 'all';

    function renderCases(items) {{
      const container = document.getElementById('casesContainer');
      container.innerHTML = '';

      if (items.length === 0) {{
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No test cases match the active filter.</div>';
        return;
      }}

      items.forEach(c => {{
        const card = document.createElement('div');
        card.className = 'case-card';
        card.onclick = (e) => {{
          const body = card.querySelector('.case-body');
          body.classList.toggle('open');
        }};

        const statusClass = c.passed ? 'status-pass' : 'status-fail';
        const statusText = c.passed ? 'PASS' : 'FAIL';
        const cat = (c.metadata && c.metadata.category) ? c.metadata.category : 'General';

        let metricsRows = '';
        (c.metrics || []).forEach(m => {{
          const mStatus = m.passed ? '<span style="color:#10b981;">PASS</span>' : '<span style="color:#ef4444;">FAIL</span>';
          metricsRows += `<tr>
            <td><strong>${{m.name}}</strong></td>
            <td>${{typeof m.score === 'number' ? m.score.toFixed(2) : m.score}}</td>
            <td>${{m.threshold !== null ? m.threshold : 'N/A'}}</td>
            <td>${{mStatus}}</td>
            <td style="color:#8b949e;">${{m.reasoning || ''}}</td>
          </tr>`;
        }});

        card.innerHTML = `
          <div class="case-header">
            <div class="case-id-group">
              <span class="status-chip ${{statusClass}}">${{statusText}}</span>
              <span class="case-id">${{c.test_id}}</span>
              <span class="case-category">${{cat}}</span>
            </div>
            <div style="color:#8b949e; font-size:12px;">${{c.latency_ms.toFixed(1)}} ms</div>
          </div>
          <div class="case-prompt"><strong>Prompt:</strong> ${{escapeHtml(c.prompt)}}</div>
          <div class="case-body">
            <div class="section-title">Actual LLM Output</div>
            <div class="code-box">${{escapeHtml(c.actual_output)}}</div>
            ${{c.expected_output ? `<div class="section-title">Expected / Reference Target</div><div class="code-box">${{escapeHtml(c.expected_output)}}</div>` : ''}}
            <div class="section-title">Evaluation Metrics & Checks</div>
            <table class="metrics-table">
              <thead><tr><th>Metric</th><th>Score</th><th>Threshold</th><th>Result</th><th>Notes</th></tr></thead>
              <tbody>${{metricsRows}}</tbody>
            </table>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    function filterCases() {{
      const q = document.getElementById('searchInput').value.toLowerCase();
      const filtered = data.filter(c => {{
        const matchesFilter = (currentFilter === 'all') ||
                              (currentFilter === 'pass' && c.passed) ||
                              (currentFilter === 'fail' && !c.passed);
        const matchesSearch = !q ||
                              c.test_id.toLowerCase().includes(q) ||
                              c.prompt.toLowerCase().includes(q) ||
                              c.actual_output.toLowerCase().includes(q);
        return matchesFilter && matchesSearch;
      }});
      renderCases(filtered);
    }}

    function setFilter(f, el) {{
      currentFilter = f;
      document.querySelectorAll('.filter-pills .pill').forEach(b => b.classList.remove('active'));
      el.classList.add('active');
      filterCases();
    }}

    function escapeHtml(str) {{
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }}

    // Initial Render
    renderCases(data);
  </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return os.path.abspath(output_path)
