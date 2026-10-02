"""Render benchmarks.jsonl (from `benchmark.py --log`) as a self-contained HTML page.

    python benchmark_report.py                                  # benchmarks.jsonl -> benchmark_report.html
    python benchmark_report.py --log benchmarks.jsonl --out report.html --open

No server or dependencies: the data is embedded in the page and rendered with plain JS/SVG.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark Results</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #e3e2dd;
  --good: #1f8a4c; --warn: #b7791f; --bad: #c0392b; --accent: #2f6fdb; --accent2: #8a4fd8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --panel: #1f1f1d; --text: #ecebe6; --muted: #9c9b95; --line: #33332f;
    --good: #4cc27d; --warn: #e0a54a; --bad: #ef6b5b; --accent: #6b9cf0; --accent2: #b58af0;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 12px; }
.sub { color: var(--muted); margin: 0 0 20px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
         padding: 16px; margin-bottom: 16px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
td.num, th.num { text-align: right; }
tr.sel td { background: color-mix(in srgb, var(--accent) 10%, transparent); }
tbody tr { cursor: default; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.chart svg { width: 100%; height: 200px; display: block; }
.axis { stroke: var(--line); } .tick { fill: var(--muted); font-size: 11px; }
.mark-good { color: var(--good); font-weight: 700; } .mark-warn { color: var(--warn); font-weight: 700; }
.mark-bad { color: var(--bad); font-weight: 700; }
.bar { display: inline-block; height: 8px; border-radius: 4px; background: var(--accent); vertical-align: middle; }
.wrap { white-space: normal; min-width: 220px; }
select { font: inherit; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--line);
         background: var(--panel); color: var(--text); max-width: 100%; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px;
        background: color-mix(in srgb, var(--accent) 15%, transparent); }
.empty { color: var(--muted); }
</style>
</head>
<body>
<main>
  <h1>Benchmark Results</h1>
  <p class="sub">Factual probes and true-versus-false pairs from <code>benchmark.py</code>. Generated __GENERATED__.</p>

  <section class="panel">
    <h2>Runs</h2>
    <table id="runs"><thead><tr>
      <th>When</th><th>Checkpoint</th><th class="num">Step</th><th class="num">Probe top-1</th>
      <th class="num">Top-5</th><th class="num">Answer log-p</th><th class="num">Pairs won</th>
      <th class="num">Margin (bpb)</th></tr></thead><tbody></tbody></table>
  </section>

  <section class="charts">
    <div class="panel chart"><h2>Pairs won vs step</h2><svg id="c-pairs"></svg></div>
    <div class="panel chart"><h2>Mean answer log-prob vs step</h2><svg id="c-logp"></svg></div>
  </section>

  <section class="panel">
    <h2>Run detail <select id="pick"></select></h2>
    <h2 style="margin-top:12px">Probes</h2>
    <table id="probes"><thead><tr><th></th><th>Prompt</th><th>Answer</th><th class="num">Rank</th>
      <th class="num">p(first token)</th><th>Model's top-3</th></tr></thead><tbody></tbody></table>
    <h2 style="margin-top:20px">True vs false pairs</h2>
    <table id="pairs"><thead><tr><th></th><th>True</th><th>False</th><th class="num">bpb true</th>
      <th class="num">bpb false</th><th class="num">Margin</th></tr></thead><tbody></tbody></table>
  </section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const runs = JSON.parse(document.getElementById("data").textContent);
const pct = x => (100 * x).toFixed(0) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const short = p => p.split("/").slice(-2).join("/");

function renderRuns(sel) {
  const tb = document.querySelector("#runs tbody");
  tb.innerHTML = runs.length ? "" : '<tr><td colspan="8" class="empty">No runs logged yet.</td></tr>';
  runs.forEach((r, i) => {
    const tr = document.createElement("tr");
    if (i === sel) tr.className = "sel";
    tr.innerHTML = `<td>${esc(r.time || "")}</td><td>${esc(short(r.ckpt))}${r.chat ? ' <span class="pill">SFT</span>' : ""}</td>
      <td class="num">${r.step ?? "?"}</td><td class="num">${pct(r.top1)}</td><td class="num">${pct(r.top5)}</td>
      <td class="num">${r.mean_answer_logprob.toFixed(2)}</td><td class="num">${pct(r.pair_acc)}</td>
      <td class="num">${(r.mean_pair_margin_bpb >= 0 ? "+" : "") + r.mean_pair_margin_bpb.toFixed(3)}</td>`;
    tr.onclick = () => pick(i);
    tb.appendChild(tr);
  });
}

function chart(id, key, fmt) {
  const svg = document.getElementById(id);
  const pts = runs.filter(r => r.step != null).map(r => ({x: r.step, y: r[key], chat: r.chat}));
  const W = 400, H = 200, L = 44, R = 12, T = 12, B = 28;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!pts.length) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" class="tick" text-anchor="middle">No data</text>`; return; }
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x0 === x1) { x0 -= 1; x1 += 1; } if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const pad = (y1 - y0) * 0.15; y0 -= pad; y1 += pad;
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R), sy = y => H - B - (y - y0) / (y1 - y0) * (H - T - B);
  let s = `<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H-B}"/>`;
  for (let k = 0; k <= 3; k++) {
    const y = y0 + (y1 - y0) * k / 3;
    s += `<text class="tick" x="${L-6}" y="${sy(y)+4}" text-anchor="end">${fmt(y)}</text>`;
    const x = x0 + (x1 - x0) * k / 3;
    s += `<text class="tick" x="${sx(x)}" y="${H-8}" text-anchor="middle">${Math.round(x).toLocaleString()}</text>`;
  }
  const base = pts.filter(p => !p.chat).sort((a, b) => a.x - b.x);
  if (base.length > 1) s += `<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${base.map(p => sx(p.x)+","+sy(p.y)).join(" ")}"/>`;
  pts.forEach(p => { s += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="4" fill="${p.chat ? "var(--accent2)" : "var(--accent)"}"><title>step ${p.x}: ${fmt(p.y)}${p.chat ? " (SFT)" : ""}</title></circle>`; });
  svg.innerHTML = s;
}

function pick(i) {
  document.getElementById("pick").value = i;
  renderRuns(i);
  const r = runs[i];
  const maxLog = Math.log10(Math.max(...r.probes.map(p => p.rank), 10));
  document.querySelector("#probes tbody").innerHTML = r.probes.map(p => {
    const m = p.rank === 1 ? ["✓", "good"] : p.rank <= 5 ? ["~", "warn"] : ["✗", "bad"];
    const w = Math.max(4, 120 * (1 - Math.log10(p.rank) / maxLog));
    return `<tr><td class="mark-${m[1]}">${m[0]}</td><td class="wrap">${esc(p.prompt)}</td><td><b>${esc(p.answer.trim())}</b></td>
      <td class="num"><span class="bar" style="width:${w}px"></span> ${p.rank.toLocaleString()}</td>
      <td class="num">${(100 * p.p_first).toFixed(1)}%</td><td>${p.top3.map(t => "<code>" + esc(t) + "</code>").join(" ")}</td></tr>`;
  }).join("");
  document.querySelector("#pairs tbody").innerHTML = r.pairs.map(p => {
    const d = p.bpb_false - p.bpb_true;
    return `<tr><td class="mark-${p.won ? "good" : "bad"}">${p.won ? "✓" : "✗"}</td><td class="wrap">${esc(p.true)}</td>
      <td class="wrap">${esc(p.false)}</td><td class="num">${p.bpb_true.toFixed(3)}</td><td class="num">${p.bpb_false.toFixed(3)}</td>
      <td class="num mark-${d > 0 ? "good" : "bad"}">${(d >= 0 ? "+" : "") + d.toFixed(3)}</td></tr>`;
  }).join("");
}

const sel = document.getElementById("pick");
runs.forEach((r, i) => { const o = document.createElement("option"); o.value = i;
  o.textContent = `${short(r.ckpt)} · step ${r.step ?? "?"}${r.chat ? " · SFT" : ""} · ${r.time || ""}`; sel.appendChild(o); });
sel.onchange = e => pick(+e.target.value);
chart("c-pairs", "pair_acc", y => pct(y));
chart("c-logp", "mean_answer_logprob", y => y.toFixed(2));
if (runs.length) pick(runs.length - 1); else renderRuns(-1);
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="benchmarks.jsonl")
    ap.add_argument("--out", default="benchmark_report.html")
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    args = ap.parse_args()

    runs = [json.loads(l) for l in Path(args.log).read_text().splitlines() if l.strip()] if Path(args.log).exists() else []
    import time
    html = (PAGE.replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M"))
                .replace("__DATA__", json.dumps(runs).replace("</", "<\\/")))
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(runs)} run{'s' if len(runs) != 1 else ''})")
    if args.open:
        webbrowser.open(Path(args.out).resolve().as_uri())


if __name__ == "__main__":
    main()
