#!/usr/bin/env python
"""Build a self-contained HTML results page (and a PNG contact sheet) for the generated dataset.

    python data_processing/make_preview.py --n 6 --out data/datasets/stm_gen/preview/index.html \
        --target 100000 --rate-done-per-hour 24

Each example shows the original clip, the Segment-Any-Motion mask, the foreground layer and the
MiniMax-Remover background as small looping videos embedded as data URIs (no external assets).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stm.data_generation.config import DecomposerConfig  # noqa: E402
from stm.data_generation.manifest import load_records, summarize  # noqa: E402
from stm.data_generation.video_io import read_video, write_video  # noqa: E402

KINDS = [
    ("videos", "", "Input clip"),
    ("masks", "-00", "Segment-Any-Motion mask"),
    ("fg", "-00", "Foreground layer"),
    ("bg-inpainted", "-bg", "Inpainted background"),
]


def small_video_data_uri(path: Path, size=(360, 240)) -> str:
    import cv2

    frames = read_video(path)
    small = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.mp4"
        write_video(p, small, fps=16)
        return "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()


def contact_sheet(cfg: DecomposerConfig, examples, out_png: Path) -> None:
    import cv2

    rows = []
    for r in examples:
        row = []
        for kind, suf, _ in KINDS:
            v = read_video(cfg.dataset_root / r["source"] / kind / f"{r['clip_id']}{suf}.mp4", num_frames=25)
            row.append(cv2.resize(v[-1], (240, 160), interpolation=cv2.INTER_AREA))
        rows.append(np.concatenate(row, axis=1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def pick_examples(recs, n: int, prefer: str):
    done = [r for r in recs if r.get("status") == "done"]
    pool = [r for r in done if r["source"] == prefer] or done
    pool = sorted(pool, key=lambda r: r["foreground_ratio"])
    if len(pool) <= n:
        return pool
    idx = np.linspace(0, len(pool) - 1, n).round().astype(int)  # spread over the fg-ratio range
    return [pool[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--prefer", default="openvid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--rate-done-per-hour", type=float, default=None)
    ap.add_argument("--rate-note", default="")
    args = ap.parse_args()

    cfg = DecomposerConfig()
    recs = load_records(cfg)
    done = [r for r in recs if r.get("status") == "done"]
    counts = summarize(cfg)
    skip_fg = sum(v for k, v in counts.items() if k.startswith("skipped:fg_ratio"))
    skip_nomotion = counts.get("skipped:no_dynamic_object", 0)
    by_src = {}
    for r in done:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    examples = pick_examples(recs, args.n, args.prefer)
    out = Path(args.out) if args.out else cfg.dataset_root / "preview" / "index.html"
    contact_sheet(cfg, examples, out.with_name("contact_sheet.png"))

    seg_summary = {}
    p = cfg.dataset_root / "segmenter_eval" / "summary.json"
    if p.exists():
        seg_summary = json.loads(p.read_text())

    timings = {
        k: float(np.median([r["timings"][k] for r in done if r.get("timings", {}).get(k)]))
        for k in ("mask", "caption", "inpaint", "total")
        if any(r.get("timings", {}).get(k) for r in done)
    }
    ratios = np.array([r["foreground_ratio"] for r in done]) if done else np.zeros(0)
    n_done = len(done)
    remaining = max(args.target - n_done, 0)
    if args.rate_done_per_hour:
        eta_days = remaining / args.rate_done_per_hour / 24
        eta_txt = f"{eta_days:.0f} days" if eta_days >= 2 else f"{eta_days * 24:.0f} h"
        rate_txt = f"{args.rate_done_per_hour:.0f} / h"
    else:
        eta_txt, rate_txt = "—", "—"

    cards = []
    for r in examples:
        vids = ""
        for kind, suf, label in KINDS:
            uri = small_video_data_uri(cfg.dataset_root / r["source"] / kind / f"{r['clip_id']}{suf}.mp4")
            vids += f'<figure><video autoplay muted loop playsinline src="{uri}"></video><figcaption>{label}</figcaption></figure>'

        cap = html.escape((r.get("caption") or "")[:260] + ("…" if len(r.get("caption") or "") > 260 else ""))
        cards.append(
            f'<article class="example"><div class="strip">{vids}</div>'
            f'<div class="meta"><span class="tag">{html.escape(r["source"])}</span><span class="tag">fg ratio {r["foreground_ratio"]:.2f}</span>'
            f'<span class="tag">{r.get("num_objects", 1)} object{"s" if r.get("num_objects", 1) != 1 else ""}</span>'
            f'<span class="tag">{r.get("timings", {}).get("total", 0):.0f} s</span></div><p class="caption">{cap}</p></article>'
        )

    seg_html = ""
    if seg_summary:
        seg_html = (
            f'<div class="tiles small"><div class="tile"><span class="num">{seg_summary["mean_iou"]:.2f}</span><span class="lbl">mean IoU</span></div>'
            f'<div class="tile"><span class="num">{seg_summary["median_iou"]:.2f}</span><span class="lbl">median IoU</span></div>'
            f'<div class="tile"><span class="num">{100 * seg_summary["iou>=0.5"]:.0f}%</span><span class="lbl">sequences ≥ 0.5</span></div>'
            f'<div class="tile"><span class="num">{seg_summary["n"]}</span><span class="lbl">DAVIS sequences</span></div></div>'
        )

    hist_bins = [0, 0.05, 0.1, 0.2, 0.3, 0.45]
    hist = np.histogram(ratios, bins=hist_bins)[0].tolist() if len(ratios) else [0] * 5
    hist_max = max(hist) or 1
    bars = "".join(
        f'<div class="bar"><div class="fill" style="height:{100 * h / hist_max:.0f}%"></div><span>{hist_bins[i]:.2f}–{hist_bins[i + 1]:.2f}</span><b>{h}</b></div>'
        for i, h in enumerate(hist)
    )

    page = f"""<title>StM-Gen Decomposer Results</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f4f3ee;--surface:#ffffff;--ink:#17191d;--muted:#5f6370;--line:#dcdad2;--accent:#d97b2b;--accent-ink:#8a4a12;--mask:#2f8f83;--fill:#e9e6dd}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0f1115;--surface:#171a21;--ink:#ece9e1;--muted:#9a9ca3;--line:#262a33;--accent:#f0a24a;--accent-ink:#ffd39a;--mask:#5ec9ba;--fill:#20242d}}}}
:root[data-theme="dark"]{{--bg:#0f1115;--surface:#171a21;--ink:#ece9e1;--muted:#9a9ca3;--line:#262a33;--accent:#f0a24a;--accent-ink:#ffd39a;--mask:#5ec9ba;--fill:#20242d}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5}}
main{{max-width:1180px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;font-weight:700;font-size:clamp(28px,4vw,44px);letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}}
h2{{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;font-weight:600;font-size:22px;margin:44px 0 14px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.lede{{max-width:64ch;color:var(--muted);margin:0 0 24px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.tile{{background:var(--surface);border:1px solid var(--line);padding:14px 16px;display:flex;flex-direction:column;gap:2px}}
.tile .num{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;font-size:26px;font-weight:500;color:var(--ink)}}
.tile.hot .num{{color:var(--accent)}}
.tile .lbl{{font-size:12.5px;color:var(--muted)}}
.tiles.small .tile .num{{font-size:22px}}
.example{{background:var(--surface);border:1px solid var(--line);padding:14px;margin-bottom:16px}}
.strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
figure{{margin:0}}
video{{width:100%;aspect-ratio:3/2;display:block;background:#000}}
figcaption{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-top:6px}}
.meta{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 6px}}
.tag{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;padding:3px 8px;border:1px solid var(--line);color:var(--muted)}}
.caption{{margin:0;font-size:14px;max-width:110ch;color:var(--ink)}}
.hist{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;align-items:end;height:150px;background:var(--surface);border:1px solid var(--line);padding:14px 16px 8px}}
.bar{{display:flex;flex-direction:column;justify-content:flex-end;height:100%;text-align:center;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted)}}
.bar .fill{{background:var(--mask);width:60%;margin:0 auto 6px}}
.bar b{{color:var(--ink);font-weight:500}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:var(--surface);border:1px solid var(--line)}}
td,th{{padding:8px 12px;border-top:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);border-top:none}}
td.num{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.wrap{{overflow-x:auto}}
code{{font-family:"IBM Plex Mono",monospace;font-size:13px;background:var(--fill);padding:1px 5px}}
@media (max-width:760px){{.strip{{grid-template-columns:repeat(2,1fr)}}}}
@media (prefers-reduced-motion: reduce){{video{{animation:none}}}}
</style>
<main>
<p class="eyebrow">Split-then-Merge · Decomposer · Segment-Any-Motion</p>
<h1>StM-Gen decomposition results</h1>
<p class="lede">Unlabeled videos split into caption, foreground mask, foreground layer and inpainted background — every clip 49 frames at 480×720, 16 fps. Videos below loop the actual dataset files.</p>
<div class="tiles">
  <div class="tile hot"><span class="num">{n_done:,}</span><span class="lbl">clips decomposed (target {args.target:,})</span></div>
  <div class="tile"><span class="num">{rate_txt}</span><span class="lbl">accepted clips per hour {html.escape(args.rate_note)}</span></div>
  <div class="tile hot"><span class="num">{eta_txt}</span><span class="lbl">ETA to target at this rate</span></div>
  <div class="tile"><span class="num">{skip_fg:,}</span><span class="lbl">skipped: fg ratio outside [0.01, 0.45]</span></div>
  <div class="tile"><span class="num">{skip_nomotion:,}</span><span class="lbl">skipped: no moving object</span></div>
  <div class="tile"><span class="num">{timings.get("total", 0):.0f} s</span><span class="lbl">median time per accepted clip (mask {timings.get("mask", 0):.0f} s · inpaint {timings.get("inpaint", 0):.0f} s)</span></div>
</div>

<h2>Examples</h2>
{"".join(cards)}

<h2>Foreground ratio of accepted clips</h2>
<div class="hist">{bars}</div>

<h2>Segmenter quality (DAVIS 2017, no ground truth used)</h2>
{seg_html or "<p class='lede'>benchmark not run yet</p>"}

<h2>Composition</h2>
<div class="wrap"><table><tr><th>source</th><th>accepted clips</th><th>role</th></tr>
{"".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v:,}</td><td>{'validation (GT masks)' if k == 'davis' else 'training'}</td></tr>" for k, v in sorted(by_src.items()))}
</table></div>
<p class="lede" style="margin-top:20px">Rebuild: <code>python data_processing/make_preview.py</code> · status: <code>python data_processing/generate_data.py status</code> · manifest: <code>python data_processing/generate_data.py manifest</code></p>
</main>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), contact sheet {out.with_name('contact_sheet.png')}")


if __name__ == "__main__":
    main()
