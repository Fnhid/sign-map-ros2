#!/usr/bin/env python3
import argparse
import http.server
import socketserver
from datetime import datetime, timezone
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description="Serve sign-map-ros2 outputs with Python HTTP server.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def _fmt_mtime(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def build_runs_index(root: Path) -> None:
    rows = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        run = d.name
        mtime = _fmt_mtime(d.stat().st_mtime)

        viewer_link = "-"
        if (d / "viewer.html").exists():
            viewer_link = f'<a href="./{escape(run)}/viewer.html">viewer</a>'
        elif (d / "qualitative_viewer.html").exists():
            viewer_link = f'<a href="./{escape(run)}/qualitative_viewer.html">qual_viewer</a>'

        report_link = "-"
        if (d / "ocr_report.html").exists():
            report_link = f'<a href="./{escape(run)}/ocr_report.html">ocr_report</a>'
        elif (d / "kidnap_consistency_report.json").exists():
            report_link = f'<a href="./{escape(run)}/kidnap_consistency_report.json">kidnap_report</a>'
        elif (d / "summary.json").exists():
            report_link = f'<a href="./{escape(run)}/summary.json">summary.json</a>'

        extra_links = []
        for rel, title in [
            ("poses_metrics.json", "poses_metrics"),
            ("sign_3d_detections.json", "3d_dets"),
            ("nav_frame_index.json", "nav_index"),
        ]:
            if (d / rel).exists():
                extra_links.append(f'<a href="./{escape(run)}/{rel}">{title}</a>')
        extras = " | ".join(extra_links) if extra_links else "-"

        run_link = f'<a href="./{escape(run)}/">{escape(run)}</a>'
        rows.append(
            "<tr>"
            f"<td>{run_link}</td>"
            f"<td>{mtime}</td>"
            f"<td>{viewer_link}</td>"
            f"<td>{report_link}</td>"
            f"<td>{extras}</td>"
            "</tr>"
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Sign Map Runs</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:16px;color:#222;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #d8dde6;padding:6px;text-align:left;font-size:13px;}"
        "th{background:#f2f5f9;position:sticky;top:0;}"
        "a{text-decoration:none;color:#114b9b;}"
        "a:hover{text-decoration:underline;}"
        "</style></head><body>"
        "<h2>Sign Map Web Runs (Auto)</h2>"
        "<div style='margin-bottom:10px;font-size:13px;color:#4a5568;'>"
        f"Updated: {_fmt_mtime(datetime.now(tz=timezone.utc).timestamp())} | Root: {escape(str(root))}"
        "</div>"
        "<table><tr><th>Run</th><th>Updated</th><th>Viewer</th><th>Report</th><th>Extra</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )
    (root / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    build_runs_index(root)
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **kw)
    with ReusableTCPServer(("0.0.0.0", args.port), handler) as httpd:
        print(f"Serving {root} on http://0.0.0.0:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
