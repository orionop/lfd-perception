"""
Per-trial diagnostic report (single PDF).

One artifact you can hand the lab after every bag: a one-document review
with the timeline, event frames, propagated mask strips, mask-area plot,
DADO comparison (if available), and a summary table from the sidecar.

Builds a LaTeX file under Docs/reports/, compiles to PDF, opens it.

Standalone — uses only the analysis venv (LaTeX is system-installed).

Usage:
    .venv_analysis/bin/python Code/trial_report.py --trial Data/lfdws_t001/lfdws_t001
"""
import argparse
import csv
import json
import os
import subprocess
from datetime import datetime

REPORT_DIR = "Docs/reports"
# filenames only -- joined with --fig_dir at render time, since each trial's
# figures live under its own subdirectory (figures/<trial>/...), not the
# fixed global figures/ path (that's lfdws_t001's canonical location only)
FIG_REFS = [
    ("timeline.png",            "Proprioceptive timeline (gripper width + force magnitude)."),
    ("event_frames.png",        "Raw ZED frames at the detected events."),
    ("segmented_events.png",    "SAM (ViT-H) on event frames."),
    ("propagation_both_strip.png", "SAM 2 propagation, carrot (green) and cup (magenta)."),
    ("mask_area_over_time.png", "Per-frame mask area over the demo."),
    ("force_overlay_press.png", "Press frame: carrot + cup masks + uncalibrated force arrow."),
    ("dado_events.png",         "DADO-style label-free baseline (negative result)."),
]

SIDECAR_JSON_DEFAULT = "figures/identify/objects.json"


def latex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#")
            .replace("$", r"\$").replace("{", r"\{").replace("}", r"\}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out_basename", default=None,
                    help="basename for tex/pdf in Docs/reports/ (auto-derived if omitted)")
    ap.add_argument("--sidecar_json", default=SIDECAR_JSON_DEFAULT,
                    help="build_sidecar.py output for this trial "
                         "(different trials should use different --out, "
                         "e.g. figures/identify_depth/objects.json)")
    ap.add_argument("--fig_dir", default="figures",
                    help="directory containing this trial's timeline.png/"
                         "event_frames.png/etc (default 'figures' matches "
                         "lfdws_t001's canonical location; pass e.g. "
                         "figures/t004 for other trials)")
    ap.add_argument("--no_open", action="store_true")
    args = ap.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    trial_id = os.path.basename(args.trial.rstrip("/")) or "trial"
    base = args.out_basename or f"report_{trial_id}"
    tex_path = os.path.join(REPORT_DIR, f"{base}.tex")

    # ---- load sidecar if present ----
    sidecar = None
    if os.path.exists(args.sidecar_json):
        with open(args.sidecar_json) as f:
            sidecar = json.load(f)
        print(f"[load] {args.sidecar_json}", flush=True)
    else:
        print(f"[warn] {args.sidecar_json} missing -- report will skip object summary",
              flush=True)

    # ---- compose LaTeX ----
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{booktabs}",
        r"\usepackage{xcolor}",
        r"\usepackage{enumitem}",
        r"\usepackage{caption}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\begin{document}",
        r"\begin{center}{\Large\textbf{Trial Diagnostic Report --- \texttt{"
        + latex_escape(trial_id) + r"}}}\\",
        r"\small Generated " + latex_escape(datetime.now().strftime("%Y-%m-%d %H:%M"))
        + r"\\[6pt]\end{center}",
    ]

    # ---- summary header ----
    if sidecar is not None:
        ev = sidecar.get("events", {})
        objs = sidecar.get("objects", {})
        frames = sidecar.get("frames", [])
        lines.append(r"\section*{Summary}")
        lines.append(r"\begin{tabular}{lll}\toprule")
        lines.append(r"\textbf{Item} & \textbf{Value} & \textbf{Notes} \\\midrule")
        lines.append(f"trial dir & {latex_escape(sidecar.get('trial_dir',''))} & \\\\")
        lines.append(f"merged CSV & {latex_escape(sidecar.get('csv',''))} & \\\\")
        lines.append(f"image dir & {latex_escape(sidecar.get('image_dir',''))} & \\\\")
        lines.append(f"frames with at least one mask & {len(frames)} & per-frame entries in JSON \\\\")
        lines.append(f"tracked objects & {len(objs)} & by stable obj\\_id \\\\")
        lines.append(r"\bottomrule\end{tabular}")

        # events
        lines.append(r"\subsection*{Events}")
        lines.append(r"\begin{tabular}{llll}\toprule")
        lines.append(r"\textbf{event} & \textbf{$t_{\text{rel}}$ (s)} & \textbf{row idx} & \textbf{img ts} \\\midrule")
        for name, ed in ev.items():
            lines.append(
                f"{latex_escape(name)} & {ed.get('t_rel_s',-1):.2f} & "
                f"{ed.get('row_idx','?')} & {latex_escape(ed.get('img_ts',''))} \\\\"
            )
        lines.append(r"\bottomrule\end{tabular}")

        # objects
        lines.append(r"\subsection*{Tracked objects}")
        lines.append(r"\begin{tabular}{lll}\toprule")
        lines.append(r"\textbf{obj\_id} & \textbf{role} & \textbf{colour (BGR)} \\\midrule")
        for oid, info in objs.items():
            col = info.get("color", [])
            lines.append(
                f"{latex_escape(oid)} & {latex_escape(info.get('role',''))} & "
                f"{latex_escape(tuple(col))} \\\\"
            )
        lines.append(r"\bottomrule\end{tabular}")

        # press-frame highlight (if present)
        press = ev.get("press")
        if press is not None:
            press_idx = press.get("row_idx")
            # find the closest frame in sidecar matching the press image ts
            press_img = press.get("img_ts")
            row = next((f for f in frames if f["img_filename"]
                        == f"{press_img}.png"), None)
            if row:
                lines.append(r"\subsection*{Press-frame entry (sidecar excerpt)}")
                lines.append(r"\begin{tabular}{llrr}\toprule")
                lines.append(r"\textbf{obj\_id} & \textbf{role} & \textbf{bbox xyxy} & \textbf{mask px} \\\midrule")
                for o in row.get("objects", []):
                    bb = o.get("bbox_xyxy") or [-1, -1, -1, -1]
                    lines.append(
                        f"{o['obj_id']} & {latex_escape(o['role'])} & "
                        f"{latex_escape(bb)} & {o['mask_px']} \\\\"
                    )
                lines.append(r"\bottomrule\end{tabular}")

    # ---- figures ----
    lines.append(r"\section*{Figures}")
    for fname, cap in FIG_REFS:
        path = os.path.join(args.fig_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip fig] {path}", flush=True)
            continue
        # report sits 2 levels deep (Docs/reports/) so figure paths go up
        rel = os.path.join("..", "..", path)
        lines.append(r"\begin{figure}[H]\centering")
        lines.append(r"\includegraphics[width=\textwidth]{" + rel + r"}")
        lines.append(r"\caption{" + latex_escape(cap) + r"}")
        lines.append(r"\end{figure}")

    lines.append(r"\end{document}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[write] {tex_path}", flush=True)

    print("[compile] pdflatex pass 1 ...", flush=True)
    subprocess.run(
        ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode",
         f"-output-directory={REPORT_DIR}", tex_path],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("[compile] pdflatex pass 2 ...", flush=True)
    res = subprocess.run(
        ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode",
         f"-output-directory={REPORT_DIR}", tex_path],
        capture_output=True, text=True,
    )
    pdf_path = os.path.join(REPORT_DIR, f"{base}.pdf")
    if not os.path.exists(pdf_path):
        print(res.stdout[-1500:], flush=True)
        print(f"[fail] {pdf_path} not produced", flush=True); return
    print(f"[compile] -> {pdf_path}", flush=True)
    if not args.no_open:
        subprocess.run(["open", pdf_path])
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
