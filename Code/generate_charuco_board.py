"""Generate a print-ready ChArUco board matching calibrate_hand_eye.py's
solve step exactly, so the board Mark receives needs no parameter guessing
on either side.

The dictionary (DICT_5X5_100) is hard-coded in calibrate_hand_eye.py, not a
CLI arg there -- this script uses the identical dictionary constant so the
printed board is guaranteed detectable by the solver as long as the physical
board matches these defaults (or whatever --squares_x/--squares_y/
--square_size_m/--marker_size_m you pass here are then also passed to
calibrate_hand_eye.py's solve step).

Physical-size correctness (the actual point of this script): cv2's
generateImage() only guarantees a *minimum* margin and centers the board
inside whatever outSize you give it, which does not by itself give a
predictable pixels-per-metre mapping unless outSize's aspect ratio exactly
matches the board's. So this script generates the board at zero margin (outSize
sized to exactly match the board's own aspect ratio, giving an exact,
verifiable pixels-per-metre scale), then pads the margin itself with
cv2.copyMakeBorder -- deterministic, no dependence on generateImage's
internal fitting behaviour.

The PDF is written with matplotlib at the same DPI as the pixel generation
and with the axes filling the entire figure (no auto-margins), so the PDF
page's physical size in inches is exactly (pixels / dpi). Printed at 100%
/ "actual size" (NOT "fit to page"), the printed square size should match
--square_size_m to within normal printer accuracy -- but MEASURE THE PRINTOUT
before running calibrate_hand_eye.py's solve step and pass the measured
value, not the nominal one (printer scaling drift is a known silent error
source -- see calibrate_hand_eye.py's docstring).

Usage:
    .venv_analysis/bin/python Code/generate_charuco_board.py \
        --out_prefix charuco_board_5x7_35mm

Then print charuco_board_5x7_35mm.pdf at 100% scale, measure a square with
calipers, and pass the measured size to calibrate_hand_eye.py's
--square_size_m (and the marker's measured size to --marker_size_m).
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Must match calibrate_hand_eye.py's hard-coded dictionary exactly.
ARUCO_DICT = cv2.aruco.DICT_5X5_100
M_PER_INCH = 0.0254


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squares_x", type=int, default=5)
    parser.add_argument("--squares_y", type=int, default=7)
    parser.add_argument("--square_size_m", type=float, default=0.035,
                        help="nominal design square size; measure the actual "
                             "printout and use that for calibrate_hand_eye.py")
    parser.add_argument("--marker_size_m", type=float, default=0.026)
    parser.add_argument("--dpi", type=int, default=300,
                        help="print resolution; 300 is standard laser-printer "
                             "quality, plenty for ArUco marker detection")
    parser.add_argument("--margin_squares", type=float, default=1.0,
                        help="white margin around the board, in units of "
                             "square_size_m, required so ArUco corner "
                             "detection isn't clipped at the board edge")
    parser.add_argument("--out_prefix", default="charuco_board_5x7_35mm")
    args = parser.parse_args()

    if args.marker_size_m >= args.square_size_m:
        raise ValueError("marker_size_m must be smaller than square_size_m")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_size_m,
        args.marker_size_m, aruco_dict)

    px_per_m = args.dpi / M_PER_INCH
    board_w_px = round(args.squares_x * args.square_size_m * px_per_m)
    board_h_px = round(args.squares_y * args.square_size_m * px_per_m)
    margin_px = round(args.margin_squares * args.square_size_m * px_per_m)

    print(f"[gen] board-only image: {board_w_px}x{board_h_px}px at {args.dpi} dpi "
          f"({args.squares_x}x{args.squares_y} squares, "
          f"{args.square_size_m * 1000:.1f}mm/square, "
          f"{args.marker_size_m * 1000:.1f}mm/marker, DICT_5X5_100)")
    # marginSize=0 and outSize matched to the board's own aspect ratio ->
    # generateImage cannot letterbox, so the mapping is exactly px_per_m.
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0)
    full_img = cv2.copyMakeBorder(
        board_img, margin_px, margin_px, margin_px, margin_px,
        cv2.BORDER_CONSTANT, value=255)

    png_path = f"{args.out_prefix}.png"
    if not cv2.imwrite(png_path, full_img):
        raise RuntimeError(f"failed to write {png_path}")
    print(f"[write] {png_path} ({full_img.shape[1]}x{full_img.shape[0]}px, "
          f"reference only -- do not print this directly, use the PDF for "
          f"guaranteed physical scale)")

    width_in = full_img.shape[1] / args.dpi
    height_in = full_img.shape[0] / args.dpi
    fig = plt.figure(figsize=(width_in, height_in), dpi=args.dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(full_img, cmap="gray", vmin=0, vmax=255,
              interpolation="none", aspect="equal")
    ax.axis("off")
    pdf_path = f"{args.out_prefix}.pdf"
    fig.savefig(pdf_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[write] {pdf_path} -- page size {width_in:.3f}in x {height_in:.3f}in "
          f"({width_in * 25.4:.1f}mm x {height_in * 25.4:.1f}mm)")

    print("\n[print] Print the PDF at 100% / \"actual size\" -- NOT \"fit to "
          "page\" or \"shrink to fit\". After printing, measure a square edge "
          "with calipers or a ruler and pass the MEASURED value (not "
          f"{args.square_size_m * 1000:.1f}mm) to calibrate_hand_eye.py's "
          "--square_size_m, and likewise for --marker_size_m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
