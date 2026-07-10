"""
Find candidate lens-glass circular faces in the camera-bracket assembly
STEP file, using pythonocc-core's plain STEPControl_Reader (not XCAF/labels
-- side-steps a SWIG binding mismatch on TDF_Label.FindAttribute in this
pythonocc-core build). STEPControl_Reader resolves all assembly placements
internally and returns one shape already in GLOBAL (assembly-root)
coordinates, so every face we enumerate is already positioned correctly --
no manual transform composition needed.

Enumerates every cylindrical face in the whole assembly, filtered to a
small-radius range (a ZED Mini lens opening is ~5-12mm radius -- much
smaller than the ~19-20mm locating-pin/screw boss already ruled out by
hand in FreeCAD).

Usage (inside the 'occ' conda env):
    conda run -n occ python Code/cad_find_lens_occ.py --step "<path>.STEP"
"""
import argparse

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--max_radius_mm", type=float, default=13.0)
    ap.add_argument("--min_radius_mm", type=float, default=3.0)
    ap.add_argument("--near", default=None,
                    help="x,y,z (mm) -- only report candidates within "
                         "--near_dist of this point (the known camera "
                         "housing origin, to exclude gripper/finger screws)")
    ap.add_argument("--near_dist", type=float, default=100.0)
    args = ap.parse_args()
    near = None
    if args.near:
        near = [float(v) for v in args.near.split(",")]

    print(f"[load] {args.step}", flush=True)
    reader = STEPControl_Reader()
    status = reader.ReadFile(args.step)
    if status != IFSelect_RetDone:
        print("[fatal] STEP read failed", flush=True)
        return
    n = reader.TransferRoots()
    print(f"[load] transferred {n} root shape(s)", flush=True)
    shape = reader.OneShape()

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    print(f"[load] overall assembly bbox: "
          f"x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}] z[{zmin:.1f},{zmax:.1f}] mm",
          flush=True)

    print(f"\n[scan] enumerating cylindrical faces, radius in "
          f"[{args.min_radius_mm}, {args.max_radius_mm}] mm ...", flush=True)

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    candidates = []
    n_faces = 0
    while exp.More():
        n_faces += 1
        face = exp.Current()
        surf = BRepAdaptor_Surface(face, True)
        if surf.GetType() == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            radius = cyl.Radius()
            if args.min_radius_mm <= radius <= args.max_radius_mm:
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                com = props.CentreOfMass()
                axis_loc = cyl.Location()
                axis_dir = cyl.Axis().Direction()
                if near is not None:
                    d = ((com.X()-near[0])**2 + (com.Y()-near[1])**2 +
                         (com.Z()-near[2])**2) ** 0.5
                    if d > args.near_dist:
                        exp.Next()
                        continue
                candidates.append({
                    "radius_mm": radius,
                    "face_centroid": (com.X(), com.Y(), com.Z()),
                    "axis_origin": (axis_loc.X(), axis_loc.Y(), axis_loc.Z()),
                    "axis_dir": (axis_dir.X(), axis_dir.Y(), axis_dir.Z()),
                })
        exp.Next()

    print(f"[scan] {n_faces} total faces in assembly, "
          f"{len(candidates)} cylindrical candidates in radius range", flush=True)

    for i, c in enumerate(candidates):
        print(f"\n  candidate {i+1}: radius={c['radius_mm']:.2f}mm", flush=True)
        print(f"    face_centroid  = {tuple(round(v,2) for v in c['face_centroid'])} mm",
              flush=True)
        print(f"    cylinder_axis_origin = {tuple(round(v,2) for v in c['axis_origin'])} mm",
              flush=True)
        print(f"    cylinder_axis_dir    = {tuple(round(v,3) for v in c['axis_dir'])}",
              flush=True)

    if not candidates:
        print("\n[note] no cylindrical faces matched -- the lens glass may be "
              "modeled as a flat PLANE (a disc) rather than a cylinder wall. "
              "Re-run with a planar-face + small-bounding-circle scan if needed.",
              flush=True)


if __name__ == "__main__":
    main()
