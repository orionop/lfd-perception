"""
Second hypothesis for the ZED-M lens position, after cad_find_lens_occ.py's
cylindrical-bore search (widened to radius 2-20mm, still only finding the
same 88mm-apart bore pair already ruled out as mounting/dowel holes, not
lens barrels -- doesn't match the ZED Mini's 63mm published stereo
baseline). This script tests the alternative hypothesis flagged in that
script's own fallback note: the lens glass may be modeled as a flat
circular PLANAR face (a disc/window) rather than a cylindrical bore.

Enumerates every planar face in the assembly bounded by a single circular
edge (i.e. a true disc, not an annulus or complex polygon), in a plausible
lens-window radius range, and reports pairs whose centroid separation is
close to 63mm (the constraint we're actually trying to satisfy).

Usage (inside the 'occ' conda env):
    conda run -n occ python Code/cad_find_lens_planar.py --step "<path>.STEP"
"""
import argparse
import itertools

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Circle
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--max_radius_mm", type=float, default=15.0)
    ap.add_argument("--min_radius_mm", type=float, default=2.0)
    ap.add_argument("--near", default=None, help="x,y,z (mm)")
    ap.add_argument("--near_dist", type=float, default=120.0)
    ap.add_argument("--target_separation_mm", type=float, default=63.0,
                    help="ZED Mini's published stereo baseline -- pairs "
                         "close to this are flagged")
    ap.add_argument("--separation_tol_mm", type=float, default=15.0)
    args = ap.parse_args()
    near = [float(v) for v in args.near.split(",")] if args.near else None

    print(f"[load] {args.step}", flush=True)
    reader = STEPControl_Reader()
    if reader.ReadFile(args.step) != IFSelect_RetDone:
        print("[fatal] STEP read failed", flush=True)
        return
    reader.TransferRoots()
    shape = reader.OneShape()

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    print(f"[load] assembly bbox: x[{xmin:.1f},{xmax:.1f}] "
          f"y[{ymin:.1f},{ymax:.1f}] z[{zmin:.1f},{zmax:.1f}] mm", flush=True)

    print(f"\n[scan] enumerating planar faces bounded by a single circular "
          f"edge, radius in [{args.min_radius_mm}, {args.max_radius_mm}] mm ...",
          flush=True)

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    candidates = []
    n_faces, n_planar = 0, 0
    while exp.More():
        n_faces += 1
        face = exp.Current()
        surf = BRepAdaptor_Surface(face, True)
        if surf.GetType() == GeomAbs_Plane:
            n_planar += 1
            edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
            edges = []
            while edge_exp.More():
                edges.append(edge_exp.Current())
                edge_exp.Next()
            # a true disc/window has exactly ONE boundary edge, and it's a circle
            if len(edges) == 1:
                curve = BRepAdaptor_Curve(edges[0])
                if curve.GetType() == GeomAbs_Circle:
                    circ = curve.Circle()
                    radius = circ.Radius()
                    if args.min_radius_mm <= radius <= args.max_radius_mm:
                        props = GProp_GProps()
                        brepgprop.SurfaceProperties(face, props)
                        com = props.CentreOfMass()
                        if near is not None:
                            d = ((com.X()-near[0])**2 + (com.Y()-near[1])**2 +
                                 (com.Z()-near[2])**2) ** 0.5
                            if d > args.near_dist:
                                exp.Next()
                                continue
                        loc = circ.Location()
                        normal = circ.Axis().Direction()
                        candidates.append({
                            "radius_mm": radius,
                            "centroid": (com.X(), com.Y(), com.Z()),
                            "circle_center": (loc.X(), loc.Y(), loc.Z()),
                            "normal": (normal.X(), normal.Y(), normal.Z()),
                        })
        exp.Next()

    print(f"[scan] {n_faces} total faces, {n_planar} planar, "
          f"{len(candidates)} single-circular-edge disc candidates in range",
          flush=True)

    for i, c in enumerate(candidates):
        print(f"\n  candidate {i+1}: radius={c['radius_mm']:.2f}mm", flush=True)
        print(f"    centroid = {tuple(round(v,2) for v in c['centroid'])} mm", flush=True)
        print(f"    circle_center = {tuple(round(v,2) for v in c['circle_center'])} mm",
              flush=True)
        print(f"    normal = {tuple(round(v,3) for v in c['normal'])}", flush=True)

    if not candidates:
        print("\n[note] no single-circular-edge planar disc faces found in "
              "range -- the lens window is not modeled as a flat disc "
              "either, at least not distinguishably from other planar "
              "faces at this fidelity.", flush=True)
        return

    print(f"\n[pair-check] looking for pairs separated by "
          f"{args.target_separation_mm}mm +/- {args.separation_tol_mm}mm "
          f"(ZED Mini's published stereo baseline) ...", flush=True)
    found_pair = False
    for (i, a), (j, b) in itertools.combinations(enumerate(candidates), 2):
        ca, cb = a["circle_center"], b["circle_center"]
        dist = ((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2 + (ca[2]-cb[2])**2) ** 0.5
        if abs(dist - args.target_separation_mm) <= args.separation_tol_mm:
            found_pair = True
            print(f"  MATCH: candidate {i+1} <-> candidate {j+1}, "
                  f"separation={dist:.2f}mm", flush=True)
    if not found_pair:
        print("  no pair found within tolerance of the target separation.",
              flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
