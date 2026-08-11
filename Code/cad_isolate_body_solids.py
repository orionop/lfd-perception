"""
Corrects a mistaken conclusion mid-session: the ZED-M_body_REV01_GJN
part's own SHAPE_REPRESENTATION entity (#39528 in the assembly STEP, see
grep by name) has only an axis placement as its item, which looked like
"no geometry" -- but it's linked via a SHAPE_REPRESENTATION_RELATIONSHIP
(#50962 in the file this was built against) to a SEPARATE
ADVANCED_BREP_SHAPE_REPRESENTATION (#1822) containing two real
MANIFOLD_SOLID_BREP solids ('Fillet4', 'Cut-Extrude7') in the SAME
representation context. The body is NOT geometry-free; it's a simplified
2-solid placeholder block. This script isolates just those two solids
(by bounding-box proximity to the already-confirmed housing origin, since
pythonocc-core's plain STEPControl_Reader doesn't expose per-solid names
-- the XCAF label API needed for that has a SWIG binding mismatch in this
build, see Code/cad_find_lens_occ.py's docstring) and searches THEIR
faces specifically, instead of the whole assembly at once -- a much
cleaner signal, not diluted by the bracket's many mounting-hole faces.

Usage (inside the 'occ' conda env):
    conda run -n occ python Code/cad_isolate_body_solids.py --step "<path>.STEP" \
        --housing_origin="-8.70,-198.463,-25.889" --housing_size_mm=110
"""
import argparse

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def bbox_of(shape):
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    return bbox.Get()  # xmin,ymin,zmin,xmax,ymax,zmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--housing_origin", required=True, help="x,y,z mm")
    ap.add_argument("--housing_size_mm", type=float, default=115.0,
                    help="max plausible bbox diagonal for the ZED-M body "
                         "solids (real ZED Mini is ~103x30x30mm)")
    args = ap.parse_args()
    origin = [float(v) for v in args.housing_origin.split(",")]

    print(f"[load] {args.step}", flush=True)
    reader = STEPControl_Reader()
    if reader.ReadFile(args.step) != IFSelect_RetDone:
        print("[fatal] STEP read failed", flush=True)
        return
    reader.TransferRoots()
    shape = reader.OneShape()

    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []
    while exp.More():
        s = exp.Current()
        xmin, ymin, zmin, xmax, ymax, zmax = bbox_of(s)
        cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
        diag = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2) ** 0.5
        dist_to_housing = ((cx-origin[0])**2 + (cy-origin[1])**2 + (cz-origin[2])**2) ** 0.5
        solids.append({
            "shape": s, "centroid": (cx, cy, cz), "bbox_diag": diag,
            "dist_to_housing": dist_to_housing,
            "size": (xmax-xmin, ymax-ymin, zmax-zmin),
        })
        exp.Next()

    print(f"[scan] {len(solids)} total solids in assembly", flush=True)
    print(f"\n[filter] solids within {args.housing_size_mm}mm of housing "
          f"origin {tuple(origin)}, sorted by distance:", flush=True)
    solids.sort(key=lambda d: d["dist_to_housing"])
    for i, s in enumerate(solids):
        if s["dist_to_housing"] > args.housing_size_mm * 2:
            continue
        print(f"  solid {i}: centroid={tuple(round(v,1) for v in s['centroid'])} "
              f"size(xyz)={tuple(round(v,1) for v in s['size'])}mm "
              f"bbox_diag={s['bbox_diag']:.1f}mm "
              f"dist_to_housing={s['dist_to_housing']:.1f}mm", flush=True)

    # The real ZED Mini body is roughly 103 x 29 x 33mm -- look for solids
    # in that size ballpark near the housing origin specifically (not the
    # much larger bracket or gripper solids)
    candidates = [s for s in solids
                  if s["dist_to_housing"] < args.housing_size_mm
                  and 60 < s["bbox_diag"] < args.housing_size_mm * 1.3]
    print(f"\n[candidates] {len(candidates)} solid(s) plausible as the "
          f"camera body (size + proximity match)", flush=True)

    for i, s in enumerate(candidates):
        print(f"\n--- candidate solid {i}: size={tuple(round(v,1) for v in s['size'])}mm "
              f"centroid={tuple(round(v,1) for v in s['centroid'])} ---", flush=True)
        cyl_exp = TopExp_Explorer(s["shape"], TopAbs_FACE)
        n_faces, n_cyl, n_plane = 0, 0, 0
        while cyl_exp.More():
            n_faces += 1
            face = cyl_exp.Current()
            surf = BRepAdaptor_Surface(face, True)
            t = surf.GetType()
            if t == GeomAbs_Cylinder:
                n_cyl += 1
                cyl = surf.Cylinder()
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                com = props.CentreOfMass()
                print(f"    [cyl face] radius={cyl.Radius():.2f}mm "
                      f"centroid={(round(com.X(),1), round(com.Y(),1), round(com.Z(),1))}",
                      flush=True)
            elif t == GeomAbs_Plane:
                n_plane += 1
            cyl_exp.Next()
        print(f"    total faces={n_faces}  cylindrical={n_cyl}  planar={n_plane}",
              flush=True)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
