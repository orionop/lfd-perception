"""
Extract component placements from the lab's camera-bracket assembly STEP file
(ZED-M_assy_body-bracket-gripper_REV02.STEP) to derive the fixed
bota->camera transform for calibration.yaml.

STEP (ISO 10303-21) is plain text. Assembly structure:
  NEXT_ASSEMBLY_USAGE_OCCURRENCE (NAUO)      parent PD -> child PD instance
  CONTEXT_DEPENDENT_SHAPE_REPRESENTATION     links a NAUO to a
    REPRESENTATION_RELATIONSHIP..WITH_TRANSFORMATION whose
    ITEM_DEFINED_TRANSFORMATION references two AXIS2_PLACEMENT_3D frames
    (child frame expressed in parent coordinates).
  AXIS2_PLACEMENT_3D = (CARTESIAN_POINT origin, DIRECTION z, DIRECTION ref_x)

Output: the assembly tree with each component's 4x4 placement composed to the
assembly root, so the SensONE (bota) and ZED-M camera frames can be related.

Usage:
    .venv_analysis/bin/python Code/cad_extract_transform.py \
        --step "<path to .STEP>"
"""
import argparse
import re

import numpy as np


def parse_entities(text):
    """Return {id: (type, [args...])} for all top-level entities.
    Handles multi-line entities by splitting on ';'."""
    # normalize whitespace, split records on ';'
    entities = {}
    # data section only
    m = re.search(r"DATA;(.*?)ENDSEC;", text, re.S)
    body = m.group(1) if m else text
    for record in body.split(";"):
        record = record.strip()
        mm = re.match(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)\s*$", record, re.S)
        if mm:
            eid, etype, args = int(mm.group(1)), mm.group(2), mm.group(3)
            entities[eid] = (etype, args)
            continue
        # complex entity instance: #id = ( TYPE1(...) TYPE2(...) ... )
        mc = re.match(r"#(\d+)\s*=\s*\((.*)\)\s*$", record, re.S)
        if mc:
            eid, args = int(mc.group(1)), mc.group(2)
            entities[eid] = ("COMPLEX", args)
    return entities


def split_args(args):
    """Split a STEP argument list at top-level commas."""
    out, depth, cur, in_str = [], 0, "", False
    for ch in args:
        if ch == "'" :
            in_str = not in_str
            cur += ch
        elif in_str:
            cur += ch
        elif ch == "(":
            depth += 1; cur += ch
        elif ch == ")":
            depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def ref(tok):
    tok = tok.strip()
    return int(tok[1:]) if tok.startswith("#") else None


def get_triplet(entities, eid):
    """CARTESIAN_POINT or DIRECTION -> np.array(3)."""
    _, args = entities[eid]
    m = re.search(r"\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)", args)
    return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])


def a2p3d_to_matrix(entities, eid):
    """AXIS2_PLACEMENT_3D -> 4x4 homogeneous matrix."""
    _, args = entities[eid]
    parts = split_args(args)
    origin = get_triplet(entities, ref(parts[1]))
    z = get_triplet(entities, ref(parts[2])) if ref(parts[2]) else np.array([0, 0, 1.0])
    x_ref = get_triplet(entities, ref(parts[3])) if len(parts) > 3 and ref(parts[3]) else np.array([1.0, 0, 0])
    z = z / np.linalg.norm(z)
    x = x_ref - np.dot(x_ref, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    return T


def product_name_for_pd(entities, pd_id):
    """PRODUCT_DEFINITION -> formation -> PRODUCT -> name."""
    try:
        _, args = entities[pd_id]
        formation_id = ref(split_args(args)[2])
        _, fargs = entities[formation_id]
        product_id = ref(split_args(fargs)[2])
        _, pargs = entities[product_id]
        name = split_args(pargs)[0].strip().strip("'")
        return name
    except Exception:
        return f"PD#{pd_id}"


def build_srr_links(entities):
    """Same-frame representation links: SHAPE_REPRESENTATION_RELATIONSHIP
    (without WITH_TRANSFORMATION — those cross assembly frames and would
    leak other components' geometry into the walk)."""
    links = {}
    for eid, (etype, eargs) in entities.items():
        is_srr = (etype == "SHAPE_REPRESENTATION_RELATIONSHIP" or
                  (etype == "COMPLEX" and "SHAPE_REPRESENTATION_RELATIONSHIP" in eargs))
        if not is_srr or "WITH_TRANSFORMATION" in eargs:
            continue
        refs = [int(t[1:]) for t in re.findall(r"#\d+", eargs)]
        for a in refs:
            for b in refs:
                if a != b:
                    links.setdefault(a, set()).add(b)
    return links


def collect_points_for_pd(entities, pd_id, ref_cache, srr_links=None):
    """BFS from the part's SHAPE_REPRESENTATION(s), collecting every
    CARTESIAN_POINT reachable through entity references. Returns (N,3)."""
    # find SHAPE_DEFINITION_REPRESENTATION whose definition -> this PD
    rep_ids = []
    for eid, (etype, eargs) in entities.items():
        if etype != "SHAPE_DEFINITION_REPRESENTATION":
            continue
        parts = split_args(eargs)
        pds_id = ref(parts[0])
        if pds_id is None or pds_id not in entities:
            continue
        _, pds_args = entities[pds_id]
        if ref(split_args(pds_args)[2]) == pd_id:
            rid = ref(parts[1])
            if rid is not None:
                rep_ids.append(rid)
    pts = []
    seen = set()
    stack = list(rep_ids)
    while stack:
        eid = stack.pop()
        if eid in seen or eid not in entities:
            continue
        seen.add(eid)
        etype, eargs = entities[eid]
        if etype == "CARTESIAN_POINT":
            m = re.search(r"\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)", eargs)
            if m:
                pts.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            continue
        if eid not in ref_cache:
            ref_cache[eid] = [int(t[1:]) for t in re.findall(r"#\d+", eargs)]
        stack.extend(ref_cache[eid])
        if srr_links and eid in srr_links:
            stack.extend(srr_links[eid])
    return np.array(pts) if pts else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--bbox", action="store_true",
                    help="also compute each component's geometry bounding box "
                         "in root coordinates (slower; full reference walk)")
    args = ap.parse_args()

    with open(args.step, errors="replace") as f:
        text = f.read()
    entities = parse_entities(text)
    print(f"[parse] {len(entities)} entities", flush=True)

    # NAUOs: instance id -> (parent_pd, child_pd)
    nauos = {}
    for eid, (etype, eargs) in entities.items():
        if etype == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
            parts = split_args(eargs)
            nauos[eid] = (parts[0].strip().strip("'"), ref(parts[3]), ref(parts[4]))

    # CONTEXT_DEPENDENT_SHAPE_REPRESENTATION -> transformation per NAUO
    # CDSR( representation_relation RRWT, represented_product_relation PDS )
    # PDS references the NAUO; RRWT references ITEM_DEFINED_TRANSFORMATION
    nauo_to_T = {}
    for eid, (etype, eargs) in entities.items():
        if etype != "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION":
            continue
        parts = split_args(eargs)
        rr_id, pds_id = ref(parts[0]), ref(parts[1])
        # PDS -> which NAUO
        _, pds_args = entities[pds_id]
        pds_parts = split_args(pds_args)
        nauo_ref = ref(pds_parts[2])
        # RRWT: (name, desc, rep1, rep2) + WITH_TRANSFORMATION(idt)
        _, rr_args = entities[rr_id]
        idt_match = re.search(r"ITEM_DEFINED_TRANSFORMATION", rr_args)
        idt_id = None
        for tok in re.findall(r"#\d+", rr_args):
            tid = int(tok[1:])
            if entities.get(tid, ("",))[0] == "ITEM_DEFINED_TRANSFORMATION":
                idt_id = tid
        if idt_id is None:
            continue
        _, idt_args = entities[idt_id]
        a2ps = [int(t[1:]) for t in re.findall(r"#\d+", idt_args)
                if entities.get(int(t[1:]), ("",))[0] == "AXIS2_PLACEMENT_3D"]
        if len(a2ps) != 2:
            continue
        # Empirically in this SolidWorks export: the FIRST placement is the
        # child's pose in the parent frame and the SECOND is identity (the
        # child's own origin). child-in-parent = T1 @ inv(T2). Getting this
        # backwards silently passes for ~180-degree placements (near
        # self-inverse) and only breaks on tilted ones -- verified against
        # the drawing's 30/45-degree camera bracket geometry.
        T1 = a2p3d_to_matrix(entities, a2ps[0])
        T2 = a2p3d_to_matrix(entities, a2ps[1])
        nauo_to_T[nauo_ref] = T1 @ np.linalg.inv(T2)

    # build tree, compose to root
    children = {}
    all_children = set()
    for nid, (name, parent_pd, child_pd) in nauos.items():
        children.setdefault(parent_pd, []).append((nid, child_pd))
        all_children.add(child_pd)
    roots = set(p for p, _, in [(v[1], v[2]) for v in nauos.values()]) - all_children

    ref_cache = {}
    srr_links = build_srr_links(entities) if args.bbox else None

    def walk(pd, T_acc, depth):
        pname = product_name_for_pd(entities, pd)
        o = T_acc[:3, 3]
        print(f"{'  '*depth}{pname}: origin_in_root = "
              f"[{o[0]:9.3f}, {o[1]:9.3f}, {o[2]:9.3f}] mm", flush=True)
        Rz = T_acc[:3, 2]
        print(f"{'  '*depth}  z-axis in root = [{Rz[0]:7.4f}, {Rz[1]:7.4f}, {Rz[2]:7.4f}]",
              flush=True)
        if args.bbox:
            pts = collect_points_for_pd(entities, pd, ref_cache, srr_links)
            if len(pts):
                P = (T_acc[:3, :3] @ pts.T).T + T_acc[:3, 3]
                lo, hi = P.min(axis=0), P.max(axis=0)
                c = (lo + hi) / 2
                print(f"{'  '*depth}  bbox_in_root: min=[{lo[0]:8.2f},{lo[1]:8.2f},{lo[2]:8.2f}] "
                      f"max=[{hi[0]:8.2f},{hi[1]:8.2f},{hi[2]:8.2f}] "
                      f"center=[{c[0]:8.2f},{c[1]:8.2f},{c[2]:8.2f}]  ({len(pts)} pts)",
                      flush=True)
        for nid, child_pd in children.get(pd, []):
            T_child = nauo_to_T.get(nid, np.eye(4))
            walk(child_pd, T_acc @ T_child, depth + 1)

    for root_pd in roots:
        print(f"\n[tree] root: {product_name_for_pd(entities, root_pd)}", flush=True)
        walk(root_pd, np.eye(4), 0)


if __name__ == "__main__":
    main()
