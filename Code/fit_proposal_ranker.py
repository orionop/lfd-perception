"""Fit the grasped-role proposal ranker instead of hand-setting its weights.

WHAT THIS REPLACES
------------------
Code/select_objects.py scores proposals with constants chosen by hand:

    score = 0.35*overlap + 0.15*proximity + 0.20*stationarity
          + 0.10*novelty + 0.15*sam_quality + 0.05*border

Those six numbers were never fitted, because the only labelled data was twelve
cycles. Code/harvest_proposal_dataset.py lifts that to thousands of labelled
proposal instances by reusing the already-propagated reference tracks, so the
weights can be estimated rather than guessed.

PRE-REGISTERED SUCCESS THRESHOLD (fixed before the first run)
-------------------------------------------------------------
The frozen baseline is grasped 0/5 cycles accepted, 4 independent groups.

    PASS    >= 3/5 cycles correct under leave-one-GROUP-out, i.e. every
            prediction made on a recording group whose data was not fitted on.
    FAIL    anything less.
    VOID    an improvement that appears only in the all-data fit and not in
            the held-out columns. That is the overfitting signature already
            seen in the 6,561-rule (2/7 all-data, 1/2 held-out) and
            19,683-rule (3/7 all-data, 0.40 held-out precision) searches, and
            it does not count as evidence however large it looks.

Baselines are recomputed on the same rows so an improvement cannot be the
proposal pool doing the work:

    random        expected accuracy of a uniform pick  = mean(1/n_proposals)
    largest       pick the biggest proposal
    central       pick the proposal nearest the image centre
    sam_quality   pick by SAM's own predicted_iou
    region_rule   region_overlap alone, the one feature diagnosed as carrying
                  nearly all the discrimination in the frozen ranker
    oracle        best achievable from this pool, the ceiling

Logistic regression with L2 is used deliberately: few parameters, no tuning
knobs to search, and a linear model keeps the result interpretable against the
hand-set weights it replaces. With 4 groups, anything heavier would be fitting
noise again.

Usage:
    .venv_analysis/bin/python Code/fit_proposal_ranker.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

DATA = "figures/proposal_dataset/grasped.csv"
OUT_DIR = "figures/proposal_dataset"
OUT_REPORT = os.path.join(OUT_DIR, "ranker_fit_report.txt")

DROP = {"recording", "group", "img_id", "proposal_idx", "gt_iou", "label"}
L2 = 1.0
ITERS = 4000
LR = 0.5
SEED = 0


def log(lines, msg):
    print(msg, flush=True)
    lines.append(msg)


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    feats = [c for c in rows[0] if c not in DROP]
    X = np.array([[float(r[c]) for c in feats] for r in rows], float)
    y = np.array([int(r["label"]) for r in rows], int)
    g = np.array([r["group"] for r in rows])
    key = np.array([f"{r['recording']}|{r['img_id']}" for r in rows])
    rec = np.array([r["recording"] for r in rows])
    return X, y, g, key, rec, feats


def fit_logreg(X, y, l2=L2, iters=ITERS, lr=LR):
    """Plain L2 logistic regression, numpy only (no new dependency)."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    # Class weighting: positives are ~1% of rows, so an unweighted fit would
    # score well by predicting "never correct".
    pos = max(1, int(y.sum()))
    sw = np.where(y == 1, len(y) / (2.0 * pos), len(y) / (2.0 * max(1, len(y) - pos)))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (sw * (p - y)) / len(y)
        grad[:-1] += l2 * w[:-1] / len(y)
        w -= lr * grad
    return w, mu, sd


def score_with(w, mu, sd, X):
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    return Z @ w


def top1_by_frame(scores, y, key):
    """Fraction of frames whose highest-scoring proposal is a correct one."""
    ok = tot = 0
    for k in np.unique(key):
        m = key == k
        if not y[m].any():
            continue          # frame has no correct proposal in pool; undecidable
        tot += 1
        ok += int(y[m][np.argmax(scores[m])] == 1)
    return ok, tot


def cycle_table(scores, y, key, rec):
    """Per-recording decision: a recording counts correct if the frame-level
    majority of its decisions are correct. Recordings, not frames, are what the
    deliverable's 0/5 is counted in."""
    out = {}
    for r in np.unique(rec):
        rm = rec == r
        ok, tot = top1_by_frame(scores[rm], y[rm], key[rm])
        out[r] = (ok, tot)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA, help="harvested dataset CSV")
    ap.add_argument("--out", default=OUT_REPORT, help="report path")
    a = ap.parse_args()
    data_path, out_report = a.data, a.out
    if not os.path.exists(data_path):
        raise SystemExit(f"missing {data_path}; run Code/harvest_proposal_dataset.py first")
    X, y, g, key, rec, feats = load(data_path)
    groups = sorted(set(g))
    lines = []

    log(lines, f"rows={len(X)}  features={len(feats)}  positives={int(y.sum())} "
               f"({100.0*y.mean():.2f}%)  groups={len(groups)} {groups}")
    log(lines, "")

    # ---------- baselines, same rows ----------
    def col(name):
        return X[:, feats.index(name)]

    rng = np.random.default_rng(SEED)
    base = {
        "largest":     col("area_fraction"),
        "central":     -col("dist_center"),
        "sam_quality": col("predicted_iou") + col("stability"),
        "region_rule": col("region_overlap"),
        "random":      rng.random(len(X)),
    }
    log(lines, "BASELINES (all rows, no fitting)")
    for name, s in base.items():
        ok, tot = top1_by_frame(s, y, key)
        log(lines, f"  {name:<12} top-1 frames {ok}/{tot} = {100.0*ok/max(1,tot):5.1f}%")
    ok, tot = top1_by_frame(y.astype(float), y, key)
    log(lines, f"  {'oracle':<12} top-1 frames {ok}/{tot} = {100.0*ok/max(1,tot):5.1f}%  (ceiling)")
    log(lines, "")

    # ---------- leave-one-group-out ----------
    log(lines, "LEAVE-ONE-GROUP-OUT (the number that counts)")
    held_scores = np.zeros(len(X))
    for gid in groups:
        te = g == gid
        tr = ~te
        if y[tr].sum() == 0 or y[te].sum() == 0:
            log(lines, f"  {gid:<16} SKIPPED (no positives on one side)")
            continue
        w, mu, sd = fit_logreg(X[tr], y[tr])
        s = score_with(w, mu, sd, X[te])
        held_scores[te] = s
        ok, tot = top1_by_frame(s, y[te], key[te])
        log(lines, f"  {gid:<16} top-1 frames {ok}/{tot} = "
                   f"{100.0*ok/max(1,tot):5.1f}%")
    ok_h, tot_h = top1_by_frame(held_scores, y, key)
    log(lines, f"  {'POOLED HELD-OUT':<16} top-1 frames {ok_h}/{tot_h} = "
               f"{100.0*ok_h/max(1,tot_h):5.1f}%")
    log(lines, "")

    log(lines, "PER-RECORDING, HELD-OUT (deliverable counts recordings, not frames)")
    tab = cycle_table(held_scores, y, key, rec)
    rec_ok = 0
    for r in sorted(tab):
        ok, tot = tab[r]
        passed = tot > 0 and ok > tot / 2.0
        rec_ok += int(passed)
        log(lines, f"  {r:<24} {ok}/{tot} frames correct  -> "
                   f"{'CORRECT' if passed else 'wrong'}")
    log(lines, f"  recordings correct (held-out): {rec_ok}/{len(tab)}")
    log(lines, "")

    # ---------- all-data fit, reported ONLY to expose overfitting ----------
    w, mu, sd = fit_logreg(X, y)
    ok_a, tot_a = top1_by_frame(score_with(w, mu, sd, X), y, key)
    log(lines, "ALL-DATA FIT (diagnostic only - never evidence)")
    log(lines, f"  top-1 frames {ok_a}/{tot_a} = {100.0*ok_a/max(1,tot_a):5.1f}%")
    gap = (ok_a / max(1, tot_a)) - (ok_h / max(1, tot_h))
    log(lines, f"  all-data minus held-out gap: {100.0*gap:+.1f} points"
               f"{'   <-- OVERFITTING SIGNATURE' if gap > 0.15 else ''}")
    log(lines, "")

    log(lines, "FITTED WEIGHTS (standardised; sign and magnitude are comparable)")
    for name, coef in sorted(zip(feats, w[:-1]), key=lambda t: -abs(t[1])):
        log(lines, f"  {name:<20} {coef:+.3f}")
    log(lines, "")

    # ---------- verdict against the pre-registered threshold ----------
    log(lines, "VERDICT against the pre-registered threshold")
    log(lines, "  baseline to beat: grasped 0/5 cycles, 4 groups")
    if gap > 0.15 and rec_ok < 3:
        verdict = "VOID - improvement is all-data only, held-out did not move"
    elif rec_ok >= 3:
        verdict = f"PASS - {rec_ok}/{len(tab)} recordings correct held-out"
    else:
        verdict = f"FAIL - {rec_ok}/{len(tab)} recordings correct held-out, need >=3"
    log(lines, f"  {verdict}")
    log(lines, "")
    log(lines, "Standing caveat: 4 independent groups. A pass here means the "
               "approach is worth more data, NOT that it generalises.")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
