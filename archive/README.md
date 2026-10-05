# archive/ — nothing here is deleted

Everything in this directory was real work that produced a real answer. It is
archived because it is **out of scope**, **superseded**, or **a measured
negative result** — not because it is broken or worthless.

The reason for keeping it is practical: a negative result that gets deleted
looks untried, and gets attempted again. Several ideas in this repository were
re-derived more than once before that discipline was adopted.

| Directory | What it is | Why it is here |
|---|---|---|
| `wrench_force_calibration/` | Camera-to-F/T-sensor calibration, hand-eye, wrench-ray grounding, CAD extraction | Out of scope. Not required for vision-based object recognition; the active pipeline has zero dependency on it. Six hypotheses tested and rejected for a 3/7-vs-6/7 scoring gap — full history in its `CALIBRATION_LOG.md`. |
| `manuscript_b_paper/` | Paper draft and publication strategy | Not being pursued. The lab deliverable is the pipeline. |
| `negative_results/` | Image stationarity, cross-trial object identity, DADO proposer, depth ranking | Each ran, each produced a measurable result, each result was negative. See its `README.md` for the numbers. |
| `superseded_selectors/` | `auto_seed.py` and the constant-pixel / early SAM 1 seeding era | Replaced by `Code/select_objects.py` with its abstention contract. `auto_seed` scored 1/6. |
| `superseded_sidecar/` | Fixed two-role sidecar builder and its two propagation scripts | Replaced by `Code/build_sidecar_multi.py`, which handles any object count through one tool. |
| `legacy_extraction/` | `bag_to_csv.py`, `unbag_pipeline.py` | Replaced by the lab's `ros2_unbag` plus their merge script. Note `Code/mcap_extract.py` is **not** legacy — it is the active fallback for bags `ros2_unbag` cannot export. |
| `backup_results_20260830/` | Output snapshot from 2026-08-30 | Historical artifact, kept for provenance. |

## Re-running anything from here

Archived scripts import shared modules (`event_utils`, `select_objects`,
`deliverable_events`) that live in `Code/`. They will need a `sys.path` entry
pointing at `Code/`, or to be run from the repository root with
`PYTHONPATH=Code`.

## Rule

**Do not delete from this repository.** Move things here instead, with a note
saying why. If something here turns out to be needed again, move it back and
say what changed — do not silently resurrect it.
