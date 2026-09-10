# Calibration investigation log (archived)

**Archived 2026-09-10 — moved out of `LAB_DELIVERABLE_A.md` verbatim.** This
was the running record of the hand-eye/wrench-ray calibration attempt while
it was still tracked as part of Deliverable A. It no longer is (see
`../README.md` and `LAB_DELIVERABLE_A.md`'s current "Archived" note) — kept
here as a complete, unedited history of what was tried and found, in case
this work is ever resumed.

---

## First physically-measured T_bota_camera: recovered, scored, rejected (2026-09-09)

Mark sent a real ChArUco hand-eye recording (`lfdws_t002`, 94.9s, our own sent
board: 5×7, `DICT_5X5_100`, 35mm/26mm nominal, exact-scale A4 PDF he printed at
100% — no re-measurement needed since we control the print source). Merged via
`Code/mcap_extract.py` (no synced master-topic CSV was provided) into
`Data/charuco_calib_002`, then `Code/calibrate_hand_eye.py solve`:

- board detected in 81,853/93,888 rows (87%);
- 162 independent poses kept after near-duplicate filtering (script minimum
  is 3, recommended 10-15 — 162 is a large, healthy set);
- board-in-base-frame position residual std-dev across those 162 poses:
  [3.7, 5.1, 3.2] mm — well inside the script's own >10-20mm "bad detection"
  flag;
- debug overlay (6 sample frames) visually confirmed correct board-axis
  detection in every sample.

This is real, internally self-consistent evidence — the first physically-
measured `T_bota_camera` candidate that isn't a failed CAD guess. It is
**not** a validated result: internal pose-agreement proves the rig moved
rigidly and the solver is self-consistent, not that the recovered transform
is metrically correct against physical reality.

Scored against the same 7-event wrench-ray harness every prior candidate
used (`Code/wrench_ray_validate.py --raw_R ... --raw_t_mm ...`, values taken
directly from `calibration_handeye_result.yaml`, nothing refit):

**3/7 hits — worse than the currently-adopted CAD-derived candidate's 6/7.**
Misses: `lfdws_t001/press`, `lfdws_t001_labexport/press`,
`lfdws_t001_depth/charger_grasp` (ray entirely outside frame, 0 pixels),
`lfdws_t001_depth/charger_lift`. Hits: `lfdws_t001_depth/plate_press`,
`screwdriver_contact`, `charger_dock`.

**Verdict: REJECTED at this transform. Not written to `calibration.yaml`;
`bota_to_camera.filled` stays `false`.** `Code/calibrate_hand_eye.py` never
writes `calibration.yaml` automatically and the printed
"REVIEW before pasting" instruction was followed.

Most likely explanation, tying together the good internal residual and the
bad external score: this calibration is only as correct as the assumption
that `current_pose`'s origin coincides with the Bota sensor's actual
force-measurement origin. AX=XB fitting is blind to a *constant* rigid offset
between the two — it will still converge tightly (explaining the good
residual) while silently absorbing that offset into `T_bota_camera`, which
then miscarries the wrench line of action by exactly that offset (explaining
the bad wrench-ray score, since the wrench itself is expressed in the true
Bota frame per `wrench_ray.py`'s header comment). This is the same open
question from Vlutters' 2026-08-28 email — whether `current_pose` reports the
Bota sensor origin or the Franka tool frame at the sensor flange — now with
concrete evidence that getting it wrong costs 3 of 7 real contact events, not
just a documentation nicety.

Also found while running this: `calibrate_hand_eye.py`'s console line
`"[result] T_bota_camera (bota origin frame -> camera-frame point)"` describes
the mapping direction backwards — the printed matrix is `R_cam2gripper`/
`t_cam2gripper` straight from `cv2.calibrateHandEye`, i.e. it maps a
camera-frame point INTO the bota frame, which is exactly what the script's own
residual check and `wrench_ray.py`'s `T_base_camera = T_base_bota @
T_bota_camera` composition both correctly assume. The matrix values and every
downstream consumer are correct; only that one print statement's English is
backwards. Not yet fixed — flagging here so it isn't mistaken for a data bug.

Next step is not re-running this calibration hoping for a different number —
the solve is already tight. It's resolving the `current_pose` frame question
directly (Vlutters' pending confirmation), or re-deriving the calibration with
an explicit free translational offset between `current_pose` and the true
Bota origin as an additional unknown, rather than assuming they coincide.

## The frame-offset hypothesis, tested directly: rejected (2026-09-10)

Rather than wait on Vlutters' confirmation, the offset hypothesis above was
tested using only evidence already in hand: `Code/handeye_offset_search.py`
holds the hand-eye rotation fixed (a rigid translational offset between two
points on the same end-effector stack cannot change recovered orientation)
and grid-searches a bounded translation correction on top of it, scored
against the same 7-event wrench-ray harness as every other candidate. The
bound (±50mm) is physically motivated, not arbitrary: the Bota SensONE's own
datasheet (`bota_sensone_dimensions.png`) gives 38.0mm as the sensor body's
total robot-mounting-to-tool-mounting thickness, so any offset between
`current_pose`'s claimed origin (Vlutters: the tool-mounting face) and the
sensor's true F/T coordinate origin is bounded by that dimension.

**Result: rejected, not confirmed.** The in-fold best (4/7, 0.571) already
sits at the grid's dz=-50mm edge, meaning even ±50mm isn't large enough to
reach an actual optimum in that direction — a bad sign on its own. Worse,
**966 of the 9,261 candidates (10.4% of the entire grid) tie at that same
best hit rate**, spanning the full ±50mm range on every axis: this is a
broad plateau, not a peak, meaning these 7 events cannot pin the offset down
to any value, physically plausible or not. Leave-one-recording-out
cross-validation confirms this isn't just an ambiguous-but-real signal: every
held-out fold scores 0/1 or 0/5 (one fold reaches a perfect in-fold 1.000 on
2/2 training events and then 0/5 on the 5 held out) — zero generalization in
any fold.

**Conclusion:** a simple constant translational offset between `current_pose`
and the true Bota origin does not explain why the 2026-09-09 hand-eye
candidate scored 3/7 against the adopted candidate's 6/7. The frame question
may still matter (Vlutters' confirmation is still open and still worth
having), but it is evidently not a small, physically-bounded translation-only
correction on top of an otherwise-correct rotation — which redirects the
likely root cause back toward the ROTATION itself, and specifically toward
the motion/sync issue already flagged: the calibration recording was
continuous motion throughout (median arm speed 1cm/s, only 11% of samples
under 1mm/s), not the static pauses `calibrate_hand_eye.py`'s protocol
assumes, and unlike a fixed origin offset, motion-induced pose/detection
error corrupts rotation too. `--max_speed_mps` (added 2026-09-09, still
untested against the wrench-ray score) is the next concrete thing to actually
try, not another offset search on this same rotation.

## The motion-blur hypothesis, tested directly: also rejected (2026-09-10)

`calibrate_hand_eye.py --max_speed_mps 0.001` re-run on the same
`charuco_calib_002` recording, restricted to instants with current_pose speed
under 1mm/s: 10,500/93,888 rows qualified, board detected in 10,498, **17**
independent poses survived dedup (well above the 3-minimum, near the
recommended 10-15), residual std-dev [3.2, 5.6, 1.4]mm — as tight as the
unfiltered solve.

**Result: the recovered transform barely moved** (rotation differs only in
the 3rd decimal, translation shifts 1-5mm from the 2026-09-09 all-frames
result) **and `wrench_ray_validate.py` scores it identically: 3/7, the exact
same three events hit and four missed.** Motion blur during capture is not
the explanation for the 3/7-vs-6/7 gap either.

Before concluding anything further, the scripts themselves were re-audited
(prompted by a hunch, not a specific suspicion) while this run was in
flight: `speed[i]`'s indexing against `df.iterrows()` was verified correct
via direct checks on the merged CSV (timestamps strictly monotonic, zero
duplicate/negative steps; DataFrame index is a plain `0..N-1` RangeIndex),
and `calibrate_hand_eye.py`'s `quat_to_R` was diffed against
`wrench_ray.py`'s independent copy — mathematically identical (only
whitespace differs), same ROS (x,y,z,w) convention, same argument order in
both call sites. No bug found in either script.

**Two well-tested hypotheses are now rejected** (frame-origin offset,
2026-09-10 morning; motion blur, this section) **with no bug found in the
scripts that tested them.** The real hand-eye calibration still cannot beat
the untrusted, unvalidated CAD candidate's 6/7 on this test. Two honest
possibilities remain open, neither yet tested: (1) a real, still-unidentified
error in the hand-eye recovery itself (untested: camera intrinsics, ZED
left/right lens or rectification convention, board planarity/detection
bias), or (2) the 7-event wrench-ray test is too thin to reliably distinguish
a correct transform from a lucky one — this file has said as much about the
*adopted* candidate before (`mark_sheet_azimuth.py`'s 50-of-360-degree
plateau), and the same caveat now cuts against trusting either candidate's
score alone. Do not re-run this calibration again expecting a different
number from the same recording; the next real move is either a different
kind of test of the two current candidates (not another candidate search),
or a second independent calibration recording to check repeatability.

## The current_pose frame question is now RESOLVED (2026-09-10)

Vlutters confirmed it directly, from Franka Desk's End Effector panel (not a
new recording, not new work): the active profile ("Bota + Hand + Bracket_r2 +
ZED") configures Flange→TCP as `(x=0, y=0, z=0.035)` m with zero rotation,
and "the TCP is placed 35 mm lower than the flange, such that it coincides
with the measurement frame of the force sensor." He further confirmed
`franka_robot_state_broadcaster/current_pose` publishes `O_T_EE` — the TCP
pose in the base frame.

**This means `current_pose` already reports the Bota SensONE's own F/T
measurement frame directly** — position and (since the configured rotation
offset is zero) orientation both. There is no separate current_pose-vs-Bota
correction to apply. `calibration.yaml`'s `end_effector` block is updated:
`current_pose_is: bota_origin`, `bota_to_tcp: [0, 0, 0]` (previously `null`).
**This fact is real and stays recorded in the live `calibration.yaml`** even
though the rest of this investigation is archived.

This retroactively explains, not just resolves, `Code/handeye_offset_search.py`'s
result above: no bounded translational offset improved or generalized the
2026-09-09 hand-eye candidate's 3/7 score **because there was never a
frame-origin bug for an offset to fix.** The true offset is exactly zero, by
design. The 3/7-vs-6/7 gap therefore has a different cause. The leading
remaining, and now essentially the only remaining, hypothesis is that the
calibration recording's continuous motion (median 1cm/s, only 11% of samples
under 1mm/s) corrupted the recovered ROTATION during capture — an
origin-offset search cannot detect or fix a rotation error.

## Six more things checked (2026-09-10, prompted by a direct "the code is wrong" push): five ruled out, one real but insufficient

- **`--max_speed_mps` result** (the "next concrete step" above): done. 17
  static-only poses, residual 3.2/5.6/1.4mm, but `wrench_ray_validate.py`
  scores it identically 3/7, same events hit and missed as the unfiltered
  run. Motion blur rejected (see the section above this one).
- **ChArUco `legacyPattern` bug** (a real, documented OpenCV backward-
  compatibility break for even-row-count boards): tested directly on our
  exact 5×7 board — `legacyPattern=True` vs `False` produce a **pixel-
  identical image (0/350,000 differing pixels)**. Our board has two odd
  dimensions; this bug specifically requires an even row count. Not
  applicable, ruled out.
- **`CharucoBoard` constructor argument order** (squareLength vs
  markerLength swap is a classic mistake): verified via keyword arguments
  against the real OpenCV 4.13.0 API — our positional call matches exactly
  (`squareLength=0.035 > markerLength=0.026`, correct). Ruled out.
- **`cv2.calibrateHandEye` solver choice**: this script always used
  `CALIB_HAND_EYE_TSAI`, and TSAI has documented reliability issues on some
  inputs. Added `--method` and now compute all 5 OpenCV hand-eye solvers
  (TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS) from the same detected poses every
  run. **All 5 agree tightly** (max 4.1mm translation, 0.40° rotation apart).
  Independent algorithms converging together is evidence against a
  method-specific bug, not for one. Ruled out.
- **Board print scale** (never independently verified — only the source PDF
  was checked exact, not what actually came off Mark's printer):
  `Code/verify_board_scale.py` cross-checks solvePnP's assumed-35mm depth
  against the ZED's own independently-measured stereo depth at the same
  pixel, across 15 distinct static frames (dedup by image id — the first
  version of this check accidentally measured the same one frame 15 times
  and was fixed before trusting it). **Real signal: ZED depth is
  consistently ~3-5% shorter than solvePnP assumes** (13/15 frames clustered
  ratio 0.95-0.98; 2 early-transient frames read 1.12, likely depth
  settling, not the main effect), implying a true square size around 34mm,
  not 35mm. **This is real but too small to be the explanation**: a 3-5%
  scale error propagates to roughly a 4-6mm translation shift, and the
  `handeye_offset_search.py` ±50mm search already explored a translation
  neighborhood 10x larger than that around this exact transform and found
  nothing beating 4/7 in-fold. Recorded as a real, worth-fixing-eventually
  finding, not as the answer.

**Camera intrinsics, tested directly (same day):** `calibration.yaml`
assumes `dist: [0,0,0,0,0]`, never independently re-verified. New
`Code/verify_camera_intrinsics.py` runs a fresh `cv2.calibrateCamera()` from
the same 17 diverse frames (via the identical detection + diversity filter
as the hand-eye solve) and compares against the lab-provided K.
**Reprojection RMS 0.27px (excellent fit). fx/fy within 0.3% of the lab
values (negligible) but distortion is NOT actually zero:**
`[0.0112, -0.0440, -0.0003, -0.0033, 0.0380]`, plus principal point off by
~5.5px in x. Re-ran `calibrate_hand_eye.py --dist_override` (new flag) with
these corrected coefficients: **the resulting transform shifted by only
~1mm/negligible rotation, and `wrench_ray_validate.py` scores it identically
— 3/7, the exact same three events hit and four missed.** Distortion
correction rejected as the explanation too.

**Running tally: 6 hypotheses tested and rejected with direct evidence**
(frame offset, motion blur, ChArUco pattern bug, constructor arg order,
solver method, camera distortion), **2 real-but-insufficient findings**
(board scale ~3-5% off, distortion nonzero but small), **0 bugs found in
our own scripts under direct, repeated audit.**

The convergence itself is now the finding: six independent corrections —
some testing real, measured discrepancies (distortion, board scale), others
testing hypothetical bugs (frame offset, motion, solver, pattern) — all
perturb `T_bota_camera` by only a few mm / a fraction of a degree, and every
single one lands on the identical 3/7 with the identical three events hit.
That is not the signature of an undiscovered code bug waiting to be found;
a real bug of the size needed to flip 3/7 to 6/7 would show some sensitivity
to at least one of these six corrections, and none showed any. It is much
more consistent with `T_bota_camera` itself being close to correct, and the
discrepancy being about which of the two candidates (the untrusted,
never-validated CAD guess or this real, tightly-self-consistent measurement)
the thin 7-event wrench-ray test can actually be trusted to discriminate —
this file has already shown a comparably-scored CAD sweep can sit on a
50-of-360-degree plateau (`mark_sheet_azimuth.py`), meaning the CAD
candidate's 6/7 was never proven to reflect a more correct transform, only a
better-scoring one on a thin test.

**This is where the investigation stopped, by explicit decision, not because
it ran out of ideas.** Six rejections with a stable, convergent answer would
have been the natural stop condition even if the deliverable's scope hadn't
also been clarified to not need this work at all (see `../README.md`). Had
it continued, the next step would have been a second calibration recording
(repeatability) or more contact events (statistical power) — not more code
archaeology.
