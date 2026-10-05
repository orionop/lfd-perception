# Superseded selectors — replaced by Code/select_objects.py

Earlier approaches to choosing which object to segment. All replaced by the
role-aware proposal selector with the safe-abstention contract.

- `auto_seed.py` — vision-only SAM automatic-mask seed picker. Scored **1/6**
  against constant-pixel seeders' 5/6 (`seed_scoreboard.py`, 2026-08-29). Its
  role priors are tuned to lfdws_t001's object scale and reject a correct
  contact receiver larger than 40% of the frame. Never an acceptable substitute
  for `select_objects.py` in a production run.
- `auto_seed_depth_prior.py` — tested whether real per-pixel depth improves
  auto_seed's scoring. Did not rescue it.
- `grasped_seed_pixel.py` — the grasped seed as a constant image pixel, valid
  because the arm pose cancels algebraically on an eye-in-hand rig. 95.5%/96.4%
  of hold frames on the two carried-object recordings, but `held_out_validated:
  false` — only two independent carried-object demos exist, so the number is
  in-sample.
- `seed_scoreboard.py` — the scored table that produced the 1/6 vs 5/6 result.
  Read it asymmetrically: the constant pixels were fitted on exactly those
  tracks, so 5/6 is in-sample and is NOT a validated replacement.
- `seed_e2e_check.py`, `segment_events.py` — early SAM 1 point-prompt work.
- `identify_objects.py` — intended one-shot end-to-end; OOMs on M3 Pro (18 GB)
  when several SAM 2 objects share one model state. The split
  propagate-per-object pattern in `Code/` is the working form.
