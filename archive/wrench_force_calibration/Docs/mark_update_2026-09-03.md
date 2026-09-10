# Draft update to Mark — 2026-09-03

Dear Prof. Mark,

Thank you for the correction and for laying out the rotation convention so
clearly. I went back through the complete projection path rather than trying to
resolve the remaining discrepancy from the Euler notation alone.

The direction forward is now clear. The matrix currently giving the best
agreement in the recordings is an empirical candidate, but the same seven
events cannot both choose the convention and independently validate it. I have
therefore separated that calibration question from the work that can be
completed and tested now. The code retains the candidate for reproducible
diagnostics but rejects it in the production path.

I am pursuing the following:

1. Use the robot signals only to identify every grasp/contact cycle, then
   select role-specific image objects from segmentation proposals using the
   fixed camera view. Ambiguous cases now abstain instead of silently using a
   fitted pixel or an uncertain transform.
2. Propagate every accepted object with box prompts and publish a versioned,
   role-tagged sidecar through one guarded command. The output is checked for
   missing roles, broken paths, caption artefacts, implausible masks, and track
   discontinuities before it can be marked accepted.
3. Freeze the method and evaluate automatic correctness and coverage by
   recording group, followed by the downstream ROS 2/LfD integration check.
   Calibration-based contact projection remains a separate validation path and
   will not block this calibration-free evaluation.

The first evaluation pass has now isolated the remaining failure rather than
leaving it as a rotation-convention question. For contact, the correct object
is present among the image proposals in all 7 evaluated events, but the current
calibration-free evidence separates it confidently in only 2. For grasped
objects, an early-hold attachment test showed that two recordings are sampled
before the object proposal has settled, so changing ranking weights cannot fix
those cases. I have therefore frozen the current result instead of lowering the
confidence bar. The next iteration will require either a re-frozen proposal
timing protocol or independent geometric/data evidence; the uncertain matrix
will remain disabled in production meanwhile.

Best regards,  
Anurag
