# Superseded sidecar builders — replaced by Code/build_sidecar_multi.py

`build_sidecar.py` handled exactly two fixed roles (grasped + contact_receiver)
and composed `propagate_demo.py` + `propagate_cup.py`. The canonical builder
takes any number of `--object obj_id:role:summary_csv:bgr_color` entries, so
single-object and 4-object trials go through one tool.

`build_sidecar_multi.py`'s mask recovery also checks all three BGR channels,
which is stricter and more correct than the 1-2 channel check here.

Kept for backward compatibility with artifacts built before 2026-08-12.
