# Legacy bag extraction — superseded by the lab's ros2_unbag

`bag_to_csv.py` is the lab's original extractor; `unbag_pipeline.py` was a
Python replacement for it. Both are superseded: the lab now runs `ros2_unbag`
plus an in-house merge script and we consume that output.

`Code/mcap_extract.py` is still active and is NOT legacy — it is the fallback
for bags `ros2_unbag` cannot export (anything carrying the ZED
`compressedDepth` topic).
