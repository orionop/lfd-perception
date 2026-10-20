# plan_21oct.md — two tasks on the RTX 4080

Both run on the Ubuntu / RTX 4080 box. Task A is ~5 minutes. Task B is the
one that matters and will probably fail — that is the point of it.

---

## Task A — object naming ("can it tell me it's a carrot")

This is what the 2026-10-20 email to Mark promised, so it comes first.

```bash
cd ~/lfd-perception          # or wherever the box's clone lives
git checkout main
git pull
python Code/run_vlm_object_label.py --backend molmo2
```

- Covers all 12 cases, 5 grasped + 7 contact.
- Self-scores. Acceptable answers are pre-registered inside the script from
  CLAUDE.md's trial table, written before any run — do not edit them after
  seeing replies.
- Expected objects: carrot, cup, plate, screwdriver, charger, Rubik's cube,
  nut/fastener, latch.

Push the result:

```bash
git add figures/vlm_object_label/labels_molmo2.json
git commit -m "Object naming run: Molmo 2 labels on all 12 cases"
git push
```

If a model other than `molmo2` is what's cached on the box, use that and say
which in the commit message. If nothing is cached, the first run downloads
weights.

---

## Task B — the handover gate (this is the real one)

`LAB_DELIVERABLE_A.md` lists three acceptance gates. One is blocked on
recordings. One needs Mark's consumer code, which we do not have. **This is
the only one fully in our hands, and it has never been attempted.**

The test: can someone who is not Anurag go from a clean clone to a valid
sidecar, using only what is written down?

**It is expected to fail.** A fresh clone has none of this — all gitignored:

| missing | size |
|---|---|
| `sam_vit_h_4b8939.pth` | 2.4 GB |
| `sam2.1_hiera_large.pt` | 856 MB |
| `Data/<any trial>` | 532 MB – 1.3 GB |
| `.venv_analysis`, `.venv_sam2` | — |

So the deliverable currently has **no documented path from clone to output**.
That gap is the finding. Record exactly where it stops rather than working
around it from memory.

### Run it

```bash
cd /tmp
rm -rf handover_test && git clone https://github.com/orionop/lfd-perception.git handover_test
cd handover_test
git checkout main
```

Then follow **only** what `README.md` says, and keep a note of every point
where you had to know something it does not tell you.

```bash
# 1. environments — README names three venvs. Can you build them from it alone?
python3 -m venv .venv_analysis
.venv_analysis/bin/pip install numpy pandas matplotlib opencv-python pyyaml

# 2. checkpoints — README gives both URLs
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
python3 Code/download_sam2_ckpt.py        # needs .venv_sam2 first?

# 3. data — THIS IS THE EXPECTED STOPPING POINT.
#    No trial ships with the repo and the README does not say how to obtain one.
ls Data/
```

To get past step 3, copy one trial over from the Mac (smallest first):

```bash
# from the Mac, ~554 MB
rsync -av --progress Data/lfdws_t005 <box>:/tmp/handover_test/Data/
```

Then the actual end-to-end run:

```bash
.venv_analysis/bin/python Code/run_deliverable.py \
    --trial Data/lfdws_t005 --out figures/handover_check --offload-video-to-cpu
echo "exit code: $?"
```

Exit `0` accepted · `2` abstained, review required · `1` execution failure.
**Exit 2 is a pass for this gate** — the gate is "does it run end to end and
behave per contract", not "does selection succeed". Selection failing is a
separate, known, data-blocked problem.

Then confirm the output is contract-valid:

```bash
.venv_analysis/bin/python Code/validate_sidecar.py \
    figures/handover_check/identify/objects.json
.venv_analysis/bin/python Code/sidecar_consumer.py \
    figures/handover_check/identify/objects.json
```

### What to write down

The output of this gate is a list, not a pass/fail:

1. Every step where the README was wrong, incomplete, or assumed knowledge.
2. Every package installed that no document mentions.
3. Where it stopped, with the exact error.
4. Total wall-clock from clone to first output.

That list becomes the fix. A handover gate that passes on the machine that
built the system proves nothing — the value is in the gaps it exposes.

---

## Order

Task A first (5 min, promised to Mark). Task B after, with time to spare —
it will take longer than it looks.

## Rules

- Branch `main`.
- Do not edit the pre-registered acceptable answers in Task A.
- Do not fix README gaps while testing — **record them, fix after.** Fixing
  as you go destroys the measurement.
- Nothing deleted; `/tmp/handover_test` can stay for inspection.
