# plan_5oct.md — VLM grasped-object selection on the RTX 4080

**For: a Claude session on the Ubuntu / RTX 4080 box.**
Written 2026-10-05 on the Mac. Everything you need arrives via `git pull`.
No manual file transfer, no scp, no USB stick.

---

## 0. Read this first — what the job is and is not

**The job:** run one or more open-weights VLMs on 5 images. Ask each model to
point at the object the robot's gripper is holding. Map that point to a cached
candidate mask. Write the predictions to JSON. Commit. Push. Stop.

**The job is NOT:** scoring, tuning, prompt engineering, or deciding whether
the result is good. You have no ground truth on that box and that is
deliberate. Scoring happens back on the Mac.

**Why the split.** The whole value of this experiment is that a zero-shot model
has nothing fitted to our data, so a good result cannot be us fooling
ourselves. Five earlier attempts failed exactly by fitting to 5 recordings
(a 6,561-rule search, a 19,683-rule search, an attachment gate, a 17-feature
logistic fit, two region rules — all VOID or FAIL). If anyone looks at the
answers and then adjusts the prompt, this becomes attempt number six and is
worth nothing. The prompt is frozen in the script. Leave it alone.

---

## 1. Background in four lines

- The pipeline detects grasp/contact events, cuts the frame into ~60–280
  candidate object masks, then must pick the right one. Picking is the only
  broken stage.
- Current best honest result: **2 of 5 recordings**, from a single hand-written
  rule. The pre-registered bar is **≥3/5**.
- Overnight 2026-10-05 the candidate pool was improved (right answer present in
  30/41 frames, up from 27/41) — but picking did not improve.
- HOI-DETR and DistinctNet were already tested on this box and both failed.
  A VLM is a different bet: those failed from a human-hand morphology gap and a
  static-camera assumption respectively, neither of which applies here.

---

## 2. Get the code and data

```bash
git clone https://github.com/orionop/lfd-perception.git   # if not already there
cd lfd-perception
git checkout interaction-bakeoff-prep              # NOT main
git pull
```

Verify the inputs arrived (all are tracked in git — this must print 5 and 12):

```bash
python3 -c "
import json; b=json.load(open('figures/interaction_bakeoff/input/benchmark.json'))
g=[c for c in b['cases'] if c['role']=='grasped']
print('grasped cases:', len(g))
for c in g: print(' ', c['case_id'], '|', c['independent_group'])
"
ls figures/interaction_bakeoff/input/images/ | wc -l      # 12
ls figures/interaction_bakeoff/input/proposals/ | wc -l   # 12
```

Expected 5 grasped cases:

| case_id | independent group |
|---|---|
| `lfdws_t001__grasped__c1` | t001_original |
| `lfdws_t002_new__grasped__c1` | t002_cube |
| `lfdws_t002_labexport__grasped__c1` | t002_cube |
| `lfdws_t004__grasped__c2` | t004 |
| `lfdws_t005__grasped__c1` | t005 |

**There are no reference masks in that bundle.** It is stamped
`inference_only_no_reference_masks`. That is correct, not a missing file.

---

## 3. Environment

```bash
python3 -m venv .venv_vlm
source .venv_vlm/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pillow opencv-python numpy pyyaml einops
nvidia-smi    # confirm the 4080 is visible
```

RTX 4080 = 16 GB. A 7–8B VLM fits in fp16/bf16. If you hit OOM, add
`bitsandbytes` and load in 4-bit (≈5–6 GB) — that is a memory workaround, not a
change to the experiment, so it is fine.

---

## 4. Run the models

The runner is already written: `Code/run_vlm_grasped_bakeoff.py`.
Read its docstring before running. Do not edit `PROMPT`.

### Method A — Molmo 2 (start here)

Pointing-native, strongest pointing/grounding benchmarks, open weights.

```bash
python Code/run_vlm_grasped_bakeoff.py --backend molmo2
```

### Method B — MolmoAct 2 (robotics-tuned, best match on paper)

Built on Molmo 2, evaluated on a real arm **with a wrist camera** — which is
our exact camera geometry.

```bash
python Code/run_vlm_grasped_bakeoff.py --backend molmoact2
```

### Method C — Qwen2.5-VL (easiest fallback)

```bash
python Code/run_vlm_grasped_bakeoff.py --backend qwen25vl
```

### Method D — RoboPoint (purpose-built, needs its own repo)

```bash
git clone https://github.com/wentaoyuan/RoboPoint.git .external/RoboPoint
pip install -e .external/RoboPoint
ROBOPOINT_REPO=.external/RoboPoint \
  python Code/run_vlm_grasped_bakeoff.py --backend robopoint
```

**Run all four if time allows** — that is the point of having four backends,
so nobody has to make this trip twice. Running several models is *not*
prompt-shopping: the prompt stays fixed, each model is reported separately,
and all results get reported including the bad ones.

### Do not use `--pool dense`

`--pool dense` exists in the runner but the dense cache is **deliberately not
in this repo** (8.7 MB of regenerated masks, kept on the Mac). The run would
abort with `SKIP no pool at ...`.

**Use the default frozen pool for everything.** It ships with the tracked
bundle and is what all the existing baselines were measured against, so it is
the correct comparison anyway.

### If a model refuses to emit coordinates

The parser handles Molmo XML (`<point x="52.1" y="31.4">`), bare `(x, y)`,
normalised 0–1, and 0–100 percentages. If a backend still returns no point,
the run records `no_point_parsed` plus the raw reply — **that is a result, keep
it.** If you must reword the prompt to get any parseable output at all, record
the change in the output JSON as `prompt_variant` and flag it loudly in your
report. Never silently swap it.

---

## 5. Ship the results back

```bash
git add figures/vlm_grasped_bakeoff/predictions_*.json
git commit -m "VLM grasped-role bakeoff: predictions from <backends run>

Models: <list>. Pool: frozen (and dense if run).
Prompt frozen per Code/run_vlm_grasped_bakeoff.py. Predictions only;
not scored on this host by design."
git push
```

The JSON files are small (a few KB). Nothing else needs to leave the box.

**Do not** copy reference masks onto this machine to check your work.
**Do not** score locally. **Do not** report whether it "looks right."

---

## 6. What happens next (on the Mac — not your job, listed so you know)

```bash
.venv_analysis/bin/python Code/score_vlm_grasped_bakeoff.py \
    figures/vlm_grasped_bakeoff/predictions_molmo2_frozen.json
```

Pre-registered marks, fixed before any model ran:

| result | meaning |
|---|---|
| **5/5** | works. Take it to the lab with the data request. |
| **4/5** | meets the frozen stop gate (≥4/5 across ≥3 groups). Same action. |
| **3/5** | beats everything tried (best 2/5), not conclusive. |
| **≤2/5** | no better than the existing hand-written rule. Say so plainly. |

Chance is ~1/87 per case on the frozen pool, ~1/208 on dense. 5/5 by luck is
about 1e-10, so a clean sweep is real evidence — which is the entire reason
for keeping this zero-shot.

---

## 7. Report back with

1. Which backends ran, exact HF model ids, and whether each loaded.
2. The per-case log lines (point coordinates, chosen proposal index, area).
3. Any case where no point parsed, **with the raw reply**.
4. Wall-clock per case and total.
5. Anything you changed and why — especially any prompt variant or quantisation.

Do **not** report a verdict. You cannot compute one without the references.

---

## 8. Hard rules

- **Branch is `interaction-bakeoff-prep`, not `main`.**
- **Never edit `PROMPT`** in `Code/run_vlm_grasped_bakeoff.py` to chase a better
  number.
- **Never bring reference masks onto this host.**
- **Never delete anything** — this repo archives, it does not remove. Keep
  failed logs.
- **No `Co-Authored-By: Claude` trailer** in commit messages in this repo
  (project rule in `CLAUDE.md`).
- Report failures as plainly as successes. Four methods failing cleanly is a
  genuinely useful outcome and is not a bad day.

---

## 9. Known traps

- **`import torch` took ~8 minutes** in the Mac venv. If it looks hung on the
  box, time it before assuming a crash — this was misdiagnosed twice.
- **SAM returns nested subpart/part/whole masks** for the same spot by design,
  so the whole-frame mask contains every point. The runner picks the
  *smallest* proposal containing the point for that reason. Do not "fix" it to
  first-match.
- **`figures/`, `Data/` are largely gitignored.** If something is missing, check
  `git check-ignore -v <path>` before concluding the pull failed.
- MPS-specific float32 patching in `Code/regenerate_dense_proposals.py` is a Mac
  concern and is irrelevant on CUDA.

---

## 10. Context if asked

- Deliverable and scope: `LAB_DELIVERABLE_A.md`
- Repo conventions and failure modes: `CLAUDE.md`
- Overnight pool result: `figures/proposal_dataset/dense_ceiling_report.txt`
- Failed ranker fits: `figures/proposal_dataset/ranker_fit_report.txt`,
  `ranker_fit_dense.txt`
- Earlier external-model bakeoff: `Docs/UBUNTU.md`

The honest standing position: the pipeline runs end to end, the candidate-pool
blocker was found and measured and partly fixed, and the picking stage cannot
be settled with 5 recordings. The VLM test is the last cheap thing to try
before the conclusion becomes "we need more demonstrations from the lab."
