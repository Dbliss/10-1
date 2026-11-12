# NNUE card-play prototype

The `nnue` bot replaces v1 Hybrid's card-value network with a trained, floating
point NNUE evaluator. Bidding, blind-round policy, belief sampling, top-four
shortlist size, simulation blend, exact endgames and rule-based rollouts retain
v1 behaviour. The default remains `hybrid:field=previous,time_ms=2`.

## What is implemented

`ohhell/learn/nnue.py` uses 484 inputs, a 128-unit feature transformer, a
32-unit hidden layer and one round-score output. Card features use a fixed
rotation relative to trump, never a changing hand-size-based suit ordering.
Inputs contain only our hand and public information: seen cards, current
winning card, bids, tricks won, voids, positions and scoring parameters.

Each bot owns an accumulator. Changes to active features add or subtract their
first-layer contributions. Every legal candidate card branches from the same
position accumulator; branches do not modify it. The accumulator is rebuilt
every 64 updates to bound floating-point drift. Cached checkpoint weights are
read-only and shared; mutable accumulators are not shared. The implementation
is float32, with no integer quantization or specialised SIMD kernels.

This first candidate evaluates root-card choices in the existing hybrid
search. **NNUE is not yet called at internal simulated positions in the C
rollout/search engine.** Faster root inference therefore does not demonstrate
deeper or stronger search. Adding neural leaf evaluations is a separate
experiment requiring a continuation/value target appropriate to those leaves.

## First training run

The initial checkpoint was `ohhell/models/nnue_candidate/play.npz` (removed in the
12 November 2026 model cleanup; retrain to reproduce), with a
companion `training.json`. It was distilled from the current v1 play network
on `runs/cases/predetermined-v1.jsonl`, rather than learning fresh game outcomes.
This deliberately avoids changing rollout policy at the same time. It is a
new architecture, feature representation and distilled checkpoint together;
the experiment cannot attribute any difference to architecture alone.

Training seed 20261110; 20 epochs; batches of 1,024; Adam rate 0.001; hidden
sizes 128/32. Only non-blind positions with multiple legal cards are used.
Related deals stay together across splits. Validation selects the checkpoint;
test is evaluated only after selection. No current checkpoint was overwritten.

| Split | Action rows | Independent deals | RMSE against teacher, points | Teacher-best agreement |
| --- | ---: | ---: | ---: | ---: |
| Training | 71,960 | 368 | not measured | not measured |
| Validation | 6,731 | 27 | 10.200 | 45.90% |
| Test | 8,048 | 37 | 11.204 | 48.22% |

The test split has 2,308 choice positions. Mean teacher-score difference
between the teacher's best card and the candidate's selection is 1.322 points.
These measure imitation, not actual round scores or playing strength. Training
took 7.806 seconds after 6.073 seconds of data/teacher-target preparation on
this host. These times are a single run, not a hardware-independent promise.

Checkpoint SHA-256:
`d204435159cb7b31511830c43b8a873fd19d9a378a8fb0cfa8c8de8c947929bb`.
The training report preserves source/checkpoint fingerprints and the bank
manifest. The unchanged teacher hash is recorded in `BOT_EXPERIMENTS.md`.

To train another candidate into a fresh directory:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.train_nnue --bank runs/cases/predetermined-v1.jsonl --out ohhell/models/nnue_next --epochs 20
```

To learn from cached actual-policy outcomes instead, substitute
`--labels YOUR_LABELS.jsonl` for `--bank ...`. This recalculates the new NNUE
features directly from observations; the old feature export is not compatible.
The trainer refuses overlapping deal splits and nonempty output directories.
Rebuild the bank via `BOT_CASES.md` if it is missing after checkout.

## Verification and local diagnostics

Nine NNUE tests cover incremental/dense numerical agreement through round
changes and cache rebuilds, stable sparse encoding, observation privacy,
independent caches, save/load, unchanged bidding and endgame paths, both
simulation backends, dealer legality via complete games, checkpoint fingerprints,
training split leakage and batches smaller than the configured batch size.

`runs/cases/nnue-v1-diagnostics.json` records 256 sampled unseen choice
positions, seed 20261110. Maximum incremental/dense discrepancy was 0.000084
points. Five timed passes gave these isolated card-evaluation averages:

| Evaluator | Microseconds per call |
| --- | ---: |
| Existing MLP, existing features | 89.66 |
| NNUE, incremental path | 50.36 |
| Same NNUE weights, dense recomputation | 57.62 |

This includes feature extraction and scoring all legal cards. The MLP and NNUE
have different widths/features, so the entire difference is not attributable
to incremental updates. Position sampling causes jumps between states;
these figures are not an end-to-end search benchmark. External host load was
not measured. Do not infer a general turn-speed or strength improvement.

Three smoke games finished with one trained NNUE seat versus v1 seats:
3 players/opening 10, 4 players/opening 10 and 8 players/opening 6, seed 17,
standard scoring, dealer restriction on, full down/blind/up, 2 ms soft turn
limits. Their scores are saved only as completion evidence, not a comparison.

Repeat diagnostics with a fresh output filename:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.check_nnue --bank runs/cases/predetermined-v1.jsonl --out runs/cases/nnue-repeat-diagnostics.json
```

## User-run comparison

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "nnue:field=previous" --label "NNUE float v1" --workers 6 --port 8772
```

Dashboard: <http://localhost:8772/>. The tournament adds the same 2 ms
allowance to candidate and baseline and fingerprints the selected NNUE play
and current bid checkpoints. For another checkpoint, add
`,play_models=ohhell/models/nnue_next` inside the candidate spec and use a
fresh label. Run separately from other comparisons. No tournament or
unseen-deal confirmation has run for this candidate; it must pass both before
promotion. The limited teacher fidelity is a concrete reason it may lose.
