# Search experiments prepared 13 November 2026

All seven proposals have implementation or tooling here. **v2 remains the
default.** Results are recorded in `BOT_EXPERIMENTS.md`: the extra-time
comparison passed its main gate, while dedupe, touching, play-evidence and
early stopping established no strength gain; banking was worse.

**The remaining commands are available experiments, not an automatic queue.**
Four stages remain: `policy-self`, `policy-all`, `typing`, `policy-inference`.
The compact policy currently matches the teacher's best move only 46% of the
time and allows roughly five times fewer sampled worlds at 2 ms. Screen its
quality/cost before a full tournament. Test opponent predictions on held-out
public histories before recommending typing or learned inference tournaments.
The bank comparison used 13% less actual time than its baseline, so it did
not isolate allocation at equal mean time. Do not rerun it unchanged.

The commands below preserve the prepared methods and settings. Long
comparisons are user-run; require a new concrete hypothesis and screening
evidence before recommending another.

## 1. Time scaling

```powershell
.venv\Scripts\python.exe -m ohhell.learn.search_experiment time
```

Dashboard: <http://localhost:8780/>. The runner prints the actual port if it
has to choose another. It builds the source-hashed native DLL first and
gives v2 10 ms against v2 at 2 ms. This measures the benefit of extra time;
it cannot establish an equal-time improvement or justify changing the default.
Add `--check-only` to any stage to load the entire lobby without playing games.
Saved artifacts go to `runs/gates/Search_*.{meta,jsonl,json}` with a fresh label.
Default deal seed is 1302026113; change it with `--seed NUMBER`.

For measured card-turn throughput rather than strength:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.search_diagnostics "hybrid:field=previous,time_ms=2" "hybrid:field=previous,time_ms=10" --out runs/cases/time-scaling-20261113.json
```

The report records actual counted and executed worlds, elapsed time, source
and checkpoint fingerprints. Resampled draws are correlated; neither count
is a count of independent deals. These isolated turns do not estimate a
whole-game strength gain or a pure C playout cost.

## 2. Three separate efficiency candidates

| Stage | Only candidate setting added | Meaning |
| --- | --- | --- |
| `dedupe` | `dedupe=1` | Merge repeated complete hands plus kinds within a native block; retain multiplicities. Fixed-budget deterministic values and RNG state match. Stochastic fields keep their original order/work. Python also caches deterministic repeats. |
| `touching` | `touching=1` | Fill shortlist slots from different touching-rank groups, retaining the model's preferred representative. Uses effective suits and bower power. **Changes behaviour:** rank-sensitive continuations can distinguish touching cards. |
| `play-evidence` | `play_evidence=1` | Blend root values with actual completed draws instead of the configured 32. At 32 draws it is the same blend. Draw count is evidence weight, not effective sample size. |

Run each separately, for example:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.search_experiment dedupe
```

The existing unsuccessful bid-evidence and adaptive-all-card experiments
remain separate. These stages do not widen the bid shortlist, change bid
blending, remove inference, or combine the three proposed changes.

## 3. Time management

Run the `early-stop` stage first. `stop_z=3,stop_min=32` can finish card search
at a block boundary when every paired simulation gap exceeds one point plus
three estimated standard errors. This is a heuristic stopping rule, not a
sequential confidence bound. It judges rollout scores; the final network
blend can choose differently. Exact endgames retain their existing stopping.

Then the `bank` stage compares that same stopping rule with and without
`controller=bank`. Only banked time changes in this comparison. The bank
earns 2 ms per decision with a choice, pays actual wall time, carries debt,
and caps saved credit at 8 ms. Forced actions earn nothing. Expensive deals
still overrun soft deadlines. Completed-round opponent reading also pays
from the bank when enabled.

The bank stage uses `--mean-time-ratio 1.05`. Even after passing score and
unseen-deal confirmation, a candidate above 105% of baseline mean time is
reported as `over time`, withholding `better`. This is a measured-time cap
with 5% tolerance, not a statistical equivalence test. Both main and
confirmation clocks, plus opponent-reading work, count. A promising bank
setting would still need an equal-mean-time comparison against unchanged v2
before promotion.

## 4. Learned continuations in C

`policy=learned_self` uses the compact learned policy for our later simulated
moves. `policy=learned_all` also uses it for simulated opponents, retaining
each kind's random-card probability. These affect simulated continuations
in **both bidding and card evaluation**. Root networks, shortlist and blend
remain v2; exact search still uses its original solver.

The prepared `ohhell/models/rollout_candidate/play.npz` is a 32-32-16-1
float MLP distilled from v2 on a fresh bower-rules case bank. Features see
only the acting seat's hand, public bids, trick, played cards and voids.
The fixed feature version and shapes are checked on load. It is a compact
feature scorer, not an incremental NNUE or an AlphaZero-trained policy.

Run `policy-self`; then `policy-all` measures only the extension
to opponents against the self-only candidate. That latter baseline is
experimental, so a positive result there does not promote a default.
The compact architecture and distillation jointly define this checkpoint;
their separate contributions are not established.

Rebuild the bank/checkpoint on another checkout (use fresh output names if
they already exist):

```powershell
.venv\Scripts\python.exe -m ohhell.learn.cases build --seed 20261113 --repeats 4 --max-hand 10 --out runs/cases/bowers-search-20261113.jsonl
.venv\Scripts\python.exe -m ohhell.learn.train_rollout --bank runs/cases/bowers-search-20261113.jsonl --out ohhell/models/rollout_candidate --epochs 40 --seed 20261113
```

To prepare one policy-improvement iteration, label a training bank with the
current learned continuation and fit a **new** checkpoint:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.position_bench label --bank runs/cases/bowers-search-20261113.jsonl --reference "hybrid:field=previous,policy=learned_self,worlds=512" --limit 3000 --out runs/cases/rollout-iteration1-labels.jsonl
.venv\Scripts\python.exe -m ohhell.learn.train_rollout --labels runs/cases/rollout-iteration1-labels.jsonl --out ohhell/models/rollout_iteration1 --epochs 40 --seed 20261114
```

Only training-split labels update parameters; validation selects the
checkpoint; test is measured afterward. Next iteration references must set
`rollout_models=ohhell/models/rollout_iteration1`. This is a prepared
distill/search/refit workflow, not evidence that bootstrapping converges.
No search-label policy-improvement iteration has been run yet.

The existing self-play generator also accepts the learned continuation through
its `--spec` seam, for updating the full root networks on newly played rounds:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.selfplay --labels search --start ohhell/models --out runs/selfplay/learned-continuation-1 --iterations 1 --rounds 150000 --epochs 4 --workers 6 --spec "hybrid:field=previous,policy=learned_self,rollout_models=ohhell/models/rollout_candidate,models={folder},time_ms=2"
.venv\Scripts\python.exe -m ohhell.learn.train_rollout --bank runs/cases/bowers-search-20261113.jsonl --teacher-models runs/selfplay/learned-continuation-1/iter1 --out ohhell/models/rollout_iteration1 --epochs 40 --seed 20261114
```

The compact policy stays fixed within that self-play iteration. The second
command distills the updated root into a new continuation checkpoint for the
next iteration. Compare updated root weights against the old root **with the
same continuation checkpoint** to isolate root training. Do not call a test
changing both weights and continuation checkpoint a single-change result.
These longer training/arena commands are prepared for you to run; only small
generator and data-path tests have run here.

## 5. Opponent reading

The `typing` stage adds per-seat persistent kind posteriors. At the end of a
visible round all played cards are public, so the tracker can reconstruct
opening hands and judge bids and decisionful plays. It excludes our seat
and blind rounds. A small prior floor and bounded log odds allow recovery
when the kind model is wrong. The same per-seat prior feeds native/Python
deal inference and simulated player kinds. It resets with a new bot/game.
The arena, tournament wrapper and web game deliver the completed-round hook.
Standalone training loops without that hook do not measure persistent reading.

The `policy-inference` stage is separate: the compact policy's softmax action
probabilities, mixed with each kind's `play_doubt`, replace the binary
rule-match likelihood. Temperature defaults to 10 round points. Rollout
continuations stay heuristic in this stage. Softmax probabilities are not
calibrated human-action probabilities; this is a hypothesis to test.

## 6. Position benchmark

Fresh reference labels use **only an observation and sampled hidden deals**,
never the bank's actual opponent hands. Every legal root card is evaluated
on common sampled worlds, using 512 draws and two seed replicates by
default. Native exact endgames remain available. Selection round-robins
table size, hand size, split and lead/follow strata.

```powershell
.venv\Scripts\python.exe -m ohhell.learn.position_bench label --bank runs/cases/bowers-search-20261113.jsonl --limit 1000 --out runs/cases/search-reference-20261113.jsonl
.venv\Scripts\python.exe -m ohhell.learn.position_bench evaluate --labels runs/cases/search-reference-20261113.jsonl --candidate "hybrid:field=previous,time_ms=2,dedupe=1" --out runs/cases/dedupe-position-report.json
```

Reports contain reference-value gaps and paired changes against a baseline,
split by existing case tags: table/hand size, bidding position, trick
position, early/late and target status. Snapshots are grouped by deal for
mean paired edges. Gaps are **not actual points lost or perfect-play error**:
sampling noise, continuation bias and strategy fusion remain. Independent
positions cannot assess a time bank's accumulated credit or persistent
opponent reading. Use full games for those.

## 7. Component seams and native requirement

Specs now select `sampler=belief|uniform`,
`policy=heuristic|learned_self|learned_all`, `evaluator=mlp|compact`, and
`controller=fixed|bank`. The new protocol interfaces live in
`ohhell/bots/components.py`. The uniform sampler disables soft inference;
it is a configuration seam, not a proposal to rerun the unchanged failed
no-inference candidate. Compact root evaluation is available for diagnosis,
but is not combined with the learned-rollout experiment.

Normal `fast=1` construction fails clearly if the matching native core is
missing. Explicit `fast=0` is the Python oracle; timed tournaments reject
it. Models are bot-local wrappers over immutable cached weights; C workspaces
remain process-local and are not thread-safe. Use worker processes.

Tournament manifests now fingerprint implementation source as well as
selected root/rollout checkpoints. An old manifest without that source
fingerprint cannot resume in the new implementation; use a new label.
No code from the earlier experiments was discarded, and no unproven option
was added to `DEFAULT_SPEC`.
