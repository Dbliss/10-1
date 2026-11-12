# Independent bidding and card-play benchmarks

The bot already has separate bid and play networks. `ohhell.learn.cases`
provides repeatable decision tests for each, without changing the default bot
or either checkpoint. Run commands from the repository root.

A bank is already prepared locally at `runs/cases/predetermined-v1.jsonl`
(seed 20261110, eight repeats): **54,560 decisions**, comprising 10,240 bid
and 44,320 play cases from **544 independent deals**. It has 45,648 training,
3,996 validation and 4,916 test snapshots. Related snapshots are not independent
trials. This 34 MB file and its manifest are gitignored; rebuild on another
checkout using the build command below with `--repeats 8` and that filename.
Large-bank labels have not been generated. Start labeling this prepared bank
by substituting its filename for `runs/cases/v1.jsonl` in the label commands.

## Build the predetermined case bank

```powershell
.venv\Scripts\python.exe -m ohhell.learn.cases build --out runs/cases/v1.jsonl --seed 20261110 --repeats 4
```

This crosses 2–8 players with every opening hand size from 1–10 where legal,
plus blind 0. For each deal it runs four bidding profiles: heuristic,
lowest-legal, highest-legal and random-legal. It saves **every bidding and
card-play decision**, including forced moves. Dealer rotates across repeats;
all bidding positions occur in every round. Increase `--repeats 100` for a
much larger bank; at least eight repeats exercise every dealer seat at every
table size. The seed fixes the deals, bids, histories and data splits.

Coverage includes trump counts, zero targets, targets already met,
overshoots, unreachable targets, early/late play and trick position. Actual
counts are recorded in the `.meta.json` sidecar; rare tactical categories
are measured rather than assumed. These are stratified generated cases,
not hand-authored proofs of the correct move.

All snapshots and bid-profile variants of a deal remain together in an
80/10/10 hash-based train/validation/test split. The proportions are approximate
for small banks. Do not train on validation/test exports. Related positions
are correlated, so increasing snapshot count does not increase independent
deal count by the same amount.

## Cache outcomes once

Fast plumbing/diagnostic labels:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.cases label --bank runs/cases/v1.jsonl --out runs/cases/v1-heuristic.jsonl --continuation heuristic --opponents mixed --samples 1
```

For labels reflecting the actual current player, run this separately yourself:

```powershell
.venv\Scripts\python.exe -m ohhell.learn.cases label --bank runs/cases/v1.jsonl --out runs/cases/v1-hybrid.jsonl --continuation "hybrid:field=previous,time_ms=2" --opponents mixed --samples 4
```

For **every legal root action**, the runner finishes the saved round. Our seat
uses `--continuation`; other seats use `--opponents`. Future bidders respond to
the forced bid, obeying the dealer restriction. Bots receive observations only,
including hidden hands in blind rounds. Common per-seat seeds couple options.
Samples vary policy randomness on the **same hidden deal**; they do not sample
additional possible hands. Timed bots are not bit-for-bit reproducible.

The heuristic labeler is cheaper but teaches heuristic continuations. Replacing
the continuation is an intentional dataset change. Larger banks and actual-bot
labels can take substantial time: no claim of fast label generation is made.
Progress prints every 1,000 cases. There is no dashboard for this CLI.

Banks and labels have completion manifests, hashes, settings and coverage;
labels also record policy source/checkpoint fingerprints. Incomplete or edited
inputs fail validation. Use fresh output filenames; existing files are refused.
Keep sidecars with their JSONL files.

## Test bidding and play separately

```powershell
.venv\Scripts\python.exe -m ohhell.learn.cases evaluate --labels runs/cases/v1-hybrid.jsonl --part bid --candidate "hybrid:field=previous,bid_models=ohhell/models/candidate,time_ms=2" --out runs/cases/bid-report.json
.venv\Scripts\python.exe -m ohhell.learn.cases evaluate --labels runs/cases/v1-hybrid.jsonl --part play --candidate "hybrid:field=previous,play_models=ohhell/models/candidate,time_ms=2" --out runs/cases/play-report.json
```

Both default to the current default bot (v2) and the test split. These examples
illustrate component isolation; neither candidate checkpoint is a newly
validated improvement. `--candidate net` tests the networks without search;
set `--baseline net` when comparing a net-only candidate to a net-only control.
Use `--split validation` during development and reserve test for final checks.

Reports contain paired round-point differences, individual decisions,
per-category results, source/model hashes, and independent deal counts. Edge
is averaged within each deal first. Forced decisions are reported separately
under `forced` versus `choice`. Timing includes bot construction and is only
diagnostic. Only the root decision changes; subsequent actions use the frozen
labeling policies. Thus this measures decision quality under those policies,
not the candidate's whole-round continuation quality.

**The hindsight gap is not optimal-play accuracy.** It compares actions on
the actual saved hidden deal, information the bot cannot see. A sensible
decision can lose this comparison. Do not train a classifier on the single
best hindsight action or demand zero gap. Learn outcome distributions/expected
scores over many deals instead. These diagnostic scores have no tournament
promotion gate and do not measure cumulative down/blind/up game strength.

## Export cached training data

```powershell
.venv\Scripts\python.exe -m ohhell.learn.cases export --labels runs/cases/v1-hybrid.jsonl --part bid --split train --out runs/cases/bid-train.npz
.venv\Scripts\python.exe -m ohhell.learn.cases export --labels runs/cases/v1-hybrid.jsonl --part play --split train --out runs/cases/play-train.npz
```

Repeat for `validation` and `test` with distinct filenames. Arrays contain
`x`, `y`, `limits`, `deal_id`, `part`, `split` and `skipped_blind_cases`:

- Bid `y` is the final trick count, suitable for the existing categorical
  distribution loss, with the candidate bid encoded in `x`.
- Play `y` is final round points divided by the existing `VALUE_SCALE` (30),
  suitable for the existing value network loss.
- Each policy sample is one row per action; related rows keep their deal ID.
- Blind cases are tested but excluded from network exports because the current
  feature encoding has no blind representation. The existing blind policy stays.

No simulation is needed to repeatedly fit these cached arrays. Actual training
speed depends on bank size and hardware. This adds data export, not a new
trainer: do not use the existing trainer's random row holdout as evidence of
generalisation on these correlated cases. A future cached-data trainer should
select checkpoints using the separate validation export and evaluate test once.

This framework leaves both models and the bot default untouched. Any strength
claim still requires the full-game main gate and unseen-deal confirmation.
