# Bot experiment history

Updated 12 November 2026. This is the durable record for future agents.

## Model cleanup (12 November 2026)

Only the promoted default's weights are kept. v2's `selfplay_bowers` networks
were moved to the default slot, `ohhell/models/bid.npz` (a84e9315...e29d) and
`ohhell/models/play.npz` (8cb5a064...d474), hashes checked after the move,
with its `training.json`. v2's spec is now `hybrid:field=previous,time_ms=2`
(no `models=`). Wherever this log names `models=ohhell/models/selfplay_bowers`,
read it as the default folder.

Deleted, so the related candidates cannot be rerun without retraining:
- v1's Ace-high networks, formerly at `ohhell/models/bid.npz` (ed3ab2e4...f12c)
  and `play.npz` (be178e28...7580). v1 was removed from `VERSIONS`.
- `ohhell/models/bowers/` (retrained only, fddb284a... / d2b19729...),
  `search_bowers/` (b5c5d534... / 74b2ba27...), `seat_bids/` (bid 011144fd...),
  `sim_bids/` (play 5a0060eb...), `candidate/` (bid 8dd7a919...) and
  `nnue_candidate/` (d2044351...). The `nnue` bot now needs `play_models=`.

Kept: the trick-estimate coefficient files and `one_card_ranges.json`, which
the `estimate=2/3` options, the bid helper and the `ranges` bot use. The
tournament candidate is now a required argument (its old default pointed at
`ohhell/models/candidate`). `tests/test_models.py` builds its own perturbed
checkpoint. Suite after the cleanup: 466 passed. No strength was measured.

## Rules change: bowers (11 November 2026)

The game now ranks trumps with bowers, always on (no rules toggle). The Jack
of trump (right bower) is highest, then the other Jack of the same colour
(left bower), then Ace down to two. The left bower is a trump for every
purpose: it follows trump leads, cannot follow its printed suit, shows voids
as a trump, and counts in trump totals. The trump suit has 14 cards that
round and its same-colour suit 12. Side suits are unchanged.

Every result below this section was measured under Ace-high trumps. Treat
those edges, margins and verdicts as history, not as evidence about the
current game. Nothing has been re-run under bowers yet.

- The engine, C core (rebuilt as a new source-hashed DLL), heuristic policy,
  bidding estimates, blind-round odds, belief sampling, exact search, feature
  flags and cases all use the bower ranking.
- All trained weights predate the change and were trained on Ace-high deals:
  `ohhell/models/bid.npz`, `ohhell/models/play.npz`,
  `ohhell/models/candidate/*.npz` and `ohhell/models/nnue_candidate/play.npz`.
  They load and play but are stale. The input layouts are unchanged, so the
  left bower is still encoded as a Jack of its printed suit; only the derived
  "wins now"/"boss" flags follow bowers. The NNUE `canonical_card` rotation
  was deliberately left alone so the existing NNUE keeps loading; give the
  left bower a fixed slot (trump, same colour, others) and bump
  `FEATURE_VERSION` when it is next retrained.
- Smoke check only, not a comparison: on seeds 0-39 four `heuristic` bots
  averaged 29.7 points per seat per 4-player game from 10; on seeds 0-2 four
  v1 Hybrid bots averaged 157.5.
- Missing evidence: no tournament under bowers and no re-baselining of v1.
  Any candidate or retrained model must be gated against v1 under the new
  rules.

### Prepared candidate: default networks retrained under bowers (11 November 2026)

Single change: v1's bid and play networks retrained under bowers with the
unchanged recipe, saved to `ohhell/models/bowers/`. Hybrid search, inference,
blending, time allowance and opponent pool are unchanged, and v1's own weights
were not overwritten.

- Command: `python -m ohhell.learn.train --rounds 400000 --epochs 12 --seed 1
  --workers 19 --out ohhell/models/bowers`. Default mixed-opponent field,
  heuristic-population rollouts, same architectures (bid 98-192-96-11, play
  241-320-160-1). Took 1,694,628 bid and 1,951,158 play samples in 84 s, and
  732 s in total. Log: `runs/training/bowers_train.log`.
- Final held-out loss: bid 1.0046, play 0.0946. Both decreased every epoch.
- SHA-256 fingerprints: `bid.npz` fddb284a21e5ba810af7a7f5f8316d004b8458825b41bb95ec3065f9089a9a12,
  `play.npz` d2b197299af67d3d35f337fcf421ae1acb8dc23803eb5975c878347bde0f4d05.
  The v1 weights being compared against are `ohhell/models/bid.npz` ed3ab2e4â€¦12c
  and `play.npz` be178e28â€¦7580.
- Offline fit on fresh bower-rules data (40,000 rounds, seed 987654, unseen
  in training; 169,620 bid / 196,953 play rows):

  | Weights | Bid NLL | Bid top-1 | Play Huber |
  | --- | --- | --- | --- |
  | v1 (Ace-high training) | 1.1501 | 0.521 | 0.1155 |
  | Retrained under bowers | 1.0107 | 0.553 | 0.1039 |

  This measures fit to the training targets, not game strength.
- Smoke check: full 3, 5 and 8 player games completed with
  `hybrid:field=previous,time_ms=2,models=ohhell/models/bowers`.
- Missing evidence: no tournament yet. Candidate spec:
  `hybrid:field=previous,models=ohhell/models/bowers`. Not promoted.

### Prepared candidate: self-play policy iteration from the bowers networks (11 November 2026)

New method in `ohhell/learn/selfplay.py`, separate from `train.py`. Each
iteration:
- deals rounds with about 60% of seats bidding and playing with the current
  networks, and the rest drawn from the `mix` population;
- labels every bid and card option at sampled model-seat decisions by
  finishing the round with that seat following the current networks;
- warm-starts and refits both networks on those labels.

This makes labels on-policy, where `train.py` labels them under the heuristic.
It starts from `ohhell/models/bowers`. The only change against that starting
point is the weights; hybrid settings are unchanged.

- Command: `python -m ohhell.learn.selfplay --start ohhell/models/bowers
  --out runs/selfplay/bowers --iterations 4 --rounds 150000 --epochs 4
  --workers 19`. Defaults: model share 0.6, epsilon 0.05, seed 2 (iteration
  seed `2000 + i`). About 550k bid and 635k play rows per iteration; 2,015 s
  in total. Log: `runs/selfplay/bowers.log`; per-iteration record:
  `runs/selfplay/bowers/history.json`.
- Built-in check: network-only bot (`net`) against `net` on the starting
  weights, 4 players, opening hand 10, full down/blind/up games, 300 seeds
  per iteration with every seat rotated:

  | Iteration | Edge vs start | Win share (fair 25%) |
  | --- | --- | --- |
  | 1 | +18.7 +/- 3.8 | 31.3% |
  | 2 | +22.1 +/- 3.9 | 32.7% |
  | 3 | +22.8 +/- 3.8 | 33.9% |
  | 4 | +23.5 +/- 3.9 | 33.9% |

  The gain levels off after iteration 2. Only a fixed 4-player table against
  one opponent type was tested.
- Iteration 4 copied to `ohhell/models/selfplay_bowers/` (with
  `training.json`). SHA-256: `bid.npz` a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d,
  `play.npz` 8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
- Quick hybrid check (arena, not the tournament):
  `hybrid:field=previous,time_ms=2,models=ohhell/models/selfplay_bowers`
  against the same spec with `models=ohhell/models/bowers`. 4 players, 300
  seeds from 500000: **+4.2 +/- 4.2** points per game, win share 26.3%,
  both at 2.0 ms/decision. Positive but within noise; the search mostly
  washes out network differences.
- Tournament (run by the user, saved as `runs/gates/Self-play_bowers_nets.json`):
  candidate `hybrid:field=previous,models=ohhell/models/selfplay_bowers,time_ms=2`
  against v1 `hybrid:field=previous,time_ms=2`, with bowers rules, the
  down/blind/up format, opening hand 10 and 2 ms per turn for every bot.
  Seed 938885093; the run fingerprints matched the hashes above, and v1's
  `ohhell/models/bid.npz`/`play.npz` were ed3ab2e4â€¦12c / be178e28â€¦7580.
  - Main stage: **+9.93 +/- 7.56** points per game after 390 sets (3,660
    games), stopping at the 390-set check.
  - Confirmation on unseen deals: **+11.37 +/- 4.35** over 390 sets.
  - Win-share edge against v1: +0.068 +/- 0.038; the dashboard reported no
    disagreement. Candidate and v1 both ran at 2.37 ms per turn.
  - Main-stage edge by table size: 3p +17.50 (60 sets), 4p +6.22 (180),
    5p +17.13 (60), 6p +11.33 (30), 7p +7.57 (30), 8p +3.62 (30).
    Per-size margins are wide.
- Verdict: **better, confirmed. Promoted as v2 Self-play bowers nets.**
- This is not a single-change result against v1. The candidate differs from
  v1 in two weight changes at once: retraining under bowers, then self-play.
  The separate contribution of each was not measured. The retrained-only
  candidate (`models=ohhell/models/bowers`) has no tournament result.

### Prepared candidate: nets trained on v2's search scores (11 November 2026)

Hypothesis (user's): let full v2, search included, play the self-play deals,
fit the nets to the scores the search gives each option, and repeat, so the
search improves the nets and the nets improve the search. Implemented as
`--labels search` in `ohhell/learn/selfplay.py`; the rollout mode is unchanged.
Single change against v2: the weights. Hybrid settings are unchanged.

- Model seats (share 0.6) are `hybrid:field=previous,time_ms=2,models=<current>`
  acting from their observations; other seats are the `mix` population.
  Epsilon 0.05 random cards at model seats.
- Labels at sampled model-seat decisions (bids 30%, cards 15%), from that
  observation with the bot's own native simulator and a fixed budget, not the
  2 ms deadline: bids are the search's trick distribution for each bid in the
  heuristic estimate -2..+3 (256 sampled deals); cards are the search's mean
  points for every legal card (128 sampled deals; exact endgames where the
  simulator uses them). For reference, v2 completes about 285 bid deals in
  2 ms. The bid net is fitted with soft-target cross-entropy
  (`soft_softmax_loss` in `nn.py`).
- Command: `python -m ohhell.learn.selfplay --labels search --start
  ohhell/models/selfplay_bowers --out runs/selfplay/search --iterations 3
  --rounds 150000 --epochs 4 --seed 3 --workers 19`. Log:
  `runs/selfplay/search.log`.
- Iteration 1 (seed 3001): 550,826 bid and 637,652 play rows in 376 s.
  Held-out loss: bid soft cross-entropy 1.0021, play Huber 0.0177 (the labels
  are averages over deals, so they are far less noisy than rollout labels).
- Iterations 2-3 did not run. Another session edited `selfplay.py` during the
  run (adding `--bid-inputs`/`--freeze-play` and a job field); the iteration-2
  worker pool loaded the edited code and crashed on the job tuple. The code as
  it now stands passes `tests/test_selfplay.py`. Iteration 1 was not affected.
- Built-in check of iteration 1 (arena, not the tournament): hybrid at 2 ms
  with the new nets against v2, 4 players, opening 10, 300 seeds from 910000:
  **-5.1 +/- 4.6 points per game**, win share 22.6%, 2.0 ms/decision. Worse,
  just outside the interval.
- Likely reason (untested): v2's search rolls out with the rule-based
  population and never queries the nets, so the labels do not depend on the
  nets. The nets become a copy of the search, so their prior adds nothing new
  and the information the self-play nets brought (values under their own
  continuation) is lost. Repeating would change only which positions are
  visited, not the labels, so the loop cannot bootstrap as hoped. For that,
  the search's continuations would have to use the nets.
- Saved: `ohhell/models/search_bowers/` (with `training.json`). SHA-256
  `bid.npz` b5c5d5341671b631a9dc679693045fc715d6a4fe146b753908816924a6fa699e,
  `play.npz` 74b2ba27796b2f54115ae47ccc5ada959fa6a00ce130aa0f0386147698623832.
  v2's weights were not overwritten.
- Missing evidence: no tournament. Candidate spec:
  `hybrid:field=previous,models=ohhell/models/search_bowers`. Not promoted.

### Prepared candidate: seat-bid inputs and net-imagined later bidders (11 November 2026)

Two changes the user asked for together, so this is **not a single-change
test** against v2:
1. Bid network inputs. The 98 inputs stay as they were, and 29 more follow
   (`BID_SEAT_INPUTS = 127` in `ohhell/learn/features.py`). For each bidding
   position 0-6 (0 is the first bidder after the dealer) there are three:
   whether that seat bid before us, its bid over the hand size, and whether
   that bid was zero. Then comes a one-hot of how many seats still bid after
   us (0-7). `Models` picks the layout from the network's input width, so
   98-input networks load and behave exactly as before (tested).
2. Later bidders imagined with the bid network. `seatbids` (in
   `ohhell/bots/seat_bids.py`) is v2's hybrid with a different bid search.
   In each sampled deal, and for each of our candidate bids, every later seat
   bids in turn. It takes the bid the network expects to score best for its
   imagined hand, seeing every earlier bid including ours, and then gets its
   kind's bid noise. The dealer rule is respected. To fit the 2 ms allowance,
   each imagined seat compares only bids from floor(estimate)-1 to
   ceil(estimate)+1 around the rule-of-thumb estimate. Over 3,000 random
   spots this window contained the network's unrestricted choice 97.3% of
   the time; the rule-of-thumb bid itself matched the network only 64.2% of
   the time. New C exports `oh_sample_worlds` and `oh_score_bids` (DLL
   `core_1a68b374890b5456.dll`) do the sampling and play-outs. Card play,
   blending, inference and endgames are v2's. With `imagine=0` the bot
   keeps the rule-of-thumb later bidders, which tests change 1 alone.

Training: `python -m ohhell.learn.selfplay --start ohhell/models/selfplay_bowers
--bid-inputs 127 --freeze-play --out runs/selfplay/seat_bids --iterations 3
--rounds 150000 --epochs 4 --seed 3 --workers 19`. v2's bid network was
widened with zero weights on the new inputs, so it started out identical to
v2's. The play network was frozen, and only bids were labelled (about 550k
rows per iteration). The run took 827 s; log: `runs/selfplay/seat_bids.log`.
- Built-in network-only check against the widened start (4 players, opening
  10, 300 games each): iteration 1 +0.2 +/- 1.6, iteration 2 -0.2 +/- 1.7,
  iteration 3 -0.4 +/- 2.0. No gain is visible.
- Offline fit on fresh labels (40,000 rounds, seed 987654, v2 policy, 145,060
  bid rows): bid NLL 1.0036 for the start (v2) against 1.0012 for
  iteration 3; top-1 0.559 against 0.560. Mean |weight| on the new inputs
  rose from 0.014 to 0.022 over the iterations; the old inputs were at 0.12.
  The new inputs add little over the earlier-bid total the network already
  had.
- Some of any change may come from three more self-play iterations rather
  than the inputs. No control run with the same recipe at 98 inputs exists.
- Iteration 3 was saved to `ohhell/models/seat_bids/`, with `training.json`.
  SHA-256: `bid.npz` 011144fd38ec32c64ffd7c6f41cd5a837beea9835946dbabab1ca19d12025df1,
  `play.npz` 8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474
  (identical to v2's play network).

Time cost: imagining with the network is expensive. In a 2 ms bid turn, v2
completes about 130-260 sampled deals; `seatbids` completes about 8-30.
Bid turns run 2.5-3.1 ms, because at least one block of 8 deals always
finishes, against 2.3 ms for v2. Card turns are unchanged.

Arena smoke check (not the tournament):
`seatbids:field=previous,time_ms=2,models=ohhell/models/seat_bids` against
v2. 4 players, opening 10, 300 seeds from 700000: **-2.7 +/- 4.1** points
per game, win share 25.0%, 2.3 ms per decision. This is within noise and
leans negative.

Tests: `tests/test_seat_bids.py` (15 tests) and an updated
`tests/test_selfplay.py`; the full suite passed (393). The tournament now
fingerprints `seatbids` weights.

Missing evidence: no tournament or confirmation yet. Candidate spec:
`seatbids:field=previous,models=ohhell/models/seat_bids`. Not promoted.

### Removed: raw-compute bot, everyone plays optimally (11 November 2026)

**Significantly worse than v2; code deleted at the user's request.** Do not
rebuild it unchanged.

What it was: `exact`, a reference bot outside the equal-time rule. On each
decision it sampled 64 deals consistent with its observation (field
`skilled`, inference on). It scored every legal card and bid by exact max-n
search, with each seat maximising its own points on that deal and ties going
against the bot, as in v1's endgame solver. It used no model and no
shortlist. Seats followed the rule-based policy until at most 20 cards were
held (v1: 12), because full-hand max-n was out of reach: about 1.3 s per deal
at 4 × 8 and 6 × 6 cards, and 4 × 9 didn't finish in minutes. Unbid opponents
bid the rule-based estimate. Its C search (`solve_exact`) had a 2^21-entry
table and a safe cut-off. It matched the unpruned Python solver on 392
decisions and cut 3 × 10 full hands from 90.8 s to under 0.1 s per deal.

Results:
- Smoke arena against v1 (4 players, opening 10, seeds 2000-2029):
  -9.8 +/- 12.0 points per game, win share 23.3% (fair 25%).
- Tournament reference run against **v2 Self-play bowers nets**:
  `python -m ohhell.tournament exact --reference --label "Raw compute
  reference"`, seed 444117355, `exact` untimed and every other bot at 2 ms.
  The user stopped it after **44 sets / 208 games** (sets 0-43). There is no
  summary `.json`; the figures were computed from
  `runs/gates/Raw_compute_reference.jsonl`. Edge **-27.5 +/- 12.8 points per
  game** (plain 95% interval over sets). Win-share edge -0.174 +/- 0.116.
  `exact` took 176 ms per turn against v2's 2.3 ms. By table size: 3p -48.1
  (7 sets), 4p -38.7 (19), 5p -24.3 (7), 6p -2.9 (4), 7p +5.0 (4), 8p +7.5 (3).
  The small tables lost badly; the large-table sets are too few to read.
- Likely reasons (untested): it assumed perfect opponents at a table of
  imperfect ones; it solved each deal as if every hand were visible
  (strategy fusion; no value for hiding or probing); and the rule-based
  opening of big hands remained. It suggests that much more compute alone
  does not help without a better model of opponents.

Removed: `ohhell/bots/exact.py`, `tests/test_exact.py`, the registry entry,
the `oh_exact_*` C functions and their bindings. `core.c` is back to source
tag `d06cd8253b1deec3`, the DLL that existed before. The intermediate DLLs
`core_bef37327…` and `core_bee2d791…` were deleted. Suite: 376 passed.

Kept: the tournament's `--reference` mode, for comparing any bot that uses
more time. Only the candidate is untimed. There are no checks, no repeat and
no promotion, the margins are plain 95%, and `.meta` records
`"reference": true`.

### Inconclusive: exact endgame from 16 cards (11 November 2026)

Single change against v2: `endgame_cards=16` (v2 uses 12), so the exact
max-n solver starts when at most 16 cards remain in all hands. Everything else
is v2: `hybrid:field=previous,models=ohhell/models/selfplay_bowers,endgame_cards=16,time_ms=2`.

- Timing before the run (native solver, one sampled deal, every legal card,
  first lead of a trick, 12 deals per size, untimed): median per deal 0.12 ms
  at 3x4 cards, 0.26 ms at 4x4, 0.11 ms at 5x3, 0.29 ms at 6x3, 0.05 ms at
  8x2. Cost depends on cards per player more than the total: 8x3 (24) 0.5 ms
  but 4x6 (24) 107 ms; 3x5 0.35 ms median with a 59 ms outlier; 4x5 3.7 ms;
  4x7 2.6 s. A rule-based play-out costs about 0.01 ms per deal, so exact
  search trades roughly 150+ sampled deals for 15-20 at 2 ms. The deadline is
  only checked between deals, so a single hard deal can overrun the turn.
- Tournament (run by the user): bowers rules, down/blind/up, opening hand 10,
  2 ms per turn for every bot, baseline v2, seed 724582025, model hashes
  `bid.npz` a84e9315...e29d and `play.npz` 8cb5a064...d474 (match v2).
  Stopped by the user after **450 sets / 2,111 games** (sets 0-449); no
  verdict and no summary `.json`. Figures computed from
  `runs/gates/Endgame_16_cards.jsonl`: edge **+0.66 points per game**, plain
  95% +/- 3.76 (planned gate margin at this point +/- 6.81). Win-share edge
  +0.016 +/- 0.033. By size: 3p +3.57 (69 sets), 4p +0.58 (208), 5p -1.65
  (69), 6p +1.95 (35), 7p -0.37 (35), 8p -0.37 (34); all margins wide. Mean
  time per turn 3.08 ms for both sides (host load was higher than in earlier
  runs).
- Verdict: **inconclusive; keep v2.** The change only affects decisions with
  13-16 cards left (about one trick per round), so any effect is small;
  resolving +/-1 point would need about 6,000 sets. Not worth repeating
  unchanged. A per-player-hand threshold is untested.

### Negative smoke check: calibrated quick trick estimate (12 November 2026)

Single change against v2: `estimate=2`, i.e.
`hybrid:field=previous,models=ohhell/models/selfplay_bowers,estimate=2,time_ms=2`.
It swaps one function, the quick trick estimate, everywhere the simulation
uses it: (a) the shortlist of our bids to simulate (now floored at certain
trump tricks), (b) the bids of seats still to bid in each sampled deal, and
(c) the inference weight of sampled deals against bids already made (bid and
card turns). Card play, networks, blend, `bid_sigma` and endgames are v2's.
The trained networks still receive the original estimate as an input.

The estimate (`ohhell/estimate.py`, C port `card_win_probs2` in `core.c`):
exact hypergeometric features per card (higher cards falling on your earlier
ones, opponents running out of the suit, an opponent out of the suit holding
a trump), first-trick seat, strength of the other cards (winning gives the
lead), trump shape, and interactions with hand size, one-card rounds and
player count. A logistic model per card type turns these into win chances.
Trumps with no unseen higher trump are certain. Fitted with
`python -m ohhell.learn.calibrate_estimate --train 100000 --test 30000 --write`
(seed 1, test seed 1000001) on play-outs where every seat bids with the
original formula and plays with the rule-based player. Coefficients
`ohhell/models/trick_estimate.json`, SHA-256
8b99e3c362a04c4805fa8a411f2bee80984444f6d4c97337a2725fbcd826485f. Native core
`core_b3406429b0d40deb.dll`. Report: `runs/trick_estimate_report.json`.

Offline results (30,000 unseen rounds, 2-8 players, hands 1-10):
- Per-seat bias -0.20 tricks (old) against +0.00 (new); at 10 cards -1.01
  against -0.08. RMSE 0.761 against 0.647. Trick-count NLL 0.894 against
  0.779. Card Brier 0.1070 against 0.0992. At 10 cards the table's estimates
  summed to 3.5 tricks short of the hand size (old) against 0.3 (new).
- Side-suit Aces, 7+ cards: the leading seat is now within about a point for
  2-5 players (4 players: actual 78.1%, new 77.2%, old 67.1%). The last seat
  (dealer) is overrated by 6-14 points, and 6+ players by 5-14 points in every
  seat. Diagnosis: non-leading Aces are often thrown away when their holder is
  out of the led suit and trying to lose (13-15% against 2% when leading; 25%
  when the dealer rule forced the dealer's bid). Seat totals stay calibrated
  (within 0.05 tricks for lead, middle and last).
- Shortlist: over 3,000 spots the old shortlist missed the best bid (blended
  value over every legal bid) 0.1% of the time, losing 0.006 points per spot,
  so the shortlist was not a reason for this change.
- Predicting v2's actual bid (7,484 bids, seed 21): same bid 74.0% (old)
  against 80.1% (new); with 7+ cards 42.1% against 62.6%, mean bid gap -0.65
  against -0.02.
- Cost at 2 ms: median deals per bid turn 257 (v2) against 193 (new), after
  tabulating the counting features per table shape (129 before). The Python
  shortlist estimate costs about 64 us per bid turn.

Arena smoke check (not the tournament): against v2, 4 players, opening 10,
300 seeds from 700000, 2 ms: **-7.2 +/- 4.1** points per game (127.4 against
134.6), win share 22.8%, 1.7 ms per decision. Clearly negative.

Untimed follow-up, same seeds, both sides fixed at `worlds=32,bid_worlds=128`
(no time limit, so equal deals per decision): **+3.0 +/- 4.0** points per game
(135.2 against 132.2), win share 26.1%, 0.9 ms per decision. Inconclusive on
its own, but about 10 points better than the timed run, which suggests the
timed loss comes largely from fewer deals per turn rather than from the
estimate itself. Two 300-game runs are not enough to confirm that; the gap
between them is roughly 10 +/- 6.

Verdict: **do not run the tournament with this candidate; keep v2.** A better
predictor of v2's bids made the bot weaker here. Untested explanations: fewer
deals per turn; `bid_sigma` and the networks' 50% blend were tuned alongside
the original estimate; imagined later bidders now bid more, which may not fit
how the simulated players then play. Tests: `tests/test_estimate.py` (C and
Python agree to 1e-9 on 2,000 hands). The bid helper uses `estimate=2` for its
shortlist and imagined bids; that is not evidence of playing strength.

### Prepared candidate: one-card bids read as card ranges (12 November 2026)

User's idea. Single change: in visible one-card rounds only (not blind 0),
v2's bid is replaced by an exact range calculation. Everything else,
including card play, blind bids and every larger hand, is v2 unchanged.
Implemented in `ohhell/bots/ranges.py`.

- Every bidder is assumed to use this same rule. The first bidder (who leads)
  bids 1 when the exact chance that no other card beats theirs is over 0.5.
  A later bidder treats each earlier bidder as holding a card from the set
  that would have produced that bid given the bids before it (bid 0 gives the
  complementary set), and everyone still to bid as random. The chance is an
  exact count over all deals consistent with those ranges. The bid is the
  one with the higher expected score, which is "bid 1 if p > 0.5" under
  default scoring. If the ranges cannot all be true at once, the latest are
  dropped until they can.
- The user expected each 1-bid to halve the range. Exact ranges with trump
  7 of clubs, all earlier bids 1, in cards out of 51: 3 players 19, 8;
  4 players 11, 5, 2; 8 players 5, 2, 1, 1, 1, 1, 1. Each 1-bid roughly
  halves the range, but the first bidder's range is narrower than the top
  half once there are more than two players.
- Computing ranges live took up to 150 ms per decision on cold 8-player
  tables, so `python -m ohhell.learn.one_card_ranges` precomputes every
  bidding sequence for 2-8 players and the 13 trump ranks (suits relabelled
  so trump is clubs) into `ohhell/models/one_card_ranges.json` (101 s).
  Lookups take no measurable time. `live=1` recomputes instead; non-default
  scoring always recomputes. The tournament fingerprints the table.
- SHA-256: `one_card_ranges.json`
  6d8435ab71d4cc14513d4dd8207bca10058840f31de46eedfa0c59f2b57bd470;
  v2 `bid.npz` a84e9315...e29d, `play.npz` 8cb5a064...d474.
- Arena smoke check (not the tournament): candidate against v2, both
  `time_ms=2`, `--hand 1` (rounds 1, blind 0, 1 only), 3,000 games each,
  seeds from 900000, 95% intervals in points per game: 3 players
  +0.2 +/- 0.1, 4 players +0.4 +/- 0.1, 6 players +0.3 +/- 0.1,
  8 players +0.5 +/- 0.1. Over a full 10-down/up game only two rounds
  change, so the full-game edge will be small relative to tournament noise.
- Tests: `tests/test_ranges.py` (15), covering exact counts against brute
  force, the table against the live calculation on all trump suits, legality
  under the dealer rule, and hybrid fallback. Full suite: 440 passed, 1
  failed. The failure is
  `test_seat_bids.py::test_timed_bids_stay_near_the_allowance` (about
  16-19 ms per seat-bids bid against a 6 ms limit), which this change does
  not touch.
- Not done: two-card hands (the user wants the same logic there; play
  matters, so it needs a design decision). Missing evidence: no tournament
  or confirmation yet. Candidate spec:
  `ranges:field=previous,models=ohhell/models/selfplay_bowers`. Not promoted.

### Invalid: v2 at 10 ms against v2 at 2 ms (12 November 2026)

**No verdict; the run measured nothing about time.** A time-only test: the
candidate is v2 unchanged, with `--candidate-time-ms 10` (new option) giving
it alone 10 ms per turn while the baseline and leaning versions keep 2 ms.

- Run by the user: `python -m ohhell.tournament "hybrid:field=previous,models=ohhell/models/selfplay_bowers,time_ms=2" --candidate-time-ms 10 --label "v2 at 10 ms vs 2 ms"`.
  Bowers, down/blind/up, opening hand 10, baseline v2, seed 352414777,
  smallest gain 1.0. Fingerprints match v2 (`bid.npz` a84e9315...,
  `play.npz` 8cb5a064...). Stopped by the user after 115 sets / 537 games;
  no summary `.json`, rows in `runs/gates/v2_at_10_ms_vs_2_ms.jsonl`.
- `ohhell/native/core.c` had been edited at 15:03, a minute before the run,
  and no DLL existed for its new hash (`core_3ff24097d769aa7b.dll`). The
  loader silently fell back to Python, so both bots took about 190 ms per
  turn and neither time limit applied. Edge -4.19 +/- 8.49 points per game,
  which says nothing about extra time.
- Fix: built the DLL (`python -m ohhell.native`); a check afterwards measured
  about 11 ms per turn for the candidate and 3.5 ms for the baseline. The
  tournament now refuses to start when the native simulator is unavailable.
  Rerun under a new label to get a real result.

### No gain: simulation-only bids with estimate 3, retrained play net, 10 ms (12 November 2026)

**Tournament verdict: no gain; keep v2.** Leans negative at every table
size. Do not rerun this bundle unchanged.

- Run by the user: `python -m ohhell.tournament "hybrid:field=previous,estimate=3,bid_blend=0,bid_sigma=0.67,models=ohhell/models/sim_bids" --label "Simulation bids, estimate 3" --time-ms 10`.
  Bowers, down/blind/up, opening hand 10 (fewer at 6+ players), every
  simulating bot at 10 ms, baseline v2, seed 103544560, smallest gain 1.0.
  Ran all 1,300 sets / 6,100 games of the main stage in 1,016 s; no repeat
  stage. Fingerprints in `.meta` match those below (`trick_estimate3.json`
  20971772..., `sim_bids/play.npz` 5a0060eb..., v2 `bid.npz` a84e9315...,
  `play.npz` 8cb5a064...). Results: `runs/gates/Simulation_bids_estimate_3.json`.
- Edge **-2.36 +/- 2.37 points per game**; win-share edge -0.019 +/- 0.019.
  By size: 3p -0.10 +/- 8.41 (200 sets), 4p -2.61 +/- 3.47 (600),
  5p -2.52 +/- 4.47 (200), 6p -4.02 +/- 4.71 (100), 7p -1.96 +/- 3.29 (100),
  8p -3.80 +/- 2.45 (100). Mean score 125.9 against v2's 128.4; 11.8 against
  11.6 ms per turn.
- The 4-player arena smoke check (+4.1 +/- 3.5, all-v2 tables) did not carry
  over to the tournament's mixed lobby (4p -2.6 +/- 3.5). Both are near
  noise; the gap between them is about 6.7 +/- 4.9.
- The bundle combined five changes, so this does not show which part hurt.
  Untested explanations: dropping the network from bids loses information
  it supplies even at 10 ms; sigma 0.67 makes inference too confident
  against the lobby's leaning versions and human-like players (fitted as one
  pooled spread, while their mean gaps differ by member, -0.39 to +0.40
  tricks); the play network, retrained at 2 ms, fits 10 ms play less well.
  Single changes worth testing, if any: v2 at 10 ms with only
  `bid_blend=0`, and v2 with only `estimate=3,bid_sigma=0.67`.

The original preparation notes follow.

**Combined candidate, not a single-change test.** At the user's request this
applies every fix proposed for the `estimate=2` bidding path at once, bids
from the simulation alone, and retrains the play network around the new bids
("the play may have been compensating for underbids"). The tournament is to
run at **10 ms per turn for every bot** (`--time-ms 10`). A result therefore
says whether the bundle beats v2 at 10 ms, not which part helped.

Candidate spec (the tournament appends `time_ms=10`):
`hybrid:field=previous,estimate=3,bid_blend=0,bid_sigma=0.67,models=ohhell/models/sim_bids`

What changed against v2 (`hybrid:field=previous,models=ohhell/models/selfplay_bowers`):

1. Quick trick estimate version 3 (`estimate=3`; `ohhell/estimate.py`,
   C port in `card_win_probs2`, coefficients `ohhell/models/trick_estimate3.json`).
   Version 2 and its file are untouched: its weights are zero-padded for the
   new features and it has no spread, so it gives exactly what it did.
   - Fitted on its own tables. `calibrate_estimate --bidders` sets which
     estimate the calibration table bids with. Pass 1: bidders version 2
     (seed 11, test seed 1000011). Pass 2, kept: bidders version 3 from pass
     1 (seed 12, test seed 1000012). 100,000 train and 30,000 test rounds
     each; pass 2 barely moved from pass 1. Reports in `runs/estimate3/`.
   - New side-card features (`estimate.DISCARDS`): other side suits void (1)
     or singleton (0.5), times top and players; last seat x first card, first
     x players, both x hand size; last x logit_top and x hand size. The first
     two attempts (discards; last x first) left the dealer's Aces unchanged:
     an Ace's logit_top saturates near 9.2, so only a seat x logit_top term
     separates it from the seat's other top cards.
   - Fitted spread: the trick distribution an imagined bidder chooses from is
     raised to 1/T and renormalised, T = 0.9406 - 0.1188 h/10 + 0.0063 n/8.
     T < 1, so the count is slightly **sharper** than independent cards, not
     flatter. Train NLL 0.7616 -> 0.7589.
   - Test (pass 2, bidders v3): bias +0.003 tricks, RMSE 0.630, trick-count
     NLL 0.762, card Brier 0.0940 (rule of thumb: -0.201, 0.760, 0.899,
     0.1037). 10 cards: bias -0.08, table-sum gap -0.28.
   - Still a known limit: side Aces in 7+ card rounds are overrated in the
     last seat by 4-10 points (2p 89.8% against 79.4% actual, 3p 83.2/75.2,
     4p 77.8/73.6, 5p 72.5/67.6, 6+ 57.6/51.9) and when leading at 6+
     players (61.0/55.2). Pass 1 before the seat features: 9-12 points. The rest looks like a
     table effect (the dealer rule forcing the dealer's target), which no
     per-card feature of one hand can see.
   - Bug found and fixed while porting: C `bid_estimate_probs` tested
     `estimate == 2`, so version 3 silently fell back to the rule of thumb
     for imagined bids and inference. It is now `>= 2`. New export
     `oh_heuristic_bid` and `tests/test_estimate.py::test_native_imagined_bids_match_python`
     check imagined bids C against Python for versions 1-3 (exact match).
2. Inference sigma fitted, `bid_sigma=0.67` (overrides every kind's 1.4).
   `python -m ohhell.learn.bid_sigma --rounds 6000 --time-ms 10 --out runs/estimate3/lobby_bids.json`
   deals rounds at the tournament's table sizes and draws the lobby as the
   tournament does (v2 standing in for the candidate seat). Over 22,153
   opponent bids, RMS of bid - estimate is 0.761 (v1 estimate, mean +0.247),
   0.667 (v2, -0.020) and 0.666 (v3, -0.021). Per member, v3: v2 0.526,
   skilled 0.621, casual 0.819, novice 0.995, leaning 0.54-0.68. The
   effective sample of one block of 256 drawn deals falls from 78% (sigma
   1.4) to 43% (sigma 0.67), 5% at the 10th percentile; `oversample` is
   left at 4 because 10 ms gives about 23 blocks per bid turn.
3. Simulation-only bids: `bid_blend=0` returns the simulated trick odds
   unblended and skips the bid network call.
4. Play network with bower inputs, retrained by self-play under these bids.
   The play layout gains four inputs (`PLAY_BOWER_INPUTS = 245`): a one-hot
   of the slot holding trump's colour partner suit, and whether the candidate
   card is the left bower (before, the left bower was a Jack in an unmarked
   side-suit slot). The network was widened from v2's with zero weights,
   then trained with
   `python -m ohhell.learn.selfplay --labels search --freeze-bid --play-inputs 245 --spec "hybrid:field=previous,estimate=3,bid_blend=0,bid_sigma=0.67,time_ms=2,models={folder}" --start ohhell/models/selfplay_bowers --out runs/selfplay/sim_bids --iterations 3 --rounds 150000 --epochs 4 --seed 5 --eval-games 300 --workers 19`
   (1,729 s; log `runs/selfplay/sim_bids.log`). Generation and labels ran at
   2 ms (labels use a fixed 128 deals per card), not 10 ms. About 635k play
   rows per iteration; held-out Huber 0.0240, 0.0225, 0.0208. Iteration
   checks against the widened start (both bidding as the candidate, 2 ms,
   4 players, 300 games): +3.6 +/- 4.6, -1.6 +/- 4.8, -1.6 +/- 4.4, all
   noise. Iteration 3 was kept (most training, lowest loss), not the best
   check. The bid network is v2's and is unused by this candidate.
   SHA-256: `ohhell/models/sim_bids/play.npz`
   5a0060ebe1b71b87f82d31a0844fc93e17b3bf86c4e16af77e34aba2642b5672,
   `bid.npz` a84e9315...e29d (v2's), `trick_estimate3.json`
   20971772043ad011255786c70128c195eeb193423d64c1b525ce85c5891aee7e.

Not changed, deliberately:
- Rollouts still play our own seat with the rule-based player. Exact endgames
  inside bid play-outs were considered but not tried: the solver returns
  points, not tricks, and the removed `exact` bot suggests perfect-information
  play-outs mislead.
- Opponents still ignore our bid. `react=1` (inconclusive on v1) and
  `seatbids` (smoke -2.7 +/- 4.1, 10x slower) already tested both ideas.
- The bid network's own left-bower encoding: unused here, since
  `bid_blend=0`.

Cost at 10 ms (120 random bid spots, 3-8 players): median deals per bid turn
v2 1,670 (p10 961), candidate 1,470 (p10 833). With `oversample=8`, 833.
v2 at 2 ms: 321.

Arena smoke check (not the tournament): `runs/estimate3/smoke.py`, candidate
against v2, both 10 ms, 4 players, opening 10, down/blind/up, 300 seeds from
700000: **+4.1 +/- 3.5** points per game (142.0 against 137.8), win share
25.5%, 7.3 ms per decision. Positive and just outside noise. Earlier smoke
checks on these seeds were at 2 ms, so they are not directly comparable.

Tests: full suite 441 passed, 1 failed under load:
`test_seat_bids.py::test_timed_bids_stay_near_the_allowance`. It ran while
the smoke check held 19 cores and passed on an idle rerun (15/15). New or
updated: `tests/test_estimate.py` (v3 C/Python agreement, padding, spread,
imagined bids), `tests/test_bower_features.py`, `tests/test_selfplay.py`,
`tests/test_tournament.py` (fingerprints both estimate files). Native core
`core_7407804b3b6a17f5.dll`; intermediate DLLs from this work were deleted.

Run as proposed (result above):
`python -m ohhell.tournament "hybrid:field=previous,estimate=3,bid_blend=0,bid_sigma=0.67,models=ohhell/models/sim_bids" --label "Simulation bids, estimate 3" --time-ms 10`
(seed from the label: 103544560). Not promoted.

### Prepared candidate: card-count blend between simulation and networks (12 November 2026)

User's idea. Single change against v2: the networks' weight scales with card
count, so small hands trust the simulation and large hands lean on the
networks. New opt-in `HybridBot` setting `blend_pivot` (default 0 = off, v2
unchanged). Scale = `(cards - 1) / (blend_pivot - 1)`, floored at 0:

- Bids, cards = hand size: network share `min(1, 0.5 * scale)`. With
  `blend_pivot=6`: 0 at 1 card (bid network not called), 0.2 at 3, 0.5 at 6
  (v2's fixed value), 0.9 at 10.
- Card play, cards = cards now in our hand: `prior_worlds` = 16 x scale
  against the nominal `worlds=32`. 0 at 1 card, 16 at 6 (v2), 28.8 at 10.
  Exact endgames (12 or fewer cards left on the table) still ignore the
  network, so for small hands the play change only matters at 5+ players.
- Blind 0 uses the exact blind odds and is unaffected. Networks, estimate,
  inference, shortlist and time allowance are v2's. `bid_sigma` and the 50%
  blend were tuned together, so the pivot value 6 is a guess, not a fit.

Candidate spec (the tournament appends `time_ms=2`):
`hybrid:field=previous,models=ohhell/models/selfplay_bowers,blend_pivot=6`.
Networks: v2's `selfplay_bowers` `bid.npz`
a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d,
`play.npz` 8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.

Arena smoke check (not the tournament): candidate against v2, both 2 ms,
4 players, opening 10, down/blind/up, 300 seeds from 710000:
**-1.4 +/- 4.5** points per game (120.8 against 122.3), win share 25.2%,
3.7 ms per decision. Noise.

Tests: `tests/test_blend_pivot.py` (14): the scale, bid share by hand size,
exact equality with v2 at the pivot hand size, no bid network call at one
card, the play prior by cards in hand, and full 3- and 8-player games with
blind 0. Full suite 454 passed, 1 failed: the known
`test_seat_bids.py::test_timed_bids_stay_near_the_allowance`, which this
change does not touch.

Missing evidence: no tournament or confirmation yet. Proposed run:
`python -m ohhell.tournament "hybrid:field=previous,models=ohhell/models/selfplay_bowers,blend_pivot=6" --label "Card-count blend, pivot 6"`
(seed from the label: 292132229). Not promoted.

**Result (12 November 2026): INCONCLUSIVE, keep v2.** Ran the proposed
command unchanged: 2 ms per turn for every bot, down/blind/up from 10, seed
292132229, 1,300 sets (6,100 games, 350 s), table split 3p 2, 4p 6, 5p 2,
6p 1, 7p 1, 8p 1. The fingerprints match v2's above. The main gate ran to
the set limit; there was no confirmation stage, since that only follows a
pass.

- Edge against v2: **-0.54 +/- 2.41** points per game (115.90 against
  116.40); win share -0.010 +/- 0.020; 3.89 against 3.91 ms per turn.
- By table: 3p -4.22 (200 sets), 4p +1.61 (600), 5p -2.70 (200), 6p -3.23,
  7p +0.61, 8p -0.22 (100 each). Every table is within noise.
- By hand size, points per round: 10 +0.18, 9 +0.75, 8 -0.19, 7 -0.29,
  6 +0.13, 5 -0.28, 4 -0.19, 3 -0.08, 2 +0.03, 1 -0.20, blind +0.08. The
  small hands, where the candidate trusts the simulation more, show no
  gain (1-5 cards sum to -0.72 per game). The 6-card and blind rows
  should be zero by construction and came out at +0.13 and +0.08, which
  shows how much these rows move from noise and from decisions
  that diverged earlier in the game.
- The interval spans roughly -3 to +2 points, so a gain as large as the 1
  point minimum is not ruled out, but there is no support for it.
  Do not re-run the unchanged candidate. A new hypothesis would need a fitted
  schedule (for example, choosing each hand size's blend by held-out bid
  calibration of simulation against network) rather than this guessed line.
- Sources: `runs/gates/Card-count_blend_pivot_6.json`, `.jsonl`, `.meta`.

### Stopped run: simulation-only bids with v2's networks, 2 ms (12 November 2026)

Single change against v2: `bid_blend=0` (bids from the simulation alone, no
bid network). Estimate, `bid_sigma`, play network and time are v2's. This is
the "v2 with `bid_blend=0` alone" follow-up proposed under the combined
estimate-3 candidate above, but at 2 ms rather than 10 ms. Started by the
user:
`python -m ohhell.tournament "hybrid:field=previous,models=ohhell/models/selfplay_bowers,bid_blend=0" --label "Simulation-only bids" --time-ms 2 --exit-when-done`
(seed from the label). Stopped by the user after 76 sets because it was
clearly losing, so there is no verdict file, only
`runs/gates/Simulation-only_bids.jsonl` and `.meta`.

- Edge from the 76 saved sets: **-17.1 +/- 9.4** points per game (1.96 x
  set SD / sqrt(sets), not the tournament's sequential gate). By table:
  3p -15.8 (12 sets), 4p -27.4 (35), 5p -0.3 (12), 6p +0.6 (6),
  7p -9.8 (6), 8p -18.8 (5).
- Caveat: it shared the machine with another tournament ("v2 at 10 ms vs
  2 ms") and the blend check, about 55 workers on about 19 cores. Under a
  2 ms limit that cuts simulated deals per bid, which hurts a
  simulation-only bidder most. The size of the loss is therefore not
  clean, though its direction is consistent with the card-count blend run
  (small hands, which trust the simulation more, showed no gain).
- Verdict: keep v2. At 2 ms the bid network is pulling its weight; do not
  drop it without a new reason. The combined estimate-3 / 10 ms candidate is
  a different test and remains unrun.

### Interrupted: v2 with simulation-only bids at 2 ms (12 November 2026)

Single change against v2: `bid_blend=0` (bids from the simulation alone; the
bid network is not consulted). Spec
`hybrid:field=previous,models=ohhell/models/selfplay_bowers,bid_blend=0`,
2 ms for every bot, baseline v2, seed from the label "Simulation-only bids",
weights v2's (`bid.npz` a84e9315..., `play.npz` 8cb5a064...).

- Started 15:03 with `--exit-when-done`. A second tournament from another
  session ("v2 at 10 ms vs 2 ms") started at 15:04 on the same machine, and
  this run's process ended at about 15:24 with no verdict, no summary `.json`
  and no error in its console log, after **76 of 1,300 sets**.
- Both runs shared the CPU, so every bot completed fewer deals than a normal
  2 ms turn. Both sides were limited equally, but the settings are not v2's
  usual ones.
- From `runs/gates/Simulation-only_bids.jsonl`: edge **-17.1 +/- 9.4 points
  per game** (plain 95% over 76 sets, not the planned margin). It suggests
  the bid network matters at 2 ms but is not a verdict. Rerun alone under a
  fresh label before relying on it.

## Current decision

The default is **v2 Self-play bowers nets**,
`hybrid:field=previous,time_ms=2`, with its networks in the default folder
`ohhell/models/` (promoted 11 November 2026; formerly `models=ohhell/models/selfplay_bowers`). It is v1's hybrid search with networks retrained
under bowers and improved by self-play. It passed the main stage and
confirmation against v1 under bowers; see the self-play section above. New
candidates should be gated against v2.

Historical note: under Ace-high rules, none of the experiments below
established a repeatable improvement over v1. In particular, do not propose
the unchanged candidate bidding checkpoint or unchanged adaptive card search
as fresh improvements.

An inconclusive result means insufficient evidence, not demonstrated equality.
Revisit an idea only with a specific new reason, such as different training,
a diagnosed implementation problem, or a deliberately planned new budget.
Record what differs from the attempt below before spending another run on it.

## Current full-game experiments

Both use the down/blind/up rules, v1 as baseline, and 2 ms per simulating bot
turn. Tables vary from 3 to 8 players, weighted towards 4, with passive and
aggressive baseline variants plus skilled, casual and novice opponents.
Each set rotates through every seat on the same preset game. Results are
checked at ten planned points; a main-stage success must pass a separate
confirmation on unseen deals. The figures below are saved main-stage edges
and the tournament's reported margins, in points per game.

| Change | Exact candidate spec (before tournament adds `time_ms=2`) | Main result | Sets / games | Seed | Verdict |
| --- | --- | --- | --- | --- | --- |
| Candidate bidding checkpoint only | `hybrid:field=previous,bid_models=ohhell/models/candidate` | +1.042 +/- 2.360 | 1,300 / 6,100 | 605225083 | Inconclusive; no confirmation stage |
| Adaptive all-card search | `adaptive:field=previous` | +0.934 +/- 2.323 | 1,300 / 6,100 | 632140155 | Inconclusive; no confirmation stage |

### Candidate bidding checkpoint only

Changed only the bid network to the existing candidate weights. Card-play
weights, inference, blending, and simulation strategy stayed at v1 settings.
Added separate `bid_models` and `play_models` selection to make this isolation
possible; `models` still selects both. This was a test of an existing checkpoint,
not a fresh retraining run. No established score or speed gain:
candidate/baseline means were 2.330/2.330 ms per turn; win-share edge was
+0.0170 +/- 0.0196.

Sources: `runs/gates/Candidate_bids_full_game.json`, `.jsonl`, and `.meta`.
The manifest records the following SHA-256 fingerprints:

| Weight file | SHA-256 |
| --- | --- |
| Candidate `bid.npz` | `8dd7a91920241d0c4713a65e68e9d5ad85e9e5c61baf12c8da0855c0b0e4002d` |
| Current `bid.npz` | `ed3ab2e4ef8cc5cb78a3dd1fdf9a29157a3848568fd078c97d874cd9f174f12c` |
| Current `play.npz` | `be178e28e8d9fcc94f7ae707d20b26bd739d585f386bfb7bc6a6b8ac664d7580` |

### Adaptive all-card search

Implemented in `ohhell/bots/adaptive.py`. Explores every legal card instead
of the model's top-four shortlist, uses successive halving to focus remaining
work on contenders, and weights the model prior against actual accumulated
sample counts. Bidding and exact endgames use the current hybrid code.
Defaults include `batch_worlds=4`, `prior_worlds=16`, and inference enabled;
the experiment used current model weights, not the candidate bid checkpoint.

Candidate/baseline means were 2.223/2.338 ms per turn; win-share edge was
+0.0109 +/- 0.0197. A separate 60-decision timing check before the tournament
also looked faster, but neither timing result establishes stronger play.
Do not describe this implementation as an achieved strength upgrade.

Sources: `runs/gates/Adaptive_all-card_search.json`, `.jsonl`, and `.meta`.
The manifest fingerprints the same current bid/play files listed above.

## Prepared candidate: evidence-weighted bidding (10 November 2026)

New hypothesis: hybrid always assigns 50% weight to the bid network even
though a timed turn may finish substantially more or fewer than `bid_worlds`
deals. `evidence:field=previous` instead uses
`model_weight = bid_prior_worlds / (bid_prior_worlds + completed_worlds)`;
the default `bid_prior_worlds=64` reproduces the current blend at exactly 64
completed worlds. Both backends now expose completed bid samples. The final
candidate preserves the current bid shortlist, model checkpoints, inference,
card play, exact endgames and blind bidding. It differs from the earlier
candidate-checkpoint and adaptive-card-search experiments.

Preliminary arena check of the final implementation:

```powershell
.venv\Scripts\python.exe -m ohhell.arena "evidence:field=previous,time_ms=2" "hybrid:field=previous,time_ms=2" --games 40 --players 4 --hand 6 --workers 6
```

Four players, starting hand 6, down/blind/up, dealer restriction enabled,
deal seeds 0 through 39, default current bid/play weights with the SHA-256
fingerprints recorded above. Each seed had a control game and four candidate
seat rotations (160 candidate games plus 40 controls). Result: **-2.8 +/- 5.8
points per game**, candidate/control means 58.2/61.0, candidate win share 25.5%,
1.4 ms per candidate decision. The interval is the arena's ordinary approximate
95% interval over 40 seed averages, not the tournament's sequential gate.
Tests briefly ran concurrently, so timing should not be treated as a benchmark.
This is inconclusive and does not establish a strength gain. No full tournament
or confirmation has been run; retain v1.

An earlier development version also added the model's two favourite legal
bids to the heuristic shortlist. Under the same arena command/settings it
returned -3.0 +/- 4.5 points, means 57.8/60.8, win share 23.6%, and 1.3 ms per
decision. That combined change was removed to isolate weighting; its result
does not evaluate the final implementation. The console summaries, rather
than saved arena records, are the available evidence for both checks.

Run the final candidate independently of the candidate bid checkpoint:

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "evidence:field=previous" --label "Evidence weighted bids" --workers 6 --port 8768
```

Dashboard: <http://localhost:8768/>. Use a fresh label for changed settings or
code. No model was overwritten and no candidate was promoted. Tests cover
actual sample weighting, equality with the current 50% blend at 64 worlds,
unchanged shortlist/card play, dealer legality, both simulator backends,
complete games including blind rounds, and checkpoint fingerprinting.

## Prepared candidate: visible-history planning (10 November 2026)

New structural hypothesis: training targets and flat simulations largely value
a fixed rule-based continuation, which can undervalue a move that only works
when followed by a different later decision. `planning:field=previous` replaces
earlier card search with a tree of publicly played card histories shared across
all sampled deals. Our decisions use UCB exploration over actual final round
points. Opponents retain the existing population policy. Each simulation
expands one untried decision and then rolls out; histories revisited on later
samples can learn better continuations. A visible response can change the
continuation, but a concealed hand or sampled kind cannot select a separate
tree node. The tree is rebuilt on each real turn.

The final candidate retains hybrid bidding, blind handling, belief inference,
current bid/play checkpoints and the existing exact endgame solver. Every legal
root card is considered outside those endgames; the model orders first visits
only, with no model pseudo-reward or final model/simulation blend. The root
choice is the most visited card, breaking ties by mean reward. Defaults are
`exploration=30` points, `tree_batch=8`, at most 32,768 nodes per turn and
`worlds * legal_cards` rollouts without a time limit. Capacity exhaustion falls
back to rule-based continuations. No evidence-bidding or checkpoint change is
combined with this candidate. This is a larger card-search replacement, not
the previously attempted adaptive root-action allocation.

Python and C implementations are provided. The native core was built locally
as `core_c759ddc26fb1648c.dll`. Windows now builds source-hashed DLL names
because the old `core.dll` was locked; already-loaded processes are unaffected.
After a fresh checkout or source edit, run
`.venv\Scripts\python.exe -m ohhell.native` before a timed comparison.

Validation of the final implementation: **330 tests passed**, including both
backends. A concrete known-deal regression has flat rollouts value the three
root cards at -20/-10/-10; tree search learns a later continuation producing
positive mean reward on a different root move. A separate synthetic test
prefers probing when a public response reveals useful information and prefers
the safe action when the hidden outcome is not revealed. Additional checks
cover sample budgets, deadlines, search reset, capacity exhaustion, unchanged
bidding/endgames, full blind/down/up games and checkpoint fingerprints.
These verify mechanics, not general playing strength.

Full-game smoke checks completed with one planning seat versus hybrid seats at
3 players/opening 10, 4 players/opening 10 and 8 players/opening 6, all seed 17,
dealer restriction enabled and 2 ms per turn. No score comparison or confidence
claim is inferred from these three games. A few initial-turn diagnostics also
confirmed native search reached later choices within the soft 2 ms allowance;
the counts depend on host load and are not a performance benchmark. Current
model SHA-256 hashes remain the current bid/play hashes recorded above.

No long arena or tournament comparison has been run. The user should run:

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "planning:field=previous" --label "Information set planning" --workers 6 --port 8770
```

Dashboard: <http://localhost:8770/>. Run sequentially with other comparisons,
use a fresh label, and require main plus unseen-deal confirmation before
promotion. The 2 ms budget, sparse reuse of future histories, fallback rollout
quality and approximate opponent policy may limit the benefit. No strength
gain has been established; v1 remains the default.

## Prepared candidate: visible-history planning (10 November 2026)

New hypothesis: v1 scores root cards using fixed rule-based continuations;
it can miss a useful probe or a line requiring different later choices.
`planning:field=previous` searches one tree shared across sampled deals, with
edges keyed only by publicly played cards. All legal root cards are explored;
the existing play model orders first exploration only. UCB exploration is 30
points, `tree_batch=8`, with at most 32,768 nodes per turn. It selects the most
visited root card, breaking ties on mean score. Both Python and native paths
are available, and the native core was built locally.

This work completed validation of the planning implementation already in the
working tree and restored v1 exact-endgame handling. Bidding, blind rounds,
opponent beliefs, checkpoints and rollout policy retain v1 behaviour. This is
a search-algorithm replacement with multiple internal differences; it does
not identify the effect of each internal choice. No evidence-weighted bidding,
candidate checkpoint, `react=1`, or inference-disabled change is combined.

Completed **small smoke check**, not a tournament:

```powershell
.venv\Scripts\python.exe -m ohhell.arena "planning:field=previous,time_ms=2" "hybrid:field=previous,time_ms=2" --games 8 --players 4 --hand 6 --workers 1 --first-seed 100
```

Four players, starting hand 6, full down/blind/up, dealer restriction on,
seeds 100â€“107, both bots given 2 ms soft turn limits. One control and four
candidate seat rotations per seed: 32 candidate games plus eight controls.
Result **+2.8 +/- 7.8 points per game**, candidate/control means 65.9/63.1,
candidate win share 28.1%, mean candidate decision 1.3 ms (includes forced
decisions). The interval is the arena's ordinary approximate 95% interval over
eight seed averages, not the tournament's sequential gate. The sample is far
too small to establish a gain or a speed improvement. Tests had completed
before this smoke check; unrelated external CPU load was not measured.

Saved evidence: `runs/gates/Planning_information_smoke_2026-11-10.json`, including
rules, seed list, report, native DLL path and SHA-256 fingerprints for both
model checkpoints and the Python planning/C core sources. It uses the current
bid/play model hashes listed above. There is no completed main gate or unseen
confirmation for this candidate. **Inconclusive; keep v1.**

Verification: **327 tests passed**, including an information-value test where
probing wins only when the observed response reveals useful information,
shared branches across hidden states, all-card exploration, sample accounting,
deadlines, unchanged bids/endgames, full blind-inclusive games in both backends,
tree isolation, and tournament checkpoint fingerprinting. Tests do not prove
perfect bidding, optimal aggression, guaranteed bailouts, or a strength gain.
Full audit: `BOT_STRATEGY_ANALYSIS.md`.

User-run full tournament:

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "planning:field=previous" --label "Planning information v1" --workers 6 --port 8770
```

Dashboard: <http://localhost:8770/>. Use a fresh label after any code/settings
change; avoid concurrent comparisons. No promotion or model overwrite.

## Independent component case framework (10 November 2026)

Added `ohhell.learn.cases` and usage guide `BOT_CASES.md`. No bot strategy,
checkpoint or default changed. This is diagnostic/data infrastructure, not a
strength experiment or a promotion. Labels enumerate legal root actions and
finish each saved round using frozen observation-only policies; they are not
perfect-play answers. Bidding and play have separate evaluation and cached
feature exports; related deal snapshots share one train/validation/test split.

Prepared `runs/cases/predetermined-v1.jsonl`, seed 20261110, eight repeats,
2â€“8 players, hand sizes 1â€“10 where legal plus blind 0, standard scoring and
dealer restriction on. Four bidding profiles per deal. Contains 54,560
decisions (10,240 bids / 44,320 plays) from 544 independent deals, with
45,648/3,996/4,916 train/validation/test snapshots. This is independent-round
coverage, not cumulative down/blind/up evaluation. Full-bank labeling and
training have not run. Manifests preserve hashes/settings and report coverage.

Completed CLI smoke run: seed 20261110, one repeat, maximum hand 1, 560 cases.
Labels use our heuristic continuation and mixed opponents, two policy samples
per action on each saved hidden deal; no time allowance applies to these
policies. Test-split heuristic-versus-identical-heuristic reports both give
zero paired edge over 40 bid and 40 play snapshots (three deals). This checks
identity, not strength. Training exports contain 388 bid and 208 play rows;
blind cases are excluded from existing neural feature encodings. Reports,
labels, manifests and exports are saved under `runs/cases/framework-smoke*`.
Current bid/play files retain the SHA-256 fingerprints recorded above;
smoke manifests also record source hashes. Nine framework tests cover legal
replay, data splits, deterministic labels, observation privacy, exact outcomes,
a zero-bid tactical choice, exports, identity comparison and manifest integrity.
Full regression verification: **347 tests passed**. No training speed benchmark,
full-game comparison or confirmation was run.

## Prepared candidate: floating point NNUE card evaluation (10 November 2026)

New hypothesis: stable sparse features and incremental first-layer updates
can reduce card-value inference cost while retaining useful v1 evaluations.
`nnue:field=previous` replaces only the play evaluator/checkpoint in v1's
existing hybrid decision pipeline. It uses 484 stable inputs and layers
128/32/1, float32 accumulators with periodic rebuilds, shared root-card
branches and per-bot caches. It is not quantized and is not used within C
rollout continuations. Bidding, blind policy, inference, simulation settings,
top-four shortlist, blending and exact endgames retain v1 behaviour.
This combines a new architecture, representation and distilled checkpoint;
it is not a single-factor test isolating each of those internal changes.

Completed training: seed 20261110, 20 epochs, batch 1024, Adam 0.001, targets
from current v1 play network on the predetermined-v1 bank. Choice-only action
rows: 71,960 training / 6,731 validation / 8,048 test, from 368/27/37 deals.
Splits are deal-separated; checkpoint chosen using validation only. Preparation
6.073 seconds, fitting 7.806 seconds on this host. Validation RMSE 10.200
points and teacher-best agreement 45.90%; unseen test RMSE 11.204 and agreement
48.22%, mean teacher-score selection gap 1.322 points across 2,308 positions.
These are teacher-fidelity metrics, not actual score edges or proof of strength.

Saved candidate: `ohhell/models/nnue_candidate/play.npz`, SHA-256
`d204435159cb7b31511830c43b8a873fd19d9a378a8fb0cfa8c8de8c947929bb`.
Its `training.json` includes settings, bank/source hashes and teacher hashes.
Current bid/play hashes remain those recorded above; no model was overwritten.

Completed diagnostics: 256 unseen choice positions sampled with seed 20261110,
maximum incremental/dense error 0.000084 points. Five passes measured isolated
card-value calls at 89.66 us for current MLP, 50.36 us for incremental NNUE,
57.62 us for identical NNUE weights evaluated densely. Includes features;
different widths/representations confound MLP comparison. Not a turn/search
benchmark, and external CPU load was not measured. No general speed claim.

Three full-game smoke completions: one trained NNUE seat versus v1 seats,
seed 17, 3 players/opening 10, 4/opening 10, 8/opening 6, dealer restriction on,
standard scoring, down/blind/up, 2 ms soft turn limits. Scores in
`runs/cases/nnue-v1-diagnostics.json` are completion evidence only. Nine new
tests validate accumulator equivalence, privacy, isolation, training leakage,
both backends and unchanged bidding/endgames. Full regression run: **356 tests
passed**. No arena/tournament comparison
or unseen-deal confirmation has run. **Experimental; retain v1.**

User-run command:

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "nnue:field=previous" --label "NNUE float v1" --workers 6 --port 8772
```

Dashboard: <http://localhost:8772/>. Run comparisons sequentially, use fresh
labels for changed candidates, and require main plus unseen-deal confirmation.
Full implementation, training and diagnostic details: `NNUE_EXPERIMENT.md`.

## Earlier experiments (historical results)

These predate the down/blind/up change. Their JSON has no round-format marker;
do not treat them as validation on the current full-game rules. The oldest
three also used an earlier fixed six-player scheme without the current
confirmation gate. Margins from different schemes are not interchangeable.

| Run / source JSON in `runs/gates/` | Change and baseline | Recorded result | Disposition |
| --- | --- | --- | --- |
| `reactive-rollouts.json` | `react=1` added to v1 | +0.297 +/- 0.898; 3,000 sets / 18,000 games; inconclusive | No established gain from making rollout players respond to other seats' bids |
| `no-belief-inference.json` | `inference=0` added to v1 | +1.543 +/- 1.540; 995 sets / 5,970 games; legacy verdict `better` | Historical signal only; later confirmation failed |
| `no-endgame-solver.json` | `endgame_cards=0` against an inference-disabled baseline | -0.594 +/- 1.593; 867 sets / 5,202 games; `no gain` | Keep exact endgames; this was not a comparison against unchanged v1 |
| `2026-11-10_No_belief_inference.json` | `hybrid:field=previous,inference=0` against v1; seed 13011509 | +0.689 +/- 1.671; 1,300 sets / 6,100 games; inconclusive | No promotion |
| `No_belief_inference_current_scheme.json` | Same inference-disabled candidate against v1; seed 791617248 | Main +1.677 +/- 1.644 over 1,300 sets; confirmation +0.127 +/- 1.576 over another 1,300 sets; 12,200 games total; **unconfirmed** | Main-stage advantage did not repeat; keep inference enabled |

In the last inference-disabled run, candidate/baseline means were
2.112/2.293 ms per turn. The user valued the reduced time, but the score gain
was unconfirmed. It remains a possible speed tradeoff, not a demonstrated
strength improvement. There is no saved completed inference-disabled run
under the new full-game rules in the results inspected for this log.

## Retained changes and limits of the evidence

- The corrected game sequence and blind-card privacy are implemented in the
  rules, engine, UI, and arena. They are required rules, not failed experiments.
- Separate model selection, model fingerprinting on resume, and the adaptive
  candidate remain available. Their presence does not mean a promotion.
- Regression checks after adaptive search: **293 tests passed**. These verify
  legality, privacy, model isolation, sampling, and deadline handling, not strength.
- Candidate play-only weights, both candidate networks together, and combinations
  such as adaptive search plus disabled inference have **no completed saved
  evaluation in this record**. Do not label them successful or already disproven.
- `runs/` is gitignored. This tracked note preserves conclusions when raw local
  artifacts are unavailable. Preserve raw artifacts locally and add future
  results here rather than relying only on chat history.
- Compare one intentional change at a time, give bots the same turn allowance,
  and avoid concurrent tournaments competing for CPU. Timed decisions are not
  fully deterministic even with the same deal seed, and turn limits are soft.
- Use fresh run labels after rule, model, or implementation changes. Existing
  resume manifests check rules/settings and model hashes, not all source code.
- The user runs long tournaments from supplied commands and opens dashboard
  links themselves. Do not automate their PC or browser.
