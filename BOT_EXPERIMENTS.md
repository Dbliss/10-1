# Bot experiment history

Updated 20 November 2026. This is the durable record for future agents.

## Promoted without tournament: v4 one-card ranges (20 November 2026)

**User's decision; no main gate or confirmation was run.** v4 =
`ranges:field=previous,inference=0,time_ms=2`: v3 with visible one-card bids
(not blind 0) replaced by the exact range reading in `ohhell/bots/ranges.py`
(see "Prepared candidate: one-card bids read as card ranges", 12 November).
All larger hands, blind bids and card play are v3's. Game bots and the hint
advisor inherit it through `DEFAULT_SPEC` plus `Game.LONG_THINK`.

- Table `ohhell/models/one_card_ranges.json` unchanged, SHA-256
  6d8435ab71d4cc14513d4dd8207bca10058840f31de46eedfa0c59f2b57bd470.
- Example (8 players, 10 of hearts turned, holding 7 of hearts, three earlier
  zero bids): win chance 46.6%, so the rule bids 0. Seven cards beat it with
  bowers (J hearts, J diamonds, A, K, Q, 9, 8 of hearts).
- Only evidence: the 12 November arena smoke check against v2 (+0.2 to +0.5
  points per game over one-card rounds only). Missing: tournament against v3,
  confirmation on unseen deals, any test run of this change (user said no
  testing needed).

## Prepared: own-bid-only fitted formula (19 November 2026)

User requested implementation and unattended duels while away. New hypothesis:
a fitted estimator can improve our bid shortlist without changing opponent
modelling. Earlier estimate=2/3 experiments also changed imagined bidders and
inference. This is a distinct isolated experiment, not an unchanged rerun.

Candidate adds `bid_estimate=3` to the game bot: existing fitted v3 coefficients
only for our own shortlist. Range floor(E)-1 .. ceil(E)+2, capped to legal
bids and floored at certain top-trump tricks, as in the existing calibrated
formula. No new training, arbitrary bonuses, or boundary expansion combined.
Opponent bidding, inference, simulation policies, network inputs, bid blend,
card play and defaults unchanged. `bid_estimate=0` preserves existing specs.

Reported hand JH/7H/4H/JS/7S/KC/QC/2C/KD/10D, hearts 2 turned,
4 players, dealer seat 2, seat 3 bid 2, us seat 0: old E=1.932987,
shortlist 0..4; fitted v3 E=2.869128, shortlist 1..5. This does not prove
5 is better, and still leaves a finite shortlist.

Candidate `hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25,bid_estimate=3`
against the same spec without `bid_estimate=3`. Plan: 50 ms screen,
seed 2026111903; only if main and unseen-deal confirmation pass, validate
at 250 ms, seed 2026111904. Each uses 18 workers, interleaved calibrated
fixed budgets, max 60,000 main rounds, 10 O'Brien-Fleming looks, smallest
gain 2 points/game. Weighted 3..8-player down/blind/up rules, bowers,
field previous hybrid x3 / skilled / casual. Duel manifests include source,
model and coefficient SHA-256 fingerprints (added independent coefficient
fingerprinting). Runner `scripts/run_bid_estimate_experiment.py` appends every
completed result below, serves read-only progress at http://localhost:8781/,
and never promotes defaults. 50/250 ms is not evidence at 750 ms.

Verification so far: 69 relevant tests passed; 4-round fixed-budget duel
smoke passed. Includes reported-hand regression, default compatibility,
legal options, unchanged native/Python simulation for equal options,
independent opponent settings, HybridBot plumbing and fingerprints.
Full suite in .venv: 710 passed, 4 failed in 64.32 s. All four are the
already-documented stale v2-default assertions (nnue, planning x2,
search_components), unrelated to this change. Relevant tests: 69 passed.




Reported-hand diagnostic (not a strength gate): all 11 legal bids evaluated
with the unchanged game bot's simulation/network blend, 10,000 sampled deals
per seed, seeds 19031/19032/19033, same public position reconstructed from
the screenshot. Bid 4 remained best in all three: make chance 45.8-46.6%,
expected points +11.64 to +12.09. Bid 5: 31.8-33.0%, +5.94 to +6.82.
Higher bids were worse. Thus the screenshot was not a demonstrated missed
best bid. Diagnostic saved as
`runs/duels/own_bid_estimate_reported_hand.json`. The diagnostic does not
know the actual hidden hands (it uses only the observation).

## Stopped, no verdict: `endgame_noise=1,endgame_table=1` at 250 ms (19 November 2026)

Two changes tested together (noise-aware exact solve in deals with noisy
seats, and per-table thresholds `ENDGAME_BY_PLAYERS` = 3: 18, 4: 19, 5: 21,
6: 22, 7: 24, 8: 24), so even a pass would not say which one helped.
Candidate `hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25,endgame_noise=1,endgame_table=1`
against the game bot (same spec without the two flags), both `time_ms=250`,
`--timed` (equal wall clock: 1,056 vs 1,054 ms per round), field
previous x3 / skilled / casual, 18 workers, seed 212962174, max 60,000
rounds, smallest gain 2.0, source sha256 4701527c..., models bid.npz
a84e9315..., play.npz 8cb5a064..., log `runs/duels/250ms_noisy_endgame_table.jsonl`.
**Stopped by the user** at 49,746 main rounds. Last planned look (48,000):
**+1.04 points a game, gate margin +/- 2.09** (-1.05 to +3.13): neither
better nor ruled out (upper end above 2.0). At the stop: +1.12, plain 95%
+/- 1.72. Only 6.8% of rounds diverged (27% for `endgame_cards=20`): the
exact solves mostly pick the playouts' card. Projection had it ending
inconclusive (final margin about +/- 1.67; the last 12,000 rounds would
have needed about +4.2 a game). **Not promoted; no unseen-deal
confirmation.** Do not rerun unchanged; a new hypothesis (e.g. a
minimum-deals rule from `ohhell.learn.endgame_deals`, tested alone)
would justify another duel.

## Measured (diagnostic): how many deals the exact endgame needs (19 November 2026)

User hypothesis: about 100 deals per turn (the `endgame_table=1` low point)
is too few when many cards are unseen; the deals needed should grow with
the unknown cards. Tool: `ohhell/learn/endgame_deals.py` (diagnostic, not
a strength gate). Positions: heuristic bids and play to a random point
from 8 cards left up to `ENDGAME_BY_PLAYERS`, 3-8 players, hand size at
most min(10, 51 // n). Per position the candidate spec's own sampler
(`field=mix,inference=1,bid_sigma=0.76`, block = `worlds`) draws up to
2,000 deals (240 s cap per position), every legal card solved with
`solve_noisy`. Consecutive chunks of N = 10..400 deals stand for one
decision; loss = held-out reference's best value minus the value of the
chunk's choice, reported as an upper bound and a split-half lower bound,
grouped by opponent cards held, players, and cards left <=12 / >12.
Records ms per deal (under the run's own load) and deals per 250 ms.
Limits: measures sampling loss only, not exact search vs play-outs; the
reference shares the perfect-information bias; heuristic bids make bid
reading better calibrated than against the real field. Checked: native
`solve_noisy_scores` matched Python `solve_noisy` on 24 sampled deals;
2-minute smoke runs (2 workers, during the `endgame_noise` duel) gave
sensible tables and stopped before the cap. Hard cap `--minutes` <= 115.
Command: `python -m ohhell.learn.endgame_deals run --minutes 110 --workers 18`;
report: `python -m ohhell.learn.endgame_deals report runs/endgame_deals/<file>.json`.

**Run** (user, stopped by the user at 29.9 of 110 minutes, after the
`endgame_noise` duel was stopped; nothing else running): 23,340
positions, all reached 2,000 deals, seed 20261119, 18 workers,
`runs/endgame_deals/endgame_deals_20261119T131633Z.json`. Points lost per
decision against unlimited deals (upper / lower bound):

- All: N=25 0.26 / 0.24, 50 0.18 / 0.16, 100 0.12 / 0.10, 200 0.07 /
  0.05, 400 0.05 / 0.02. Doubling deals cuts the loss by about 40%.
- **Hypothesis not supported**: at a fixed N the loss is about the same
  for 0-8, 9-12, 13-16, 17-20 opponent cards held (N=100: 0.10, 0.15,
  0.12, 0.09 upper); 21+ (243 positions) slightly higher, 0.16. What grows
  with unseen cards is the cost per deal (median 0.05 ms at 0-8 to 2.8 ms
  at 21+, so about 89 deals per 250 ms at 21+).
- Table size matters more: at N=100 and >12 cards left, 3 players 0.24,
  4: 0.18, 5: 0.17, 6: 0.11, 7: 0.08, 8: 0.06 (upper).

Per game (rough, scratch script not kept): heuristic games over the real
sequence counted one seat's choices per (players, cards left) from 8 to
the threshold (3p 37, 4p 32, 5p 32, 6p 21, 7p 17, 8p 12 a game; about a
third of endgame plays are forced); each cell's loss curve (mean of the
bounds, log-log interpolation, extrapolated past 400) at 250 / median
ms-per-deal deals. Lost against unlimited deals, points a game: 3p 2.6,
4p 1.0, 5p 1.3, 6p 0.4, 7p 0.3, 8p 0.2; weighted by the duel's table mix
about **1.1**; twice the deals would recover about 0.45, four times about
0.7. Deal counts ignore sampling time, so the bot gets somewhat fewer.
Concrete follow-up (not built): early stopping for exact turns
(`stop_z` is currently skipped when exact) with `controller=bank`, to move
time from easy shallow endgame turns to deep ones; to be measured, then
duelled alone.

## Stopped: `endgame_cards=20`; prepared: noise-aware exact endgame with per-table thresholds (19 November 2026)

**`endgame_cards=20` at 250 ms** against the game bot
(`hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25`),
seed 646650478, source sha256 31f56624..., budgets 22,825 / 44,989 / 24,589
vs 25,915 / 44,705 / 33,400, load about 40% of idle during calibration.
**Stopped by the agent on the user's instruction** at 16,510 main rounds:
+0.65 points a game (plain 95% +/- 2.25; last O'Brien-Fleming look,
16,000 rounds: +0.73 +/- 4.72); 27.4% of rounds diverged. **No verdict**
(missing evidence). Not promoted. Two flaws, so do not rerun unchanged:
- With `field=mix` the exact search only ran in deals where every
  opponent was drawn skilled (prior 0.4^(n-1): 16% at 3 players, 6.4% at
  4, 0.16% at 8); every other deal was played out as before.
- The fixed endgame budget is one mean over all endgame turns (mostly
  cheap play-outs), so deep turns ran thousands of exact solves: the
  candidate averaged **7,343 ms per decision against the baseline's
  4,199** (unequal compute). Endgame candidates whose cost per deal varies
  steeply with cards left must be duelled with `--timed`.

**Exact-solve timing** (scratch script, not kept; heuristic play to each
cards-left count, one sampled deal solved over all legal cards, 24
positions per cell, all-skilled field, measured while a duel used 18 of 20
cores, so pessimistic). Median ms per deal, from where it grows: 3 players
18 cards 1.8, 20: 4.1, 21: 11, 23: 375; 4 players 19: 0.7, 20: 4.2, 23:
11, 24: 34 (p90 1.3 s); 5 players 21: 3.1, 25: 43; 6 players 22: 2.0, 27:
46; 7 players 24: 3.3, 29: 56, 30: one deal 25 s; 8 players 24: 0.8, 25:
5.3, 31: 1.5 s. A solve is not interruptible; the clock is checked between
deals. Thresholds keeping about 100 deals in a 250 ms turn:
`ENDGAME_BY_PLAYERS` = 3: 18, 4: 19, 5: 21, 6: 22, 7: 24, 8: 24
(`ohhell/bots/simulation.py`).

**Prepared candidate (user request), two changes tested together, so a
"better" would not say which one helped:**
- `endgame_noise=1`: exact search also in deals with noisy opponents.
  The mover's best line (ties against us, as before) gets weight
  1 - card_noise and the mean over every legal card gets card_noise
  (expectimax; merged touching cards weighted by how many they stand
  for). Matches how play-outs model casual/novice kinds. Our seat is
  never noisy. C `solve_noisy` (separate from `solve`, which is
  unchanged), Python `ohhell.search.solve_noisy`, diagnostic export
  `oh_solve_noisy_known`. Costs 5-30% more per deal than plain `solve`
  (at the thresholds: 50-170 deals per 250 ms turn on the `mix` field).
- `endgame_table=1`: threshold per table size from `ENDGAME_BY_PLAYERS`
  (applies to the C core, `HybridBot.play`, planning, adaptive and the
  duel's endgame calibration via `endgame_limit(n)`).
- Both default off; with them off decisions are unchanged.
- Tests: `tests/test_endgame_noise.py` (115: brute force without merging
  or memo, zero noise equals `solve`, merged-card counts, C/Python parity
  on 40 positions, threshold plumbing). Full suite 701 passed, 5 failed,
  none touching this change: 4 stale tests still expect v2 as the default
  (v3 was promoted 16 November) and a wall-clock bid-timing test failing
  under duel load. DLL `core_3c38f2cd4d712dd6.dll`.
- Duel to run (`--timed`, see the budget flaw above):
  `python -m ohhell.duel "hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25,endgame_noise=1,endgame_table=1" --baseline "hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25" --timed --time-ms 250 --label "250ms noisy endgame table" --report-every 2000 --workers 18`.
  Smoke check: 4 rounds, both sides 250 ms per decision.
- Stopping `endgame_cards=20` advanced `queue_250ms_v9.sh` to its last
  item, `250ms planning timed` (started 11:24, left running).

## Better, confirmed at 250 ms: fitted bid spread `bid_sigma=0.76` (17 November 2026)

Single change against the game bot of the time,
`hybrid:field=previous,inference=1,shortlist=13,touching=1` (duel,
`ohhell/duel.py`), candidate adds `bid_sigma=0.76`: opponents' bids are
read with the spread actually measured between lobby bids and the v1 trick
estimate (RMS 0.761 over 22,153 bids, 13 November `bid_sigma` entry)
instead of v2's 1.4 for every kind. Motivated by the seat-sampler
diagnostic: at 0.76 v2's sampler gave the best trump-count prediction.
Seed 321835169, source prefix 6774f07418aaeaf2, 250 ms,
old-core reference budgets (candidate {'card': 9876, 'bid': 17558, 'endgame': 10830},
baseline {'card': 9735, 'bid': 17410, 'endgame': 10404}), first duel on the faster
core (load factor about 3.4 recorded = the faster core, not load).

- Main: **better** at 36,000 rounds, **+2.76 +/- 2.62** points a game
  (O'Brien-Fleming margin); 19.0% of rounds diverged.
- Confirmation on unseen deals: 36,000 rounds, **+2.18**, one-sided 95%
  margin 1.55: **passed**. Verdict **better**.
- Promoted into the game's long-thinking bots: `Game.LONG_THINK` =
  `inference=1,shortlist=13,touching=1,bid_sigma=0.76` (750 ms opponents,
  advisor, bid game). 2 ms `VERSIONS` unchanged (not tested there).
- Files: `runs/duels/250ms_bid_sigma_0.{meta,jsonl,json}` (a naming bug cut
  the label at ".76"; fixed in `duel.py` afterwards).
- The queued bid-blend duel had started against the old game bot; stopped
  during calibration (no rounds kept) and the queue restarted.

Re-measured duel reference after the faster core: the new game bot at 250
ms on the idle machine (400 rounds, 18 workers) does 29,706 card / 54,583
bid / 24,376 endgame deals per turn, i.e. 118.8 / 218.3 / 97.5 per ms (old
core, old bot: 39.0 / 71.8 / 41.2). `REFERENCE_SPEC` is now the game bot
and `REFERENCE_RATES` these; the duel's default baseline is the game bot.
Remaining queue (`runs/duels/queue_250ms_v6.sh`), each one change against
the current game bot at these rates: bid_blend=0.25, oversample=8,
field=mix, card value mix, endgame_cards=20, planning (timed).

Queue change, 18 November (user: "it's so slow"): measured 195 rounds a
minute (3.25 a second) on the plugged-in laptop (the log's rate averages in
sleep time; the duel also paused whenever the lid was closed and resumed
from its saved rounds; one resume was needed after the agent killed a
paused duel by mistake at round 14,025). At 250 ms budgets (about 30,000
card / 55,000 bid deals a turn) sampling noise is negligible, so remaining
candidates are screened at one fifth of the budget (`--time-ms 50`, about
5,900 card / 10,900 bid deals a turn; labels "50ms ...", logs
`runs/duels/50ms_*.log`, queue `runs/duels/queue_50ms_v1.sh`). A screen
verdict is NOT a 250 ms verdict: a candidate that passes is confirmed with
a full 250 ms duel (main gate plus unseen deals) before it is adopted. The
bid-blend duel keeps running at 250 ms (it was at 21,071 rounds, +1.96
+/- 3.78 at 18,000).

`bid_blend=0.25` against the game bot with `bid_sigma=0.76` (250 ms, seed
841642214, source prefix 492527e590e0cbea, budgets 29,799 / 53,475 /
24,930 vs 29,700 / 54,575 / 24,375): positive throughout; stopped by the
agent at 48,211 rounds on the user's instruction: **+1.46** points a game,
planned O'Brien-Fleming margin at that point +/- 1.46, plain 95% +/- 1.23;
9.6% of rounds diverged. **Adopted by user decision, not by the gate**
(it sat exactly on the main-gate boundary; no confirmation run).
`Game.LONG_THINK` = `inference=1,shortlist=13,touching=1,bid_sigma=0.76,
bid_blend=0.25`; `runs/duels/game_bot_spec.txt` and `duel.REFERENCE_SPEC`
match (rates unchanged: the blend does not change cost).
Reading: with about 55,000 bid deals a turn and the fitted sigma, the
simulation's trick odds deserve more weight than v2's even split; at 10 ms
with sigma 1.4 (15 November screen) the best share leaned the other way.
Next (user): `bid_blend=0` (simulation only) as the first 50 ms screen,
then oversample=8, field=mix, card value mix, endgame_cards=20, planning
(`runs/duels/queue_50ms_v2.sh`).

50 ms screens dropped, 18 November (user): `bid_blend=0` at 50 ms (budgets
5,952 / 10,810 / 4,716 vs 5,940 / 10,915 / 4,875) stopped for futility at
6,000 rounds, -3.85 +/- 13.63: **inconclusive at 50 ms, untested at 250
ms**. The blend's best share depends on how many bid deals the simulation
gets (10 ms leaned to the network, 250 ms to the simulation), so a 50 ms
screen is biased against simulation-only bids; the same concern applies to
oversample and endgame_cards. `oversample=8` at 50 ms was stopped by the
agent at about 12,000 rounds (-0.07 +/- 6.16, no verdict). All remaining
candidates, `bid_blend=0` first, go back to 250 ms duels:
`runs/duels/queue_250ms_v8.sh` (started automatically 18 November 14:20 on
the user's instruction, after the self-play checks below).

Self-play checks, 18 November (analysis, not a candidate test). All seats the
game bot, full games (start hand down to blind and back), paired deals,
`runs/selfplay_check/`. 250 ms (70 games per setting): mean per player
190.5 / 174.2 / 100.1 at 3 / 5 / 8 players without the dealer restriction,
134.3 / 155.0 / 92.9 with it (paired effect -56.2, -19.1, -7.2). Same bot
at 2 ms: +14.7, +4.3, +3.1 points less (free). Under calls outnumber over
calls at every size (3 players: 19.0% vs 13.2%) at both 2 and 250 ms, so the
lean comes from the bidding model, not search depth. Bid thinking-time
sweep with card play fixed at 10 ms-equivalent budgets (free rule): bid
budgets of 2 / 250 / 750 ms-equivalents gave the same exact-bid rate within
about 1 point and no score difference beyond noise (2 vs 250 on 70 deals:
+2.4 +/- 9.9, -2.9 +/- 4.9, +1.3 +/- 1.6); 750 ms stopped by the user after
20-25 deals per size. More bid deals beyond 2 ms-equivalents (about 440)
do not improve bidding with the current model.

`bid_blend=0` (simulation-only bids) at 250 ms against the game bot
(`bid_blend=0.25`), seed 668196059, budgets 29,576 / 54,101 / 24,323 vs
29,700 / 54,575 / 24,375, load 76-84% of idle during calibration:
**no gain** (futility stop) after 6,000 rounds, -2.36 +/- 10.27 points a
game (950 s). Not promoted; the game bot keeps `bid_blend=0.25`. Wide
margin, so a small effect either way is not ruled out. Consistent with the
bid-time sweep: more weight on the simulation does not improve bids.
Queue continues with `oversample=8`.

`oversample=8` at 250 ms against the game bot, seed 391839718, equal-time
budgets 18,424 / 42,953 / 14,875 vs 29,700 / 54,575 / 24,375 (oversampling
costs about 38% of card deals), load 84-92% of idle during calibration:
**no gain** (futility stop) after 6,000 rounds, -1.52 +/- 8.61 points a game
(997 s). Not promoted. Queue continues with `field=mix`.

`field=mix` at 250 ms against the game bot (`field=previous`), budgets
25,915 / 44,705 / 33,400 vs 29,700 / 54,575 / 24,375: main gate **passed**,
+2.99 +/- 2.97 after 36,000 rounds (positive throughout). Confirmation on
unseen deals (36,000 rounds) was at -1.21 +/- 11.15 (10,000), -1.55 +/-
9.16 (12,000), +1.13 +/- 6.17 (18,000) when the **user adopted it, 18
November, before the confirmation verdict** (adopted by user decision, not
by the gate). `Game.LONG_THINK` now starts `field=mix` (750 ms opponents,
advisor, bid game; 2 ms `VERSIONS` unchanged, not tested there).
`duel.REFERENCE_SPEC` and `runs/duels/game_bot_spec.txt` are
`hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25`
(old spec kept in `runs/duels/game_bot_spec_field_previous.txt`);
`REFERENCE_RATES` = 103.66 / 178.82 / 133.6 deals per ms, the old rates
scaled by that duel's calibration ratio (not re-measured idle). The
confirmation keeps running against the old baseline; record its verdict
here. Remaining queue items read the new baseline when they start; the
planning item still hard-codes `field=previous`, so it now differs from the
baseline in field as well. Fixed the same day (user): the old queue shell
was stopped (confirmation left running) and the rest moved to
`runs/duels/queue_250ms_v9.sh`, which waits for the confirmation, then runs
card value mix, endgame_cards=20 and planning as
`planning:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25`
(`--timed`), so planning differs from the game bot only in its card search.
The `field=mix` confirmation was then stopped by the user at 20,000 of
36,000 rounds, +1.00 +/- 5.56: **no confirmation verdict** (missing
evidence). `field=mix` stays the default by user decision only.

Card value mix (`value_play=runs/value_play/model,value_play_mode=mix`) at
250 ms against the new game bot (`field=mix`), seed 150384647, equal-time
budgets 7,187 / 43,301 / 33,145 vs 25,915 / 44,705 / 33,400 (the value
network costs about 72% of card deals), load 62-66% of idle during
calibration: **no gain** (futility stop) after 12,000 rounds, -1.54 +/- 7.91
points a game (4,012 s). Not promoted. Queue continues with
`endgame_cards=20`.

## Faster reading in the C core: exact table-driven trick estimate and void-free dealing (17 November 2026)

User asked to speed up bid checks with a trick-estimate table. A C
benchmark built on `core.c` (4 players, 8 cards, -O3) first showed where
reading's time goes: one trick estimate 805-882 ns, one random deal
1,935-2,045 ns, a deal scored against 3 bids about 4.8 us. So dealing was
as slow as the bid checks.

Two exact changes (`runs/reading/fast_core_patch.py`, applied to `core.c`,
DLL `core_f7740c22cb772b72.dll`):
- `card_win_probs` (estimate 1, used by reading and imagined bids) reads
  per (players, hand size) tables built once with the same formulas: miss
  powers, reach chance, and the side-suit "safe" factor by trumps held,
  13- or 12-card suit, suit length and position. Checksum over 951,544
  random cards, all table and hand sizes: identical to 12 decimals.
- `sample_hands`: when no seat has shown a void, a partial Fisher-Yates
  shuffle cutting each hand off the front (uniform over consistent deals);
  with voids the old method is kept. Same distribution, different random
  stream, so individual deals (and timed decisions) differ from before.

| Benchmark (C) | Before | After |
| --- | --- | --- |
| trick estimate | 882 ns | 107 ns (8.2x) |
| deal, no voids | 1,935 ns | 243 ns (8.0x) |
| deal + log weight, 3 bids | 4,831 ns | 948 ns (5.1x) |

Deals per 2 ms turn (12 games, seat 0, measured while a duel used 16
cores, so absolute numbers are depressed): reading on 141.5 card / 173.0
bid, reading off 318.0 / 198.9. Reading now costs about 55% of a card turn
(was 83%) and 13% of a bid turn (was 58%). The rest of a card turn's
reading cost is void-constrained dealing and the play-likelihood replay.
Tests: 60 passed (native, estimate, value, duel).

Second pass (same day, `runs/reading/fast_core_patch2.py`, DLL
`core_018203996719ab10.dll`), also exact:
- Deals with shown voids: cards of unconstrained suits (placed last, in
  random order) are cut straight into the free room instead of placed one
  by one (same distribution); suit-pressure ordering in one bucketing pass
  (same order as before).
- Play likelihood: the public replay of past plays (state before each play,
  each seat's earlier plays, log(chance) per kind / options / match) is
  built once per block of candidates in `make_worlds_counted` and reused;
  outside a block it is keyed on the observation. Not used with
  `policy_inference`.
Checks (C, 60 mid-round positions, 3-6 players): checksum of log weights
over fixed candidate deals identical to the original to all digits; next
seat's mean trump count over kept deals 1.2212 (original) vs 1.2222.
End-to-end `make_worlds` (64 kept, oversample 4): 3,070 -> 1,863 ns per
candidate (1.65x); deal with voids 1,506 -> 730 ns. Python-level deals per
2 ms turn could not be measured cleanly (a duel was using 16 cores; runs
varied by about 25%). Tests: 85 passed.

Not a strength result. Effects: timed bots (the game's 750 ms opponents,
advisor) complete more deals per turn; the 2 ms v3 decision (reading off)
was made on the old core and may deserve a re-check; duel budgets still
use REFERENCE_RATES measured on the old core (re-measure after the current
queue), so queued duels keep the same deal counts and simply run faster.

## Negative diagnostic: seat-by-seat deal sampler (17 November 2026)

User's task: offline diagnostic of dealing hidden hands seat by seat
(accept each bidder's hand with probability L(bid|hand)/max L, then weight
the finished deal), against v2's whole-deal resampling. No bot setting, no
tournament; default unchanged. Code `ohhell/learn/seat_sampler.py`, tests
`tests/test_seat_sampler.py` (4 pass, incl. an exact enumeration of a
3-player one-card round with two bids).

Method. Same target for both: deals consistent with known cards and shown
voids x prod over opponents that bid of exp(-(bid - E)^2 / 2 sigma^2),
E = `expected_tricks` of the seat's opening hand (estimate 1), sigma 1.4
(`previous` field); play reading off in both. Seat sampler: bidders dealt
most extreme first (|bid - hand_size/n|), each hand uniform over the cards
its voids allow, accepted with prob a = L (max L = 1); others and the stock
filled last. Weight = prod_s C(A_s, k_s) (makes void-respecting sequential
dealing uniform) x prod_bidders Z_s(P_s), Z_s the acceptance rate given the
cards left. Estimators: `draws` = mean a(h) over 8 extra independent draws
(unbiased for Z, product unbiased for the weight); `tries` = 1/T (biased,
E[1/T] != p); `none`. Cap 200 tries per seat, restart on cap. Budget: equal
numbers of trick-estimate evaluations (v2: 64 x 4 candidates x bidders).

Diagnostic (`run --jobs 16 --games 3 --rate 0.12 --workers 4`, seed
20261117): 1,810 positions (388 bid, 1,422 card) from v2 games
(`hybrid:field=previous,inference=1,worlds=16,bid_worlds=32`), tournament
table sizes, down/blind/up, bowers; blind rounds and positions with no
opponent bid skipped. Calibration = mean log P(true owner) per hidden card
and log P(true trump count) per opponent, one pseudo-deal of smoothing.

| | v2 sampler | seat sampler (draws) |
| --- | --- | --- |
| ESS (v2: of 256 candidates) | 200.5 | 25.0 |
| distinct deals | 54.4 of 64 | 27.1 (27 deals) |
| card-owner log-lik (uniform -1.1663) | -1.1639 | -1.2011 |
| trump-count log-lik (uniform -1.1402) | -1.0243 | -1.0586 |
| restarts / dead ends | - | 0 / 202 (voids) |
| Python ms per position | 81 | 41 |

v2's ESS stays 181-244 in every group (1 to 4+ bidders, hand sizes 1-3 /
4-6 / 7-10, extreme bids): with sigma 1.4 the weights are flat, so the
hypothesised collapse does not happen. Seats needed 1.71 tries on average
(median 1.35, p90 2.53); the 8 extra draws per seat dominate the cost
(33.5 evaluations per deal). Python: 0.32 ms per v2 candidate, 1.55 ms per
seat deal; in C about 1 us per evaluation (from 250 ms throughput), so
about 34 us per seat deal vs 2-4 us per v2 candidate.

Exactness (`exact --cases 12 --reference 40000 --repeats 40 --budget
2000`, seed 11; 3-4 players, 2-3 cards, skilled bidders, observer = dealer;
mean |marginal - reference| over hidden card x opponent): reference noise
0.0012, v2 0.0020, seat draws 0.0030, seat tries 0.0015, seat none 0.0014.
All near the floor: these bids are mild, so the order correction barely
matters here and the unbiased draws mainly add noise. `runs/seat_sampler/`.

Reading also predicts trump counts better than uniform but barely improves
card-owner prediction. **Recommendation: do not build `sampler=seat`.** It
fixes a collapse v2 does not suffer, at about 8x the cost per deal, and
calibrates no better. A better opponent model (how bids and plays depend
on hands) is the more promising reading change.

### Follow-up: stricter bid reading (same positions, `--sigma`)

User asked whether seat by seat wins if "near the bids" were stricter. The
log already had the answer to what is accurate: over 22,153 lobby bids the
RMS of bid - estimate (v1 estimate, what v2 uses) is 0.761 (13 November
`bid_sigma` entry), so v2's sigma 1.4 is about twice too loose; sigma was
only ever changed inside a five-change bundle. Same positions and seed,
three spreads; "cheap seat" = estimator `tries` (no extra draws).

| sigma | v2 ESS of 256 (p10) | v2 card / trump log-lik | seat cheap deals (ESS) | seat cheap card / trump | restarts |
| --- | --- | --- | --- | --- | --- |
| 1.4 | 200.5 (123) | -1.1639 / -1.0243 | 179 (138) | -1.1389 / -1.0098 | 0 |
| 0.76 | 103.3 (16) | -1.2027 / **-0.9782** | 110 (67) | -1.1799 / -1.0051 | 181 |
| 0.5 | 52.8 (3) | -1.3716 / -1.1367 | 72 (37) | -1.2296 / -1.1090 | 813 |

Uniform: card about -1.167, trump -1.115 to -1.140. Positions where a seat
sampler produced no deal are left out (1,810 / 1,795 / 1,731). The unbiased
`draws` version trails the cheap one at every sigma (fewer deals).
- Stricter reading collapses v2's resampling (10th-percentile ESS 123 ->
  16 -> 3) and seat by seat overtakes it as sigma tightens, as the user
  expected; at 0.5 it is better on both scores, but 0.5 is overconfident
  for everyone.
- The fitted 0.76 gives the best trump-count prediction of all (v2's
  sampler), while v2's card-owner score drops below uniform there (a few
  near-duplicate deals make the marginals overconfident); the cheap seat
  sampler is better on card owners at 0.76 and at 1.4.
- Caveat: these scores reward more deals; in the bot each deal also costs
  play-outs, so only a game test settles it.
Next: duel `bid_sigma=0.76` alone at 250 ms (queued first after grouping,
`runs/duels/queue_250ms_v5.sh`). If it helps, build the cheap seat sampler
as an opt-in C/Python setting and duel it at the fitted sigma.

## New gate for long thinking times: paired round duels (16 November 2026)

User: optimise for the game's opponents (750 ms; 250 ms-1 s), testing as
fast as possible with at least 95% confidence. A 1,300-set tournament at
50 ms would take about 8 hours (`Reading at 50 ms`, stopped by the agent at
93 sets: reading +5.48, plain 95% +/- 8.55; v2 reading vs v3, every bot 50
ms, seed in its .meta; not a verdict). New tool `ohhell/duel.py` (README
"Duels at long thinking times"; tests `tests/test_duel.py`):

- One sample = one round drawn like a game's rounds (tournament table
  weights, every label incl. blind, random dealer and test seat), played
  twice on the same cards: baseline then candidate in the test seat (order
  alternates). Other seats: same opponents, same seeds, fixed budgets
  (default field `hybrid:field=previous,inference=1,worlds=16,bid_worlds=32`
  x3, `skilled` x1, `casual` x1). Difference x rounds per game = points per
  game; bots never see the game score and rounds are dealt afresh.
- Calibrated deal budgets: each version is timed at `--time-ms` for 300
  sampled rounds under the duel's own load; it then plays with fixed
  budgets equal to its mean deals per bid / card / endgame turn (new
  `card_budget`, `bid_budget`, `endgame_budget` on simulation bots; C
  `Params.budget`). Same mean compute, exactly repeatable. Identical specs
  share one calibration.
- Stopping: the tournament's O'Brien-Fleming plan (10 looks, 5% overall,
  `FINAL_Z` 2.087), better / worse / no gain (< 1 point a game); "better"
  repeats on unseen deals and must clear zero one-sided 95%. `--stopping
  cs` gives an always-valid confidence sequence (wider).

Checks:
- A/A, v2 with reading vs itself, 2 ms calibrated: exactly 0 difference,
  0 divergence (with the clock instead, 32% of rounds diverged and the
  per-round spread was about 10 points: pure timing noise).
- v3 vs v2 at 2 ms (calibrated budgets 274/255/342 vs 47/104/66): +1.59
  points per game over 40,000 rounds (plain 95% +/- 2.53; confidence
  sequence +/- 4.19), 38% of rounds diverged; 400 rounds/s. Smaller than
  the tournament's +5.2: this field has no random-card novices.
- Spread: a diverged round moves about 20 points (sd), which sets the
  sample size: about 50,000 rounds per +/-2.5 points a game. Optimal
  stratified allocation would save only ~22% (not implemented).
- 250 ms calibration (v2 with reading, 18 workers): about 9,800 card,
  18,400 bid and 10,300 endgame deals per turn; 6.2 rounds/s.

- Futility stop (added during the first 250 ms duel, applies to later
  duels): at a planned look, stop as "no gain" if even a plain 95% upper
  bound is below `--smallest-gain`. It cannot create a false "better".
- From the `shortlist=13` duel on, `--smallest-gain` defaults to 2 points a
  game (was 1): the user wants significant gains, and with a 1-point bar a
  null candidate ran the full 60,000 rounds (about 2.5-3 h at 250 ms).

Queued by the agent, all single changes against the game's bot
`hybrid:field=previous,inference=1` at 250 ms with the default field:
`play_evidence=1` (running), `endgame_cards=16`, `shortlist=13`,
`bid_blend=0.25`, `oversample=8`, `field=mix`,
`value_play=runs/value_play/model,value_play_mode=mix`, `endgame_cards=20`.
Results are appended below as they finish.

| Candidate (added to the game's bot, 250 ms) | Seed | Rounds | Edge pts/game | Verdict |
| --- | --- | --- | --- | --- |
| `play_evidence=1` | 437700812 | 60,000 (2.5 h) | -0.63 +/- 1.62 | no gain |
| `endgame_cards=16` | 366136492 | 60,000 | -0.09 +/- 1.30 | inconclusive |
| `shortlist=13` | 213407189 | 60,000 | +0.69 +/- 1.14 | no gain |

`play_evidence=1`: source prefix 3cc3104394994edf; calibrated budgets
9,750 / 17,958 / 10,297 (candidate) and 9,807 / 18,363 / 10,308 (baseline)
card / bid / endgame deals; 326 vs 328 ms per choice; 27% of rounds
diverged; started before the futility stop existed. By table size the edge
per round is within +/-0.1 everywhere.

`endgame_cards=16`: source prefix 386b1b09900413ed; budgets 8,515 / 16,354
/ 7,230 (candidate) vs 9,699 / 17,517 / 10,292 (baseline); 19% of rounds
diverged. The run straddled an overnight sleep (61,423 s wall), so the
recorded ms per choice are inflated: 2,471 vs 2,138. The candidate's ratio
(+16%) suggests fixed exact-endgame budgets cost more on hard 16-card
positions than the clock allowed; even with that extra compute, no gain.
Limitation of calibrated budgets: equal mean deals is not exactly equal mean
time when per-deal cost varies by position.

Calibration fixed 17 November (agent): the `shortlist=13` duel calibrated
its candidate at 3,811 card / 7,063 bid deals per turn and the baseline at
3,685 / 6,906, against about 9,700 / 17,500 overnight: the machine was in
use, so each spec's budget depended on the load during its own (separate)
calibration. Stopped that duel and the queue (no results kept). Now the
game bot `hybrid:field=previous,inference=1` always gets REFERENCE_RATES x
time (idle 250 ms rates: 39.0 card, 71.8 bid, 41.2 endgame deals per ms);
any other spec gets that times its deals per turn relative to the game bot,
both timed together on interleaved rounds (load cancels). The measured load
is recorded in the .meta. The `play_evidence` and `endgame_cards=16` duels
above used the earlier separate calibration on an idle machine.

`shortlist=13` (first duel with reference-rate budgets): budgets 9,533 /
17,787 / 10,512 vs 9,750 / 17,950 / 10,300; reference ran at about 40% of
its idle rate during calibration (machine in use); 18,930 s wall. Positive
throughout (+2.6 early, settling near +0.7) but no gain against the 2-point
bar; any real effect is about a point or less.

**Adopted by user decision (17 November), not by the gate:** the game's
long-thinking bots (`Game.LONG_THINK`, used by the 750 ms opponents, the
advisor and the bid game) now add `shortlist=13`. `VERSIONS` / the 2 ms
default are unchanged (not tested at 2 ms with reading). From here on the
duel queue reads the current game bot from `runs/duels/game_bot_spec.txt`
(`hybrid:field=previous,inference=1,shortlist=13`) and tests each change
against it. The first grouping duel (touching + all cards vs the old game
bot) was stopped after about 15 minutes and restarted as `touching=1`
against the new game bot, a single change; no results kept.

`touching=1` (equal-card grouping) against the all-cards game bot, seed
837799134, source prefix 7bf45324fac6595f: stopped by the agent at 22,187
rounds on the user's instruction ("if it's still winning, accept"):
+0.57 points a game (plain 95% +/- 1.59; planned O'Brien-Fleming margin at
that point +/- 2.79); 7.5% of rounds diverged. **Adopted by user decision,
not by the gate**: `Game.LONG_THINK` is now
`inference=1,shortlist=13,touching=1`, and `runs/duels/game_bot_spec.txt`
matches, so later duels test against it. Summary written to
`runs/duels/250ms_grouping.json` (verdict "stopped: adopted by user
decision"). From the `bid_sigma=0.76` duel on, duels also run on the faster
core (same results per deal, about 1.65-5x faster reading).

## Better, confirmed: no opponent reading at 2 ms; promoted as v3 (16 November 2026)

Measured first (`runs/throughput.py`, 12 games of 4-5 players, opening
9, seat 0 vs v2 seats, 2 ms): reading opponents (deal weighting by bids and
by replaying every past card through the rule-based player, oversample 4)
took about 83% of each card turn.

| Spec | Card deals / turn | Bid deals / turn |
| --- | --- | --- |
| v2 `hybrid:field=previous,time_ms=2` | 185 | 194 |
| v2 + `inference=0` | 1,084 | 457 |
| v2 + `oversample=1` | 641 | 417 |
| v2 at 10 ms | 1,121 | 1,131 |

Concrete new hypothesis (the 10 November no-inference runs were under
Ace-high rules and without this cost measurement): reading costs more
deals than it is worth at 2 ms. Three single changes against unchanged v2,
run by the agent, same rules/lobby/settings as the other 16 November runs,
source prefix 68ee3a28da433eef, v2 weights a84e9315... / 8cb5a064...:

| Candidate | Seed | Main: sets, edge | Confirmation (unseen deals) | Verdict |
| --- | --- | --- | --- | --- |
| `oversample=1` | 1061034680 | 910, +3.44 +/- 3.36 | 910, +2.25 +/- 2.47 | better |
| `oversample=2` | 641157890 | 1,040, +3.76 +/- 2.96 | 1,040, +2.91 +/- 2.46 | better |
| `inference=0` | 1068196583 | 650, +5.30 +/- 4.48 | 650, +5.18 +/- 3.03 | **better** |

Win-share edges +0.033, +0.022, +0.026 (no disagreement). `inference=0`
by table size (main): 3p +1.03, 4p +7.04, 5p +6.38, 6p +6.23, 7p +3.14,
8p +2.40. Three related candidates were tested; each passed both stages
on its own deals.

**Promoted `inference=0` as v3 "No reading"**,
`hybrid:field=previous,inference=0,time_ms=2`, in `VERSIONS`. Weights
unchanged. Limitation: measured at 2 ms only. The game (750 ms), advisor
(1 s), bid helper and bid game use `DEFAULT_SPEC` at longer times, where
reading may be worth its cost; see the 10 ms check below.

### Reading at longer times (10 ms check)

`hybrid:field=previous,inference=0` against `--baseline hybrid:field=previous`,
every bot at 10 ms, seed 44884907, source prefix e83467898b51b8e5: stopped
at 520 sets, verdict **no gain**, -4.63 +/- 5.40 (3p -7.71, 4p -5.67, 5p
+0.52, 6p -5.83, 7p -2.29, 8p -3.75). Reading is ahead again by 10 ms, so
the crossover lies between 2 and 10 ms. `game.py` now adds
`Game.LONG_THINK = 'inference=1'` to the game bots (750 ms), the advisor
(1 s) and `bid_game.py`'s bidders (60 ms); their reading is unchanged from
v2. That 750 ms reading beats 750 ms no-reading is inferred from the trend,
not measured.

### Bid-only reading (not gated)

Added `reading=all|bids` (`Params.bid_only` in C, `Belief.play_reading` in
Python): weigh deals by the bids only. Throughput at 2 ms: 162 card / 160
bid deals per turn (v2 185 / 194; v3 796 / 368 on the same check). The bid
estimate per candidate, not the play replay, is the main cost, so this is
not a cheap middle ground. No tournament run; default `all`, v2 unchanged.

### Value networks on top of v3 (2 ms, baseline v3)

| Candidate (added to v3) | Seed | Sets | Edge | Verdict |
| --- | --- | --- | --- | --- |
| `value_play=runs/value_play/model,value_play_mode=mix` | 227381608 | 1,040 | -2.04 +/- 2.95 | no gain |
| `value_play=...,value_play_mode=value` | 355903934 | 910 | -2.45 +/- 3.33 | no gain |
| `value_bids=runs/value_bid/model` (share 1/3) | 1046041687 | 1,300 | -0.64 +/- 2.29 | inconclusive |

With plenty of deals the learned values still do not beat the play-outs.
**Do not repeat these networks unchanged.**

### Search settings on top of v3 (2 ms, baseline v3, source 1d8a7ba46ad552ba)

| Candidate (added to v3) | Seed | Sets | Edge | Verdict |
| --- | --- | --- | --- | --- |
| `shortlist=13` (every legal card) | 191979746 | 1,300 | -0.09 +/- 2.32 | inconclusive |
| `endgame_cards=16` | 382303017 | 1,300 | -0.95 +/- 2.26 | inconclusive (run took 43 min, exact search is slow) |
| `prior_worlds=0` (no play-network prior) | 355618900 | 1,300 | -0.39 +/- 2.28 | inconclusive |

| `bid_blend=0.7` | 17226624 | 1,170 | -1.60 +/- 2.51 | no gain |

Settings tuned when deals were scarce do not matter measurably now.

### Target time changes the plan (user, 16 November 2026)

The user wants the bot optimised for 250 ms to 1 s a turn (the game's
opponents and the advisor), not 2 ms. v3's gain is a 2 ms result and is
expected NOT to carry over (reading already ahead at 10 ms). Every 16
November "no gain" above is a 2 ms result and does not rule the idea out at
250 ms+, where sampling noise is negligible and systematic errors (rule-bot
play-outs, opponent model, fixed blends, endgame size) dominate. Plan:
screen at 25-50 ms, confirm at 250 ms, and only count a version as better
for the game if it wins at the game's time.

## Built: learned card-play value, native search hook (16 November 2026)

New module `ohhell/learn/value_play.py` (collect / train / positions /
screen). A network sees a trick-start position (every hand, played cards,
turned-up card, bids, tricks won, leader, cards left, call, seat type) and
predicts that seat's final trick count. Targets: real outcomes of lobby
rounds. In search: play our card, finish the trick with the simulated
population, then the network's expected points instead of a play-out.

- Data: `collect --rounds 300000 --first 9000000 --time-ms 2`, 699 s on 18
  workers, `runs/value_play/train.jsonl`. Network 677-256-128-11, 10 epochs,
  25% of (trick start, seat) rows kept: 1,662,241 rows. Test log loss on
  15,000 held-out rounds 0.5500. `runs/value_play/model/value.npz`, SHA-256
  150a18960c59e6b4ca3985ae37e3033f307492e401cc8723672f2273061a53e8.
- Positions: `positions --rounds 40000 --first 9500000 --time-ms 2`: one v2
  card decision per round outside the exact endgame, v2's real choice, one
  real lobby finish per legal card. 16,219 positions.
- Screen (64 sampled deals, v2's Python belief), points per decision vs
  v2's real choice: net -0.12 +/- 0.19, sim -0.07 +/- 0.22, value +0.06 +/-
  0.22, v2-like Python blend +0.00 +/- 0.18 (71% same card), value + net
  prior +0.04 +/- 0.19, value + sim + net +0.09 +/- 0.18. **Nothing
  separates**: one playout per option leaves +/-0.2 per decision, coarser
  than a tournament. Decided to gate directly instead.
- Native: `oh_evaluate_cards_value`, `oh_vplay_points`, shared `ValueNet`
  forward pass (row-major, skips inactive units; parity tests pass to 1e-3
  points). `HybridBot` options `value_play=FOLDER`,
  `value_play_mode=value|mix` (mix averages the value and the play-out);
  shortlist, play prior and exact endgame unchanged. The same build applies
  the earlier `value_bids` code. Tournament fingerprints both value files.
  At 2 ms under load the value search finishes about half as many deals as
  v2 (26 vs 54 on a smoke check), each much less noisy.

### Learned bid reading (stopped)

`ohhell/learn/bid_policy.py`: P(bid | hand, table, position, earlier bids)
fitted to the same 300,000 lobby rounds; 99-128-64-11, test log loss 0.6958,
top-1 0.733. `runs/bid_policy/model/policy.npz`. Its screen (1,024 uniform
deals reweighted by the learned likelihood, value network on top) was
stopped by the agent after 200 of 613 chunks as too slow (16 cores, ~70
min); no result. Not run in any game.

### Tournaments run by the agent (16 November 2026)

All: bowers, cumulative down/blind/up, opening min(10, 51/n), 3-8 players,
standard scoring/dealer restriction, every bot 2 ms, 18 workers, maximum
1,300 sets, smallest gain 1.0, baseline unchanged v2
`hybrid:field=previous,time_ms=2` (bid a84e9315... / play 8cb5a064...).
Source SHA-256 prefix 68ee3a28da433eef. Each took about 4 minutes.

| Candidate (added to v2) | Seed | Sets | Edge pts/game | Verdict |
| --- | --- | --- | --- | --- |
| `value_play=runs/value_play/model,value_play_mode=mix` | 187225918 | 1,300 | -1.45 +/- 2.27 | no gain |
| `value_play=runs/value_play/model,value_play_mode=value` | 59286715 | 1,300 | -1.12 +/- 2.28 | inconclusive |
| `typing=1` (prepared opponent-typing stage) | 628565757 | 1,170 | -1.88 +/- 2.64 | no gain |
| A/A control: v2 itself as candidate | 959039852 | 1,300 | +1.33 +/- 2.20 | inconclusive |

The A/A run shows the gate is unbiased and its noise is about +/-2.2: none
of the three candidates moved strength measurably. **Keep v2; do not repeat
these unchanged.** The learned card value is about as good as the rule-bot
play-out on the same deals, at twice the cost per deal.

## Prepared: bid log-loss floor for v2 (16 November 2026)

User asked what the bid network's ~0.99 log loss (bid-blend screen,
`runs/bid_blend/t10.report.json`) should be compared against, and asked
for both limits to be measured for v2. **A measurement, not a strength
test. No full run yet. Keep v2.**

New tool `ohhell/learn/bid_floor.py` (tests `tests/test_bid_floor.py`, 2
pass). Same tables, lobby and hand-size draw as `bid_blend_fit collect`,
10 ms. At each position v2 bids for real; for that chosen bid it stores one
actual playout (the scored outcome), the network's and v2's blended odds,
and K playouts each of: `oracle` (same deal, fresh bot seeds: lower limit),
`reader` (earlier bidders' hands drawn seat by seat until their real bots
reproduce their real bids: approximately the floor for a seat that knows
every style; a later seat's bid does not reweight an earlier seat's hand),
`exact` (whole-deal rejection, a check on `reader` where it reaches K), and
`blind` (hidden hands uniform, bids ignored). Sample forecasts use
add-0.25 smoothing; finite K biases them upwards, so `report` also shows
K/2 and a 1/K extrapolation. Scored on v2's chosen bid only, unlike the
screen's all-shortlist average, so compare with the network on the same rows.

Smoke check only (8 positions, K=8, round seeds from 7,300,000, run alongside
the user's `runs/value_bid/confirm_t10.jsonl` collection): ran end to end,
about 15 s per position on one contended worker. Planned run:
`python -m ohhell.learn.bid_floor collect --positions 1500 --deals 24 --workers 18 --out runs/bid_floor/t10.jsonl`,
then `python -m ohhell.learn.bid_floor report runs/bid_floor/t10.jsonl`.

## Screened: learned full-deal value for bidding (15-16 November 2026)

New module `ohhell/learn/value_bid.py` (collect / train / screen). A network
sees the whole deal once bidding is over (every hand, the turned-up card,
every bid, table and hand size, bidding position, whether the seat is a v2
version or a skilled/casual/novice player) and predicts that seat's trick
count. Targets are real outcomes of lobby rounds where nobody saw another
hand. In the bid screen it replaces the rule-based play-out on v2's own
sampled deals (`bot.belief.worlds`, 64 per position) and imagined later bids.
**Offline screen only; no tournament. Keep v2.**

- Training data: `collect --rounds 300000 --first 8000000 --time-ms 2`,
  tournament table sizes and lobby, 730 s on 18 workers,
  `runs/value_bid/train.jsonl`.
- Network 587-256-128-11, softmax over tricks, 12 epochs, rate 1e-3, seed 1.
  285,000 training rounds / 1,337,309 seat rows. Test on 15,000 held-out
  rounds (round % 20 == 0): log loss 0.6544 with the true deal visible.
  `runs/value_bid/model/value.npz`, SHA-256
  4c887ca149cf7e227c2bfefa03e5a80a07e9474195fe7a979e132a161bce88cd.
- Screen positions are rebuilt from `bid_blend_fit` files; earlier bids are
  replayed and a position is kept only if the stored network odds match
  (about 97.5% kept). Outcomes: one real playout per option at 10 ms.

| Bid odds | t10.jsonl, 38,186 pos.: log loss / pts per bid vs v2 | fresh 25,000 rounds from 7100000, 23,810 pos. |
| --- | --- | --- |
| simulation | 1.0665 / -0.18 +/- 0.08 | 1.0907 / -0.16 +/- 0.10 |
| network | 0.9878 / -0.46 +/- 0.09 | 0.9914 / -0.38 +/- 0.11 |
| v2 (0.5/0.5) | 0.9780 / 0 | 0.9831 / 0 |
| value alone | 1.0008 / -0.07 +/- 0.09 | 1.0025 / -0.11 +/- 0.11 |
| value on true deal (oracle) | 0.8970 / +1.38 +/- 0.13 | 0.8961 / +1.28 +/- 0.17 |
| 1/3 value + 1/3 net + 1/3 sim | 0.9684 / +0.10 +/- 0.05 | 0.9717 / +0.04 +/- 0.07 |

The three-way mix was picked after seeing the first file (of three mixes
tried); on fresh positions its log-loss gain repeats but the points gain
does not (+0.04 +/- 0.07 per bid, at most about +1 point per game). Not
worth a tournament: the gate cannot resolve +1.
Readings: on the same sampled deals the value network is a much better
judge than rule-based play-outs (1.00 vs 1.07-1.09), and the gap from value
to oracle (about 0.10 log loss, +1.3 points per bid) is what better reading
of the hidden hands could be worth at most.
Opt-in `HybridBot` settings `value_bids=FOLDER`, `value_share` (1/3),
`value_playouts` (1) were added to `ohhell/bots/learned.py`, but the native
`oh_analyse_bids_value` / `Simulator.set_value` they call was NOT applied to
`core.c` (another session was active; the DLL name follows the source hash).
So `value_bids` currently fails at construction; default off, v2 unaffected.
Untested in any game. Tests: `tests/test_value_bid.py` (4, encoding only).

## Negative at screening: less noisy self-play labels (15 November 2026)

User asked whether the networks could be trained without label noise, then
asked the agent to test it while away. Two new opt-in `selfplay.py` options,
both default off, so v2 and the existing recipe are unchanged. **No
tournament was run; network-only arena checks only. Keep v2.**

- `--paired`: every option at a decision (true-deal card and bid labels) is
  finished with one shared random seed (common random numbers).
- `--play-worlds K`: each card label is the seat's mean points over K deals
  resampled from its observation (`Belief(inference=True, field='mix')`),
  true kinds and model seats kept, finished by the same self-play policies
  (networks, not the rule bot). Not combined with `--paired` here.

### Noise check first

Script `runs/noise/noise_check.py`, output
`runs/noise/noise_check_20261115.txt`. Self-play table recipe (model share
0.6, mix kinds), v2 weights, seeds 424242+, 1,402 model-seat card decisions /
4,657 card labels, K=16 resampled deals. Variances in round points squared:

| Quantity | All | 2-3 cards | 4-6 | 7-10 |
| --- | --- | --- | --- | --- |
| Play-out randomness, same deal | 54.7 | 11.8 | 45.3 | 121.0 |
| Hidden cards + play-out, per deal | 201.1 | 88.6 | 204.3 | 323.9 |
| Net MSE vs one true-deal label | 253.4 | 127.7 | 265.0 | 375.7 |
| Card-difference noise unpaired / paired | 41.6 / 35.5 | 8.5 / 4.8 | 32.8 / 27.4 | 95.8 / 85.4 |
| Net centred MSE vs 16-deal mean (noise left) | 29.8 (6.4) | 15.9 (1.8) | 35.1 (5.9) | 36.1 (12.4) |
| Net best card = 16-deal best | 47.0% | 58.7% | 41.2% | 33.6% |

About 80% of the label variance the play net trains on is noise, mostly
from hidden cards; pairing seeds removes only ~15% of the card-difference
noise because paths diverge after the first card. The net's own error is
real but much smaller than the label noise. The 16-deal reference uses the
bot's belief model, so "net error" here is relative to that model.

### Paired seeds, full iteration

`python -m ohhell.learn.selfplay --start ohhell/models --out runs/selfplay/paired_iter5
--iterations 1 --rounds 150000 --epochs 4 --seed 5 --workers 9 --paired`; control
identical without `--paired` (`runs/selfplay/control_iter5`). Run side by side
(1,600 s generation each). Held-out bid / play loss: paired 0.9930 / 0.1399,
control 0.9893 / 0.1401.
- Built-in net-only checks vs v2 (4p, opening 10, 300 games): control
  +1.0 +/- 3.4, paired +1.4 +/- 3.3. One more plain self-play iteration on v2
  shows no gain either (self-play had already levelled off).
- **Paired vs control head to head**, net-only, 4p, opening 10, 1,500 games
  from seed 1150000: **-0.5 +/- 1.6**. No gain.
- SHA-256 paired bid/play 6996fead...b807 / 1a9eb5c2...32b9; control
  2ca86719...51f7 / 69e143e4...f59a.

### Resampled-deal card labels, K=8

`--start ohhell/models --iterations 1 --rounds 40000 --epochs 4 --seed 6
--freeze-bid --play-worlds 8` (`runs/selfplay/worlds8_play`, 11 workers,
1,306 s generation); control identical without `--play-worlds`
(`runs/selfplay/control_play40k`, 4 workers). Bid net is v2's in both.
Held-out play loss 0.0498 (averaged labels) vs 0.1363 (control): much
cleaner labels, as intended.
- Built-in checks vs v2: control -2.4 +/- 3.2, averaged -3.7 +/- 3.2.
- **Averaged vs control head to head**, net-only, 4p, opening 10, 1,500 games
  from seed 1160000: **-2.9 +/- 1.4**. Worse.
- SHA-256 averaged play 7b257e6b...df4ce8; control play 691384bb...ba87.

Reading (untested explanation): the true deal is an unbiased draw from the
real hidden-card distribution, so a single true-deal label is noisy but
unbiased, and the network averages it for free. Resampled deals come from
the bot's belief model, whose errors replace the noise with bias. Same
pattern as the search-label run. Lower label noise is not the bottleneck;
target quality is.

Missing evidence: net-only checks at one table size; no hybrid (2 ms) or
tournament comparison; one iteration each; K=8 and 40,000 rounds only.
**Do not repeat either unchanged.** Tests: two added to
`tests/test_selfplay.py` (8 pass).

## No gain: fitted bid blend and simulation-only bids at 10 ms (13 November 2026, logged 15 November)

Both user-run tournaments from the "bid blend retest" plan below. Recorded
from the saved files on 15 November; neither was in this log. Every bot at
10 ms, bowers, cumulative down/blind/up, opening min(10, 51/n), 3-8 players,
standard scoring/dealer restriction. v2 weights bid a84e9315...e29d / play
8cb5a064...d474 in both manifests. Source SHA-256
40efb3325362fab44198560849946f9e596ec256e9916058cf6b02d79ea7e21e.

Collection/fit (step 1-2 of that plan) did run: `runs/bid_blend/t10.jsonl`,
39,139 bid positions, report `runs/bid_blend/t10.report.json`. Overall log
loss of the trick odds against the real outcome: simulation only 1.072,
network only 0.988, best fixed share 0.65 (0.976). Mean 450 simulated deals
per bid, so the simulation's worse fit is bias, not sampling noise.
Table written to `ohhell/models/bid_blend.json`, SHA-256
b60e6a70fa892edec4b189b8cd012ff4bef5fffd33c2b89661b4fb04f99f2fec.

- **Fitted bid blend at 10 ms**, `hybrid:field=previous,bid_table=ohhell/models/bid_blend.json`,
  seed 551882984: verdict **no gain** at 910 sets / 4,270 games,
  **-2.72 +/- 3.23** points per game. Win-share edge -0.035 +/- 0.023, no
  disagreement. By size: 3p -6.45 (140), 4p -3.60 (420), 5p +1.19 (140),
  6p -5.05 (70), 7p +2.43 (70), 8p -0.68 (70). Evidence
  `runs/gates/Fitted_bid_blend_at_10_ms.{meta,jsonl,json}`.
- **Simulation-only bids at 10 ms**, `hybrid:field=previous,bid_blend=0`,
  seed 995630769: stopped by the user at **418 sets / 1,961 games** (sets
  0-417), no summary `.json`. Computed from the `.jsonl`: **-4.47** points
  per game, plain 95% +/- 3.87 (planned gate margin at 418 sets +/- 7.26).
  Win-share edge -0.038 +/- 0.036. By size: 3p -5.38 (65), 4p -4.13 (192),
  5p -6.97 (64), 6p -12.73 (33), 7p +1.70 (32), 8p +2.62 (32).
  Leaning worse; not a completed verdict.
- Reading: the network's bid odds beat the simulation's even with plenty of
  deals, and leaning further on the simulation does not help. Consistent with
  the rollout policy/opponent model being the limit. **Keep v2; do not rerun
  either unchanged.**

## Prepared: bid blend retest at 10 ms, fitted share table (13 November 2026)

User asked why simulation-led bidding does not do better on small hands and
small tables, where a bid "can be really accurate", and asked for it to be
tested again together with a fitted blend (not the linear `blend_pivot`
line), at 10 ms. **Nothing below is a strength result. No tournament and no
full data collection has run. Keep v2.**

What the log already had: `bid_blend=0` alone at 2 ms was stopped at 76 sets
on a contended machine (-17.1 +/- 9.4, not clean); the linear pivot-6 blend
was inconclusive (-0.54 +/- 2.41; 1-5 card rounds summed to -0.72 per game);
the five-change simulation-bid bundle at 10 ms showed no gain (-2.36 +/-
2.37). `bid_blend=0` alone at 10 ms was proposed there and never run. The
earlier `blend_check` produced no output (`runs/blend_check/check.log` is
empty), so there was no fitted schedule to test.

Hypotheses for the small-hand question, **untested**: sampling noise is not
the limit there (a 10 ms bid turn completed about 3,500 deals at one card in
the smoke run below), so more deals cannot help; what remains is bias in
what is simulated. Earlier bidders' hands are inferred from a rule-of-thumb
estimate with a 1.4-trick spread, later bidders are imagined with the same
rule and do not react to our bid, and every seat including ours is played
out by the rule-based player, while the real table is searching bots. The
network was fitted to real outcomes. The one-card `ranges` result (+0.2 to
+0.5 per game from reading bids exactly) points the same way.

New tool `ohhell/learn/bid_blend_fit.py`:
- `collect`: tournament table sizes and lobby (v2 standing in for the
  candidate), hand size uniform 1 to the table's maximum, every simulating
  bot at `--time-ms`. At the candidate seat's bid it stores the simulation's
  and the network's trick odds for every shortlisted bid, then plays the
  real deal out once after **each** of those bids (later seats really bid,
  everyone really plays, v2 plays our cards). Unlike `blend_check`, every
  option has an outcome, not only the bid v2 chose.
- `fit`: points per bid of simulation-only and network-only against v2's
  0.5, with log loss, by hand size, table size and bidding position; a
  share per (hand size, players) by log loss, each cell borrowing `--shrink`
  (default 200) pseudo-positions from its row and column; five-fold
  held-out points of that table against v2 and against simulation-only.
  Limits: one noisy playout per option, v2's card play, shortlisted bids
  only. A screen, not a gate.
- New opt-in `HybridBot` setting `bid_table=FILE`: the fitted share replaces
  `bid_blend` (and pivot scaling) for bids where the table has an entry;
  0 skips the network. Card play untouched. Default off; v2 unchanged. The
  tournament manifest fingerprints the table file.

Smoke run only (mechanics, **no conclusion**): seeds 7000000-7000159, 10 ms,
4 workers, 157 positions in 24 s. Every interval spans zero. Not saved.

Not run: the agent's 40,000-round collection (about 25 minutes on 18
workers) was declined as a long run, so the user runs it. No table has been
fitted and `ohhell/models/bid_blend.json` does not exist yet.

Commands for the user, in order (PowerShell):
1. `.venv\Scripts\python.exe -m ohhell.learn.bid_blend_fit collect --rounds 40000 --time-ms 10 --workers 18 --out runs/bid_blend/t10.jsonl`
   (seed base 7000000; no dashboard, prints progress)
2. `.venv\Scripts\python.exe -m ohhell.learn.bid_blend_fit fit runs/bid_blend/t10.jsonl --report runs/bid_blend/t10.report.json --write ohhell/models/bid_blend.json`
3. Retest of simulation-only bids, single change, every bot at 10 ms:
   `.venv\Scripts\python.exe -m ohhell.tournament "hybrid:field=previous,bid_blend=0" --label "Simulation-only bids at 10 ms" --time-ms 10`
4. Fitted table, single change, every bot at 10 ms, only if step 2's
   held-out row is positive:
   `.venv\Scripts\python.exe -m ohhell.tournament "hybrid:field=previous,bid_table=ohhell/models/bid_blend.json" --label "Fitted bid blend at 10 ms" --time-ms 10`
Dashboard http://localhost:8765/ (next free port if busy). Run them one at a
time. Both were loaded with `--preflight-only` (the second with a throwaway
table); no games played. The tournament prints edges per hand size and
table size. A pass at 10 ms would not by itself justify changing the 2 ms
default; record the table's SHA-256 and the run's fingerprints here.

### Game and bid helper thinking time (not a strength experiment)

User asked for much more than 2 ms when playing the bots and for `/bid`.
`DEFAULT_SPEC` and every tournament setting are unchanged.
- `Game.BOT_SPEC` is now v2 with `time_ms=250`; `Game.ADVISOR_SPEC` v2 with
  `worlds=64,bid_worlds=128,time_ms=1000` (`BOT_TIME_MS`, `ADVISOR_TIME_MS`
  in `game.py`).
- Found: the advisor never had a time limit. In-game advice and `/bid` call
  `analyse_bids`/`evaluate_cards` directly, and only `act` starts the
  clock, so they sampled one fixed block (128 bid deals, 64 card deals),
  fewer than a 2 ms tournament turn. New `SimulationBot.start_clock()` is
  called first. Measured after: `/bid` about 1.0-1.2 s and 141,057 deals
  (5 cards, 4 players), 91,009 (2 cards, 8 players); bot turns 251 ms.
- `website.py` now takes one lock around anything that makes a bot think
  (the native workspace is not thread-safe and calls now last up to a
  second); a waiting `/api/bid-helper` request answers 409 when a newer one
  has arrived, which the page already ignores.
- Only 10 ms against 2 ms has passed a main gate. That 250 ms or 1 s plays
  better than 10 ms is **unmeasured**; the fixed network shares do not
  shrink with more deals.

Tests: `tests/test_bid_blend_fit.py` (15). Full suite 561 passed.

## Completed: Public_bots_v2 adapted-player league (13 November 2026)

User completed the fixed-source public league: 100 deal sets / 500 games,
five entries at each table, all seat rotations with lineup shuffled per seed.
Deal seed base 202611130, opening 10, bowers, cumulative down/blind/up,
standard scoring/dealer restriction, nominal 2 ms for search players, four
workers. Direct neural/heuristic players return their own inference. No
runtime error, automatic promotion gate or unseen confirmation. Keep v2.

| Entry | Mean points/game +/- descriptive 95% | Win share | Mean ms/choice |
| --- | --- | --- | --- |
| v2 | 147.54 +/-4.66 | 51.90% | 2.316 |
| Campbell OI adapted | 124.92 +/-4.39 | 30.87% | 0.648 |
| Diminishing MC adapted | 40.44 +/-4.92 | 5.07% | 2.084 |
| Diminishing MCTS adapted | 53.30 +/-4.57 | 5.00% | 2.104 |
| DRL heuristic adapted | 64.92 +/-4.11 | 7.17% | 0.081 |

Paired v2-minus-Campbell edge **+22.62 +/-6.50** points/game; versus MC
+107.10 +/-6.94, MCTS +94.24 +/-6.18, DRL heuristic +82.62 +/-6.03.
Intervals are ordinary descriptive 95% over independent seed averages, not
sequential gates. Comparisons are at shared five-player tables, not isolated
head-to-head games. This establishes superiority within this adapted lineup
and configuration, not superiority to original projects on native rules.

User asked why opponents were weak and whether they could be better balanced.
Code/data audit identifies plausible weaknesses, not causal attributions:
- Campbell uses direct probability inference, while v2 adds search. Its
  weights predate bowers; categorical counts were saturated 18,158 times
  across feature constructions (not 18,158 independent moves). Its play
  objective remains probability of exactly making, not expected score with
  the table's miss penalties. Retraining/score-aware play could help, but
  no balanced or strengthened version has been tested.
- Both Diminishing bidders round mean trick counts under random continuation
  rather than optimizing expected points. Their random rollouts, original
  future-score heuristic for descending games, bower-grid adaptation and
  tight 2 ms cut-off can all limit play. Actual times are only about 9-10%
  below v2; that alone is not evidence explaining 94-107 point gaps.
  Per-decision completed-sample counts were not recorded in this run.
- DRL is the upstream rule-based baseline, NOT its trained DQN architecture;
  no pretrained DQN checkpoint was shipped. Its dealer bids needed 612 legal
  adjustments across the run, suggesting another adaptation/calibration
  issue to examine, not proof of how much score it cost.
- Deficits are largest on longer hands. At ten cards, mean points/round:
  v2 12.11, Campbell 9.68, MC 0.09, MCTS 1.88, DRL 3.00. Blind round means
  are all 6.00 because the exact same blind policy is shared.

No claim that increasing compute or changing objectives will close the gap.
A genuinely stronger opponent revision needs score/rule-aware adaptation,
per-entry time/sample diagnostics and independent screening. A deliberately
unequal-compute challenge lineup must be labelled as such. Preserve the
original reference lineup and results, and do not rig scores to equalize it.

Evidence: `runs/external_leagues/Public_bots_v2.{meta,jsonl,json}`; diagnostic
aggregation in `Public_bots_v2.analysis.json`. Patched DLL SHA-256
9887944b81428eaca6c02a688d2f8b5572a2256c5a8d9e81673cddf86ded4111.
Pinned upstream revisions and checkpoint hashes match the prepared-league
entry. V2 weights unchanged: bid a84e9315...e29d, play 8cb5a064...d474
(full hashes in manifest). No additional tournament or training started.

## Fixed: native MCTS heap corruption in public bot league (13 November 2026)

User-run `Public_bots_v1` aborted with BrokenProcessPool after **10 saved
sets / 50 games**. Files under `runs/external_leagues/` are preserved byte-for-
byte. This is an incomplete run of a defective adapter, not evidence for
ranking or promotion. Original exact crashing worker/seed was not recorded.
Rules/settings: all five public-league entries, 5 players/opening 10,
bowers, cumulative down/blind/up, standard scoring/dealer restriction,
nominal 2 ms, workers 4, deal seed base 202611130, planned 100 sets.

Reproduced a native heap-corruption/access-violation failure inside the
DiminishingBridge MCTS code under four-worker diagnostic load. Captured a
public observation at deal seed 202611130, rotation 3, hand 10 with two own
cards remaining, trump 2 of clubs, seat 0. Exact original user's crash seed
need not match because timed search varies with load. The captured call
also failed deterministically in a fresh process with fixed budget 100.

Two defects in the pinned C++ tree code were fixed in the compatibility patch:
- Selection began at value zero and retained an earlier selectedNode when
  all child UCB values were zero. It now resets the selection and starts at
  negative infinity on every descent; missing selections return an error.
- Upward pruning reused childNumber from a removed descendant when removing
  its parent from a grandparent. It could overwrite the wrong branch slot
  and cause invalid accesses/double frees. Removal now finds the actual
  pointer's slot in its immediate parent before compacting that array.
  The captured case still failed after the first fix alone and passed after
  this ownership-slot fix, establishing a concrete heap-corruption mechanism.

No bot was dropped or replaced; no models, gameplay rules or v2 default
changed. These are MCTS correctness fixes and can change decisions, so the
old run cannot be resumed or merged with a fixed-source run. Verified that
--resume against Public_bots_v1 refuses changed source and leaves its ten
ledger records unchanged. New user-run label: **Public_bots_v2**.

Validation:
- Captured position saved as `tests/fixtures/external_mcts_pruning.json`,
  containing only own/public observation and public-derived input arrays.
  A subprocess regression passes fixed budgets 100, 1,000, 10,000, 100,000;
  before the fixes the native process faulted at the first budget.
- Four-worker diagnostic stress completed **32 deal sets / 160 full games**
  on seeds 202611130-202611161, opening 10, 2 ms, same five entries, all
  rotations and blind rounds. Per-decision diagnostic I/O was enabled, so
  this is memory-safety/completion evidence, not a strength/timing benchmark.
  No score conclusions. Report: `runs/external_crash/fixed-stress-report.json`.
- A normal-runner smoke also completed five full rotated games at opening
  10, seed 202611130, 2 ms. Evidence:
  `runs/external_leagues/External_smoke_mcts_fix_20261113.{meta,jsonl,json}`.
  No confidence or strength claim from this one-set check.
- Focused external-league tests: 24 passed, including failure-context reporting. Full regression: 546 passed.

Patched native DLL SHA-256:
9887944b81428eaca6c02a688d2f8b5572a2256c5a8d9e81673cddf86ded4111.
Pinned source/checkpoint revisions remain those in the prepared-league entry;
all root and external weights are unchanged and fingerprinted in manifests.

The normal runner now enables per-worker native fault traces and writes the
current public set/seed/rotation/round context under LABEL.diagnostics/.
BrokenProcessPool reports point to those files and the triggering context
when available. In-flight failures still stop the event; no automatic retries,
replacement bots, or mixing changed-source results. Completed sets survive.
The guide now uses Public_bots_v2. No new 100-set league was started here.

## Prepared: public Oh Hell-family all-player league (13 November 2026)

User requested a tournament between the public projects found online, allowing
rule adjustments. At the user's explicit instruction SkatZero and BotEuchre
are ignored because their trained players require different games. The
remaining event is implemented in `ohhell.external_league`, with its own
read-only dashboard and full guide `EXTERNAL_BOT_LEAGUE.md`.

Five entries at each default table:
- Current v2, unchanged root models and ordinary hybrid search.
- Campbell OI neural: NumPy port of AiStrategyModuleOI using its original
  OVL/IVL checkpoints, not locally retrained or substituted models.
- DiminishingBridge Monte Carlo and MCTS: compiled pinned upstream C++ code,
  with explicit rule, deadline, numeric and memory-safety adaptations.
- DRL-POMDP RuleBasedPlayer: executes the upstream Python class with a
  bower-aware environment shim. No pretrained DQN checkpoint is shipped in
  that repository; this entry is clearly labelled the heuristic, not DQN.

All tables use this repository's bowers, dealer restriction, standard scoring,
full cumulative down/blind/up sequence (opening 10 by default), blind-card
privacy and shared blind bidding. Original opponent hands never enter the
adapters; simulators sample their own hidden worlds. Per seed, the lineup is
shuffled then the same full game is replayed through every seat rotation.
Every bot gets every hand; relative circular order varies between seeds.

Nominal soft search allowance 2 ms, direct neural/heuristic decisions untimed
beyond their own inference. Actual means/maxima reported. This is a reference
league of adapted players, not an automatic promotion or original-game skill
ranking. Descriptive 95% intervals cluster by deal seed across rotations;
paired differences compare points at the same mixed tables, not isolated
head-to-head matches. Optional stopping is not corrected by these intervals.

Adaptations are documented in the guide and each run manifest. Notable limits:
Campbell keeps its original Ace-high-trained weights and fixed categorical
widths (saturation counted), probability-of-making objective for play, and
stable card-order ties; bidding uses the real table score function. C++ bots
use effective-suit/power rank slots, shared simulated scoring/bounds, timed
completed-sample accounting, successful MC sample means and safety fixes.
Their future-score heuristic still assumes descending rounds. The DRL dealer
wrapper picks the nearest legal bid (lower on equal distance), with adjustment
counts. Illegal cards or native failures stop the event; no fallback bot.

Pinned upstream commits:
- campbellsoup37/OhHell: 2f4d3dd3600d1df1cf439258b6150aee3f50ce7d
- thijssnelleman/DiminishingBridge: 305af70fe0c5341aeb70d5e9d2411bb57c4e7c67
- jonas-schumacher/drl-pomdp: 1acb47b18959176bc654f10b9710209307ad8098

26 upstream source/checkpoint files downloaded to gitignored
`runs/external_bots/`, with original source left intact and generated patches
separate. No licence file was found in the pinned repository roots; these
are local research downloads, not vendored redistribution. No Java or PyTorch
required. NumPy plus g++ builds the statically linked local C++ DLL.

Default five-player N5/D1/T0 upstream checkpoint SHA-256:
- ovl.txt bd512ed2fc27e1a3a0cc2e504a01d3d4e457b08737f23b53f15cef26e68819d6
- ivl.txt 9403683205c0deed0718460f30bd5794cb584f1e1cc8e05a304b72d280ee77f5
Original main.cc SHA-256:
5694afb5f000785f504853b51aec519cf83f6fe96648a49e25fab98ce1a53afa.
DRL agent.py SHA-256:
690074979edad2a4f623fcf4e55cb82fc6aef78e92d5b80a6426ddb483009c58.
Final smoke's patched DLL SHA-256:
bd4a23bc7b693c99fb2081a7a693660e7ad0bb8ac7284aa4379f5ca203dd8a69.
V2 bid/play hashes unchanged:
a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d /
8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
All table-size checkpoint, local source and library hashes are in manifests.

Completed smoke runs only, no strength conclusions:
- `External_smoke_bowers_20261113_v1`: seed 202611130, five complete rotated
  games at 5 players/opening 10, nominal 2 ms. Earlier adapter version before
  score-bound/wide-arithmetic and lineup-shuffle corrections; do not combine
  with the final version as one unchanged experiment.
- `External_smoke_bowers_20261113_final`: seed 202611131, five complete rotated
  games, 5 players/opening 10, 2 ms, all 21 rounds (including blind) finished
  legally. Saved `.meta`, `.jsonl`, `.json` under `runs/external_leagues/`.
  Scores/clocks are completion evidence only, one independent deal set.
- CLI --check-only also completed small blind-inclusive rotated games.
- Full regression **544 passed**; 22 new tests cover real upstream execution,
  scalar/network equivalence for all supported table sizes, bower winners,
  observation privacy, completed-sample bidding, extreme public-score gaps,
  explicit failure handling, dealer adjustments, full rotated games, matched
  seat/deal allocation, clustered intervals and live read-only HTTP endpoints.
  Dashboard JavaScript parsed successfully; no browser/desktop controlled.

User-run event: `python -m ohhell.external_league --sets 100 --workers 4
--label Public_bots_v1`. Default seed 202611130; 100 deal sets = 500 games.
Dashboard http://localhost:8781/ (next free port if occupied). Preparation
is automatic, or use --prepare separately. Results live under
`runs/external_leagues/Public_bots_v1.{meta,jsonl,json}`; --resume requires
identical source, models, rules, roster and settings. No long league has run
and no main/confirmation gate or promotion is inferred. Keep v2 default.

## Worse: banked time with early stopping (13 November 2026)

Completed user-run tournament `Search_bank_20261113-120050-788211`.
Candidate `hybrid:field=previous,stop_z=3,controller=bank,time_ms=2.0` versus
experimental baseline `hybrid:field=previous,stop_z=3,time_ms=2.0`. Other
simulating lobby versions also used the early-stopping baseline. Only the
controller setting changed, but it changes BOTH time allocation and actual
average budget through repayment of soft-deadline overruns.

Bowers, cumulative down/blind/up, opening min(10, 51/n), 3-8 players,
standard scoring/dealer restriction. Nominal 2 ms, seed 1302026113, planned
maximum 1,300 sets, smallest gain 1.0, mean-time-ratio cap 1.05.

- Stopped at **780 sets / 3,660 games**, verdict **worse**:
  **-4.35 +/-3.84 points per game**, sequential-gate margin. No confirmation.
- Win-share edge -0.01848 +/-0.02399 (ordinary 95% over sets), no dashboard
  disagreement. Actual mean candidate/baseline **2.008 / 2.308 ms**, about
  **13% less elapsed time for the banked candidate**. This is not a clean
  equal-mean-time test of allocating time across decisions. The bank repays
  overruns while the fixed controller keeps receiving a fresh soft allowance.
  Reduced compute is a plausible contributor to the loss; its contribution
  versus allocation mistakes has not been measured separately.
- Per-size edges (sets): 3p -5.31 (120), 4p -4.39 (360), 5p -5.13 (120),
  6p -4.42 (60), 7p -4.38 (60), 8p -0.60 (60). Per-size margins wide.
- Manifest bid/play SHA-256:
  a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d /
  8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
  Source SHA-256 17bfc80df8f29b9bbc5d767c5127a5078ca44663be6740be95bcd7a3a09cb0e9.
- Evidence: `runs/gates/Search_bank_20261113-120050-788211.json`, `.jsonl`,
  `.meta`. No per-decision bank balance/allocation telemetry or equal-actual-
  budget control. **Reject this setting, keep v2, do not repeat unchanged.**

### Review of experiment prioritization

The user asked how many more and why the candidates were poor. Four stages
remain unrun: policy-self, policy-all, typing, policy-inference. Do not
mechanically recommend all four next. The efficiency candidates were sound
mechanical prototypes but had insufficient strength-screening evidence for
a succession of expensive user-run tournaments. In particular dedupe had
already failed to increase counted-world throughput in the small check;
early stopping saved only 1.5%; banking introduced an actual budget mismatch.
The compact policy has only 46.27% teacher-best agreement and costs about
five times the sampled worlds at 2 ms. Those are reasons to screen/refine it
before recommending a full tournament, not proof that learned continuations
cannot help. Opponent typing/likelihood also need predictive checks first.

Only extra-time search has passed a main gate (+7.62 points at 10 vs 2 ms;
no unseen confirmation). None of the five efficiency/allocation candidates
has demonstrated a strength gain. Recommendation: end the automatic test
queue and require a concrete offline/timing case for another full run.
No further tournament started, no strategy/default changed.

## Inconclusive: paired-gap early stopping at 2 ms (13 November 2026)

Completed user-run tournament `Search_early-stop_20261113-114133-336361`.
Single change: `hybrid:field=previous,stop_z=3,time_ms=2.0` versus unchanged
v2 `hybrid:field=previous,time_ms=2.0`. Minimum 32 draws; heuristic gap of
one point plus three estimated standard errors against every alternative,
checked at block boundaries. Root prior is not used in the stopping test.
Exact endgame stopping unchanged. No bank, duplicate reuse, touching-rank
shortlist or evidence-weighted blend combined.

Bowers, cumulative down/blind/up, opening min(10, 51/n), 3-8 players,
standard scoring/dealer restriction, both 2 ms. Seed 1302026113, planned
maximum 1,300 sets, smallest gain 1.0.

- **1,300 sets / 6,100 games**, final verdict **inconclusive**:
  **-0.40 +/-2.31 points per game**, tournament sequential-gate margin.
  Main gate did not pass; no unseen confirmation triggered.
- Win-share edge +0.00510 +/-0.01923 (ordinary 95% over sets), no dashboard
  disagreement. Mean candidate/baseline **2.426 / 2.463 ms per choice turn**,
  a measured saving of about 1.5% in this run, not a general speed guarantee.
- Per-size edges (sets): 3p +0.20 (200), 4p -0.78 (600), 5p -1.34 (200),
  6p +1.18 (100), 7p -1.21 (100), 8p +1.73 (100). Per-size margins wide.
- Manifest v2 bid/play SHA-256:
  a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d /
  8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
  Source SHA-256 17bfc80df8f29b9bbc5d767c5127a5078ca44663be6740be95bcd7a3a09cb0e9.
- Evidence: `runs/gates/Search_early-stop_20261113-114133-336361.json`,
  `.jsonl`, `.meta`. No demonstrated strength gain or unseen confirmation;
  no stop-frequency, confidence-calibration or per-position sample telemetry.
- **Keep v2; do not rerun this unchanged candidate.**

Next prepared comparison: `python -m ohhell.learn.search_experiment bank`.
Both candidate and baseline use stop_z=3; only candidate adds controller=bank.
This tests redistributing saved time, not repeating early stopping alone.
The early-stopping baseline remains unpromoted. The mean-time gate requires
candidate mean at most 105% of baseline, including any confirmation; a win
against this experimental baseline alone would not justify default promotion.
Dashboard http://localhost:8780/. No next tournament started by the agent.

## No gain: evidence-weighted card blending at 2 ms (13 November 2026)

Completed user-run tournament `Search_play-evidence_20261113-112751-861325`.
Single change: `hybrid:field=previous,play_evidence=1,time_ms=2.0` versus
unchanged v2 `hybrid:field=previous,time_ms=2.0`. Root card blending counts
actual completed draws instead of configured worlds=32; prior_worlds=16
unchanged. Ordinary shortlist, inference, bidding and exact endgames kept.
No duplicate reuse, touching shortlists or bid-evidence change combined.
Bowers, cumulative down/blind/up, opening min(10, 51/n), 3-8 players,
standard scoring/dealer restriction, both 2 ms, seed 1302026113, planned
maximum 1,300 sets, smallest gain 1.0.

- Stopped at the planned **1,170-set check / 5,490 games**, verdict **no gain**:
  **-1.92 +/-2.53 points per game**, tournament sequential-gate margin.
  The gate rules out the targeted +1-point improvement at this check; it
  does not establish a statistically certain loss. No confirmation triggered.
- Win-share edge -0.00768 +/-0.02051 (ordinary 95% over sets), no dashboard
  disagreement. Actual mean candidate/baseline time **2.440 / 2.442 ms**.
- Per-size edges (sets): 3p -5.57 (180), 4p -2.36 (540), 5p -2.98 (180),
  6p +2.20 (90), 7p +2.75 (90), 8p +1.40 (90). Per-size margins wide.
- Manifest v2 bid/play SHA-256:
  a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d /
  8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
  Source SHA-256 17bfc80df8f29b9bbc5d767c5127a5078ca44663be6740be95bcd7a3a09cb0e9.
- Evidence: `runs/gates/Search_play-evidence_20261113-112751-861325.json`,
  `.jsonl`, `.meta`. No demonstrated improvement or unseen confirmation.
  Draw count is not effective sample size; this does not disprove all
  possible uncertainty-aware blending rules.
- **Keep v2; do not repeat this unchanged candidate.**

Next independent prepared comparison: `python -m ohhell.learn.search_experiment
early-stop`, heuristic paired-gap stopping only, without a time bank or any
of the three tested efficiency options. Dashboard http://localhost:8780/.
No next tournament was started by the agent.

## Inconclusive: touching-rank shortlist at 2 ms (13 November 2026)

Completed user-run tournament `Search_touching_20261113-111343-746903`.
Single change: `hybrid:field=previous,touching=1,time_ms=2.0` versus unchanged
v2 `hybrid:field=previous,time_ms=2.0`. Model-preferred representatives from
touching-rank groups fill shortlist slots. No duplicate reuse or evidence
blending combined. Bowers, cumulative down/blind/up, opening min(10, 51/n),
3-8 players, standard scoring/dealer restriction, both 2 ms. Seed 1302026113,
planned maximum 1,300 sets, smallest gain 1.0.

- **1,300 sets / 6,100 games**, final verdict **inconclusive**:
  **-0.58 +/-2.28 points per game**, tournament sequential-gate margin.
  Main gate did not pass; no unseen confirmation triggered.
- Win-share edge +0.00144 +/-0.01916 (ordinary 95% over sets), no dashboard
  disagreement. Actual mean candidate/baseline time **2.389 / 2.388 ms**.
- Per-size edges (sets): 3p -2.43 (200), 4p -1.05 (600), 5p +0.94 (200),
  6p -0.98 (100), 7p +0.93 (100), 8p +1.85 (100). Per-size margins wide.
- Manifest v2 bid/play SHA-256:
  a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d /
  8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
  Source SHA-256 17bfc80df8f29b9bbc5d767c5127a5078ca44663be6740be95bcd7a3a09cb0e9.
- Evidence: `runs/gates/Search_touching_20261113-111343-746903.json`,
  `.jsonl`, `.meta`. Missing evidence: no demonstrated strength improvement,
  no confirmation or detailed shortlist/sample telemetry. This does not
  establish that touching ranks are equivalent under rank-sensitive play.
- **Keep v2; do not repeat this unchanged candidate.** No gain established.

Next independent prepared comparison: `python -m ohhell.learn.search_experiment
play-evidence`, using actual completed draws in the root play blend only.
It keeps ordinary shortlist/sampling and does not change bid blending.
Dashboard http://localhost:8780/. No next tournament started by the agent.

## Inconclusive: duplicate-deal reuse at 2 ms (13 November 2026)

Completed user-run tournament, `Search_dedupe_20261113-105825-584020`:
`hybrid:field=previous,dedupe=1,time_ms=2.0` versus unchanged v2
`hybrid:field=previous,time_ms=2.0`. Only deterministic duplicate-deal reuse
changed. Bowers, cumulative down/blind/up, opening min(10, 51/n), 3-8 players,
standard scoring/dealer restriction, both 2 ms. Seed 1302026113, planned
maximum 1,300 sets, smallest gain 1.0.

- Completed **1,300 sets / 6,100 games**. Final verdict **inconclusive**:
  **+0.18 +/-2.27 points per game**, tournament sequential-gate margin.
  Main gate did not pass; no unseen-deal confirmation was triggered.
- Win-share edge -0.00544 +/-0.01908 (ordinary 95% over sets), no dashboard
  disagreement. Actual mean candidate/baseline time **2.302 / 2.298 ms**.
- Per-size edges (sets): 3p +0.35 (200), 4p +1.09 (600), 5p -4.69 (200),
  6p +2.62 (100), 7p +1.16 (100), 8p +0.65 (100). Wide per-size margins;
  no consistent improvement established.
- Manifest weights match v2: bid
  a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d,
  play 8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
  Source fingerprint 17bfc80df8f29b9bbc5d767c5127a5078ca44663be6740be95bcd7a3a09cb0e9.
- **Keep v2; do not rerun this unchanged candidate.** Removing repeated
  rollouts did not establish stronger timed play. The earlier isolated
  timing check also showed no counted-world throughput improvement.
- Evidence: `runs/gates/Search_dedupe_20261113-105825-584020.json`, `.jsonl`,
  `.meta`. Missing evidence: no demonstrated equal-time strength advantage
  or confirmation, and no tournament sample-count telemetry.

Next independent prepared candidate: touching-rank shortlist representatives
(`python -m ohhell.learn.search_experiment touching`), retaining ordinary
sampling/blending. This is a separate behaviour change, not combined dedupe.
No next tournament was started by the agent.

## Time scaling: main gate passed at 10 ms versus 2 ms (13 November 2026)

User stopped the time-scaling run after 390 sets / 1,830 games because the
extra time improved play. Saved rows 0-389:
`runs/gates/Search_time_20261113-104915-773629.jsonl` and its `.meta`.
There is no final tournament summary and no unseen-deal confirmation.
Reconstructed analysis: same basename with `.analysis.json` (an analysis,
not an official completed tournament verdict).

- Candidate `hybrid:field=previous,time_ms=10.0`; unchanged v2 baseline
  `hybrid:field=previous,time_ms=2.0`; all other simulating lobby seats 2 ms.
  Bowers, cumulative down/blind/up, opening min(10, 51/n), 3-8 players,
  standard scoring and dealer restriction. Seed 1302026113, planned maximum
  1,300 sets, smallest gain 1.0. Native timing verified by recorded clocks.
- Main-stage edge **+7.62 points per game**, planned sequential gate margin
  **+/-7.51** at the 390-set check: **main gate passed**. Plain descriptive
  95% margin +/-3.86; this does not replace the sequential gate margin.
  Earlier planned checks at 130 and 260 sets had not passed.
- Win-share edge +0.0689 +/-0.0347 (ordinary 95% margin over sets).
  Candidate/baseline actual mean times **10.72 / 2.79 ms per choice turn**.
- Edges by table size (sets): 3p +6.33 (60), 4p +9.64 (180),
  5p +7.73 (60), 6p +6.72 (30), 7p +3.90 (30), 8p +2.50 (30).
  Per-size margins are wide; all point estimates are positive.
- Manifest bid/play hashes match current v2:
  a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d /
  8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474.
  Source fingerprint 17bfc80df8f29b9bbc5d767c5127a5078ca44663be6740be95bcd7a3a09cb0e9
  matches the current implementation when inspected.
- Reading: this is valid evidence that extra native search time helps in
  this lobby, so reducing work/noise at the existing allowance is a useful
  next direction. It does not establish that duplicate reuse, touching
  shortlists, play evidence or learned continuations improve strength.
  **No promotion:** unequal time and no unseen confirmation. Keep v2.
  Next prepared isolated comparison: `python -m ohhell.learn.search_experiment
  dedupe`, dashboard http://localhost:8780/. User runs it; none started here.

## Prepared: search components and learned continuations (13 November 2026)

User requested working through all seven items in the supplied proposal.
Implementation, exact settings, independent comparisons and user commands:
`BOT_SEARCH_WORK.md`. **No tournament or paired strength comparison was run.
Keep v2 unchanged.** Earlier uncommitted style/cash work was preserved.

Separate opt-in candidates: deterministic duplicate reuse (`dedupe=1`),
touching-rank shortlist representatives (`touching=1`), actual completed-draw
play blending (`play_evidence=1`), heuristic paired-gap early stopping
(`stop_z=3`, minimum 32 draws), banked decision time (`controller=bank`),
compact learned continuations (`policy=learned_self`, then `learned_all`),
persistent seat typing (`typing=1`) and normalized learned play likelihood
(`policy_inference=1`, softmax temperature 10 round points). All default off.

- Duplicate reuse merges resampling indices and identical hands-plus-kinds
  within a native block, preserving multiplicities. Stochastic fields keep
  their work/order. Python caches deterministic repeats. Fixed-budget values
  and RNG states match, including exact endgames. Touching cards are NOT
  proven rollout equivalents: rank-sensitive continuations distinguish them.
  That shortlist rule is a separate behaviour change, not a free optimization.
- Play evidence uses draw counts, not effective sample sizes; inference
  resamples correlated worlds. This differs from the earlier inconclusive
  bid-evidence and adaptive-all-card candidates and combines neither.
- Early stopping needs a one-point gap plus three estimated standard errors
  against every other root action, at block boundaries. It is heuristic,
  not a sequential confidence guarantee, and ignores the root model prior
  when deciding to stop. Exact endgames retain normal stopping.
- The bank earns the nominal allowance only on decisions with a choice,
  pays elapsed time, retains overrun debt, caps savings at four allowances.
  Soft per-deal overruns remain. The bank launcher compares against the same
  early-stopping baseline, changing only banking. `--mean-time-ratio 1.05`
  withholds `better` above 105% of baseline measured mean time even after
  confirmation. Both stages' clocks and opponent-reading work count. This
  tolerance is not a statistical equivalence test or comparison to plain v2.
- Learned policies use only the acting hand and public information in a
  C/Python 32-32-16-1 float MLP. They change continuations in BOTH bid and
  card simulations; keep root MLP, shortlist, blend, kind noise and exact
  solver. Architecture/representation/distillation jointly define this
  checkpoint, not an incremental NNUE or a single-factor test of internals.
  Learned-all is compared against learned-self, not directly promoted v2.
- Typing reconstructs opening hands only from fully public completed-round
  history, excluding our seat and blind rounds. Floors/bounded log odds
  permit recovery. The same seat prior feeds inference and simulated kinds
  in both backends. Arena, tournament and web game provide the round hook;
  standalone training loops without it cannot evaluate persistent reading.
  Learned inference alone keeps heuristic continuations. Softmax human
  action probability calibration is unmeasured.
- Position bench `ohhell.learn.position_bench` evaluates every root action
  on observation-consistent sampled worlds, never the saved actual opponent
  hands; stratifies by table/hand size, split and lead/follow. Default 512
  draws, two seed replicates. Reports reference-value gaps by existing case
  tags, with deal-grouped paired means, NOT actual game points lost or
  perfect-play error. Shared model bias/strategy fusion remain. Isolated
  positions cannot assess banking or cross-round reading. `train_rollout
  --labels` prepares search/refit iterations; no policy-improvement iteration
  has been run yet.
- Specs select sampler, rollout policy, root evaluator and controller.
  `fast=1` now requires the matching native core; explicit `fast=0` is the
  Python oracle, rejected by timed tournaments. Bindings now copy observed
  scoring into native Params, fixing custom-scoring parity. Manifests hash
  implementation source plus root/rollout weights; old source-less manifests
  cannot resume (new labels required). Dashboard handles unequal timing and
  over-time verdicts. C workspaces remain process-local, not thread-safe.

Completed data/training (no turn allowance): fresh bower bank, seed 20261113,
four repeats, all 2-8 player tables/legal hands through 10 plus blind 0,
272 deals, 27,280 snapshots (5,120 bids / 22,160 plays). Independent rounds,
related snapshots kept in one split. Command:
`python -m ohhell.learn.cases build --seed 20261113 --repeats 4 --max-hand 10
--out runs/cases/bowers-search-20261113.jsonl`. The older predetermined-v1
bank predates bowers and was deliberately not used for training.

Training: `python -m ohhell.learn.train_rollout --bank
runs/cases/bowers-search-20261113.jsonl --out ohhell/models/rollout_candidate
--epochs 40 --seed 20261113`. v2 teacher, Huber/Adam, batch 1024, rate .001,
validation-selected weights, 3.95 s preparation plus fitting on this host.
Choice-only rows: train 34,485 (10,210 positions / 177 deals), validation
4,918 (1,387 / 20), test 3,197 (966 / 19). Validation Huber .33644 to .05773.
Unseen test RMSE 9.596 points, teacher-best agreement 46.27%, mean teacher
choice gap 1.376 points. These are fidelity metrics, NOT playing strength.
Candidate `ohhell/models/rollout_candidate/play.npz`, SHA-256
`ca5a1ab4ec2c25bbfd7676b8be0a36f533936447902368c38ea94a56ba77ebb9`;
recipe, bank hash and teacher/source fingerprints in its `training.json`.
Default bid/play files were not overwritten:
`a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d` /
`8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474`.

Isolated card-turn diagnostics, NOT a paired comparison or stable benchmark:
seed 20261113, 24 choice positions per spec, 3-8 players, starting hand
min(10, 51/n), bowers/standard scoring, heuristic-generated public histories.
No other tests/tournaments launched concurrently; external CPU load unknown.
Native `core_f15c0ebc61508cda.dll`; every spec uses field=previous/v2 weights.

| Addition to hybrid:field=previous | Mean elapsed ms | Counted / executed worlds |
| --- | --- | --- |
| time_ms=2 | 2.143 | 158.0 / 158.0 |
| time_ms=10 | 10.187 | 881.2 / 881.2 |
| time_ms=2,dedupe=1 | 2.166 | 148.1 / 110.5 |
| time_ms=2,policy=learned_self | 2.124 | 33.5 / 33.5 |

More time completes more native draws here; STRENGTH scaling remains unknown.
Duplicate reuse removed rollouts but did not increase throughput in this
small check. Learned-self costs roughly five times as many sampled worlds
at 2 ms; whether quality compensates is entirely unmeasured. Report:
`runs/cases/search-throughput-20261113.json`, including source/model hashes.

Completion smoke: one trained learned-self seat versus v2, 2 ms each,
seed 17, 3 players/opening 10, 4/10, 8/6, bowers, standard scoring/dealer
restriction, cumulative down/blind/up. All 21/21/13 rounds completed. Saved
`runs/cases/search-components-completion-20261113.json`. No control games,
edge or performance claim. Tests also complete blind-inclusive games in
both backends with learned inference, typing and banking.

Position-bench CLI smoke: 96 stratified choices from the fresh bank,
reference `hybrid:field=previous,worlds=32`, two seed replicates, untimed;
`runs/cases/search-reference-smoke-20261113.jsonl`. Identity evaluation
`hybrid:field=previous,worlds=8` against itself on 16 test positions / eight
deals: exactly zero paired reference-value edge. Report:
`runs/cases/search-reference-identity-20261113.json`. Plumbing check only.

Launcher `python -m ohhell.learn.search_experiment STAGE` builds C and uses
a fresh label; dashboard <http://localhost:8780/> (next free port if busy).
All ten stages passed `--check-only`, without playing games. Start `time`
(candidate 10 ms, others 2 ms); then independent `dedupe`, `touching`,
`play-evidence`, `early-stop`, `bank`, `policy-self`, `policy-all`, `typing`,
`policy-inference`. Default seed 1302026113, workers 6. See guide for baseline
exceptions. Time-only results cannot justify an equal-time promotion.
No long comparison started; main gate and unseen confirmation are missing.


Verification: full regression suite 521 passed; then 35 focused component
tests passed, including one additional existing-selfplay-generator check
using the learned continuation. C passed -Wall/-Wextra syntax checks;
dashboard JavaScript parsed successfully; git diff --check passed. New
bank/reference training tools reject sources without a bowers marker.

## Negative at screening: always cash sure side winners (13 November 2026)

Follow-up to the entry below. v2 leads something else 43-57% of the time
when it is on lead, wants tricks and holds a side-suit card no unseen card
beats. Test of whether that is a leak: opt-in `cash` on `HybridBot`
(`ohhell/bots/learned.py`) overrides the search in that spot and leads the
highest such card. 1 always, 2 only when the call is above zero, 3 unless
it is below zero. Exact endgames are untouched. Default 0, so v2 is
unchanged. **No tournament was run and none is proposed. Keep v2.**

- 4 players, opening hand 10, down/blind/up, paired on the same deals with
  `python -m ohhell.learn.call_report`, hybrids **untimed**
  (`hybrid:field=previous`), points per game against the same bot without
  `cash`, 95% intervals over games.

  | Setting | Opponents | Games, first seed | Result |
  | --- | --- | --- | --- |
  | `cash=1` | skilled, casual, novice | 6,000, 3500000 | -1.76 +/- 1.74 |
  | `cash=2` | same | same | +0.01 +/- 1.08 |
  | `cash=3` | same | same | -0.28 +/- 1.13 |
  | `cash=1` | hybrid, hybrid, casual | 1,500, 3600000 | -3.49 +/- 3.49 |
  | `cash=2` | same | same | -1.91 +/- 2.88 |

- Reading: holding back a sure side winner is not a leak in v2; forcing the
  lead costs points when always on and gains nothing in over-called rounds.
- Missing evidence: nothing at 2 ms, only 4 players, no tournament.
- Tests: two added to `tests/test_style.py`.

## No gain at screening: over-call play rules (12 November 2026)

User's idea: when the table bids more tricks than exist, play more
aggressively. Tested as opt-in rules in the rule-based player
(`ohhell/bots/heuristic.py`, mirrored in `core.c`), default off, so v2 is
unchanged. **No tournament was run and none is proposed. Keep v2.**

- `style` is a sum of bits, active only while the seat wants tricks and the
  call is above zero (call = tricks still wanted by every seat minus tricks
  left, `table_call`): 1 follow suit with the top winner before the last
  seat; 2 ruff with the lowest trump that has at most two unseen trumps above
  it (none when the call is 2 or more), never a top trump; 4 with no sure
  winner, lead a side card that exactly one unseen card beats.
- `heuristic:style=N` plays that way itself. `hybrid:...,style=N` applies it
  to our own seat's continuation in simulations only; `style_seats=all`
  applies it to every simulated seat and to the play likelihood in inference.
  This differs from `react=1`, which changes two follow-suit cases for every
  seat. Native core `core_48a49a57c4272b8e.dll`.
- New tool `python -m ohhell.learn.call_report SPEC...`: each spec takes
  seat 0 of the same games; reports points per game, the paired difference
  against the first spec, and make / too many / too few by call.
- All results below: 4 players, opening hand 10, down/blind/up, paired on the
  same deals, 95% intervals over games, points per game. Hybrids were
  **untimed** (`hybrid:field=previous`, fixed 32/64 deals), not 2 ms.

  | Seat 0 | Opponents | Games, first seed | Against the plain bot |
  | --- | --- | --- | --- |
  | `heuristic:style=1` | skilled, casual, novice | 6,000, 3000000 | -0.00 +/- 0.18 |
  | `heuristic:style=2` | same | same | +0.19 +/- 0.30 |
  | `heuristic:style=4` | same | same | -0.18 +/- 0.24 |
  | `heuristic:style=7` | same | same | +0.06 +/- 0.41 |
  | hybrid, `style=5` | skilled, casual, novice | 8,000, 3200000 | -0.21 +/- 0.76 |
  | hybrid, `style=7` | same | same | -0.15 +/- 0.95 |
  | hybrid, `style=1,style_seats=all` | hybrid, hybrid, casual | 3,000, 3400000 | -1.11 +/- 2.29 |
  | hybrid, `style=5,style_seats=all` | same | same | +1.46 +/- 2.44 |
  | hybrid, `style=7,style_seats=all` | same | same | -0.45 +/- 2.52 |

  An earlier 1,500-game hybrid run (seed 3100000) gave +0.7 to +1.5 for
  styles 1, 2, 4 and 5, all inside their margins; the 8,000-game run did not
  repeat it. Bit 2 made the rule bot fall short more often (28.1% to 30.0%
  of rounds at call 3+).
- What v2 already does (1,500 untimed games, seat 0 against skilled, casual
  and one hybrid, seed 3300000, rounds of 5+ cards): it wins with a
  non-certain card in an early seat 63% of the time in over-called rounds
  and 55% in under-called ones (the rule bot never does); last to play with
  two or more winners it uses the lowest 54% over-called and the highest 46%
  under-called (rule bot: always lowest); it ruffs with its lowest trump 52%
  over-called and 26% under-called. So the search and network already shift
  play with the call, without any rule.
- v2 falls short about three times as often as it takes too many in
  over-called rounds (30% against 10%). This is not evidence of timid play:
  the call includes v2's own bid, and the scoring pays more for a made higher
  bid, so a sound bidder misses low more often than high.
- Likely reason for no gain (untested): each real turn is searched afresh, so
  the assumed continuation only shifts card values slightly, and the rules
  fire rarely.
- Missing evidence: nothing at 2 ms, no other table sizes for the hybrid, no
  tournament. Under-call rules and lead-timing rules were planned with the
  same mechanism and were not built.
- Tests: `tests/test_style.py` (19). Full suite: 485 passed.

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

**Since 16 November 2026 the default is v3 "No reading"**,
`hybrid:field=previous,inference=0,time_ms=2` (see the entry at the top);
the game, advisor and bid game keep reading at their longer times.
Before that, the default was **v2 Self-play bowers nets**,
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


## Completed: own-bid-only fitted estimate, 50 ms (19 November 2026)

Single strategy change: `bid_estimate=3` replaces only our own bid shortlist estimator with the existing fitted v3 formula. Opponent bidding, inference, network inputs, blend and card play are unchanged. This isolates the estimator from the earlier estimate=2/3 experiments.

Candidate `hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25,bid_estimate=3` against `hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25`. Seed 2026111903, allowance 50 ms, 18 workers, calibrated fixed budgets (interleaved timing), max 60,000 main rounds, 10 Oâ€™Brienâ€“Fleming looks, smallest gain 2 points/game. Bowers, down/blind/up round weights, 3â€“8 players with tournament weights; field previous hybrid x3 / skilled / casual.

**Verdict: no gain**. Main 18,000 rounds: -0.293 points/game, gate margin +/- 4.254; diverged 18.94%. Mean decision ms candidate / baseline: 287.81 / 221.97.
No unseen-deal confirmation: main gate did not pass.
Source SHA-256 `74372cf6c4a53c7c2947ca434bbe3acce6de23bcad3df2f99ea3f5228e516bce`. Model/coefficient SHA-256 fingerprints: `{"C:\\Users\\dbliss-henaghan\\Documents\\Github\\10-1\\ohhell\\models\\bid.npz": "a84e9315a14bdbeabd3964df5fa315ac64780ef09553e404b90cf33ae022e29d", "C:\\Users\\dbliss-henaghan\\Documents\\Github\\10-1\\ohhell\\models\\play.npz": "8cb5a064fe30ad03904912d0a88941da248f4507e341bdbfefdb6f187fe9d474", "C:\\Users\\dbliss-henaghan\\Documents\\Github\\10-1\\ohhell\\models\\trick_estimate3.json": "20971772043ad011255786c70128c195eeb193423d64c1b525ce85c5891aee7e"}`.
Calibration: `{"load": {"bid": 0.6738293963867046, "card": 0.6319058184724786, "endgame": 0.7318453209859351}, "rates": {"bid": 178.82, "card": 103.66, "endgame": 133.6}, "reference": "hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25", "rounds": 300, "specs": {"hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25": {"budgets": {"bid": 8941, "card": 5183, "endgame": 6680}, "measured": {"bid": [6024.708633093525, 278], "card": [3275.167857142857, 560], "endgame": [4888.726744186047, 344]}}, "hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25,bid_estimate=3": {"budgets": {"bid": 9066, "card": 5144, "endgame": 6744}, "measured": {"bid": [6108.6474820143885, 278], "card": [3250.3772241992883, 562], "endgame": [4935.457478005865, 341]}}}, "time_ms": 50.0}`.
Results `runs/duels/50ms_own_bid_estimate_v3.{meta,jsonl,json,log}`. Defaults unchanged. A 50 ms result alone does not establish strength at the gameâ€™s 750 ms allowance.

Stopping detail for this run: the no-gain verdict came from the default
futility rule, not the wider sequential gate interval. At 18,000 rounds,
plain 95% upper bound = -0.29278 + 1.96 * 149.79137 / sqrt(18000)
= +1.896 points/game, below the predeclared 2-point smallest gain.
The sequential gate interval was about [-4.55, +3.96]. No proof of
zero effect or harm; insufficient grounds for promotion. The 250 ms
stage was correctly skipped. Wall-clock means were affected by machine
load changes; allowances used calibrated fixed deal budgets, not live
50 ms deadlines.

Post-run action audit (all 18,000 saved pairs): 14,591 (81.06%) had identical actions, 102 (0.57%) first differed in bidding, and 3,307 (18.37%) first differed in card play after identical bids. Weighted contributions to the overall edge were +0.378 points/game from bid-first pairs and -0.671 from play-first pairs. This is a descriptive split, not a causal estimate of either policy. The calibration gave different card and endgame budgets despite the same card policy. A follow-up should match those budgets exactly and measure missed-bid frequency and point cost against an all-legal reference before testing a new shortlist.


## Measured: current bid shortlist against every legal bid (19 November 2026)

Baseline: current game bot `hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25`.
6,000 fresh bid positions starting at seed 947321000, sampled with the duel's
3-8-player and down/blind/up round weights, bowers and dealer restriction.
Earlier bids came from the standard duel field (previous hybrid x3, skilled,
casual). At each pruned position, ranked all legal bids together using the
current simulation/network blend and 8,192 sampled deals, with no wall-clock
limit. Each nominated missed bid and best shortlisted bid was scored again
on 8,192 independent draws. Blind is included in the denominator and uses
its existing exact bid calculation. Worked examples and scores saved in
`runs/bid_shortlist/audit_947321000.{jsonl,json,meta}`. Source SHA-256
a7869cf4dde4dbac..., bid.npz a84e9315..., play.npz 8cb5a064... (full
hashes in manifest). Smoke run: 24 positions, seed 947320000, 256 deals,
same rules and field, output `runs/bid_shortlist/audit_smoke.*`.

3,610 positions had a pruned shortlist. Seven of 6,000 (0.12%) nominated
an excluded bid, all above the range, and all seven scored higher on the
second draws. Six were 10-card rounds and one was an 8-card round. Five
winners were one above the range and two were two above. The largest
verified gap on a single hand was +17.05 points. Summing the held-out
gaps with the duel's game-length weights gives +0.136 points per game
(approximate 95% margin +/-0.144; rare-event estimate). This is the value
of the detected misses under the bot's own simulator and bid network,
with other positions assigned zero. It does not estimate realised match
points or the cost of fewer draws per bid with a fixed clock. No duel
started: the measured opportunity was well below the usual 2-point target.


## Measured: current bid search wall-clock scaling (19 November 2026)

Single-process diagnostic, current game bot (the duel's REFERENCE_SPEC),
24 sampled round seeds from 947330000, 23 ordinary bids measured after
excluding blind. Standard weighted down/blind/up, bowers, dealer rule and
duel field bidders before the tested seat. Same observation and bot seed
across 2/10/50/250/750 ms real deadlines, order rotated by seed. Reference
is the same bot and shortlist with 65,536 draws on an independent seed.
Median observed ms: 2.25 / 10.42 / 50.33 / 250.39 / 750.36. Median deals:
446 / 2,241 / 10,152 / 57,171 / 171,347. Agreement with reference:
21/23 at 2 ms, 23/23 at every longer deadline. Mean reference expected
loss at 2 ms was 0.080 points per bid position. Short sample, no realized
match scores or evidence of optimal play. No configuration change.
Bid/play hashes a84e9315... / 8cb5a064..., source a7869cf4dde4dbac...
(same as preceding audit). Rows: runs/bid_shortlist/time_probe_947330000.json.


## Measured: adaptive stopping for bid search (19 November 2026)

Opt-in candidate adds `bid_stop_z=4` to the current game hybrid
(`hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25`).
The normal network contribution is a fixed offset. The leading bid must
clear every rival by 0.25 points plus four estimated standard errors of
the paired score difference. The final candidate estimates uncertainty
across full resampling blocks, with at least eight blocks and 512 draws
by default. The bid may run to the configured cap on close calls. This
is a stopping heuristic with repeated looks, not a confidence guarantee.
Default is off. No time transfer to card play.

Paired position benchmarks: 400 drawn rounds at 50 ms and a separate 96
at 750 ms in each probe, eight workers. Same observation and seed within
a pair, alternating order, observed wall-clock time. Standard weighted
3-8-player down/blind/up rules, bowers and dealer restriction. Earlier
bids from the standard duel field. Blind positions skipped. Reference:
current game bot on an independent stream of 65,536 draws, same shortlist
and network blend. Full model hashes in both manifests (bid a84e9315...,
play 8cb5a064...). Benchmark script runs/bid_shortlist/bid_stop_probe.py.

- Initial per-draw variance, minimum 256, first seed 947334000, source
  3b4798b0..., DLL core_9f6dfaedfde687f0.dll. At 50 ms, mean elapsed
  50.64 -> 9.66 ms, two different bids in 375; both worse by the reference
  (0.66 and 7.99 points). At 750 ms, 751.40 -> 52.38 ms, 0/89 different.
  Superseded by the block variance design. Files
  runs/bid_stop/probe_947334000.{meta,jsonl,json}.
- Final block variance, first seed 947335000, source 77e9f5fe...,
  DLL core_8721fd253db3a265.dll. At 50 ms, mean 50.58 -> 11.27 ms,
  median 50.57 -> 5.03 ms, p95 50.80 -> 50.17 ms. Two differences in 382,
  and both policies matched the reference on 381/382. Mean reference
  gap 0.00024 -> 0.00088 points per position. At 750 ms, mean
  750.61 -> 61.78 ms, median 750.59 -> 4.70 ms, p95 750.84 -> 613.42 ms.
  All 90 bids identical and match the reference. Replayed the two earlier
  50 ms misses and the revised rule matched the reference on both.
  Files runs/bid_stop/probe_947335000.{meta,jsonl,json}. After the probe,
  restored the existing BidAnalysis sample-metadata behavior. The measured
  decisions and native stopping logic are unchanged.

This measured bid latency and reference agreement. Realised game score,
unseen-deal duel confirmation, and the benefit of reallocating saved time
remain unmeasured. Bid draws share the simulator RNG with card play, so
a duel that seeks to attribute effects to bid selection should also
control card RNG and budgeting. No promotion.
Final verification: 97 targeted tests passed. Full suite: 712 passed, with four previously recorded stale v2 default assertions failing.


## Reverted bid stopping and checked forecast calibration (19 November 2026)

User requested removal of adaptive bid stopping. Reverted the code and its
tests, keeping the saved measurements as experiment history. The restored
native core builds and 64 relevant tests pass. The earlier independent
bid-estimator candidate and shortlist audit are retained.

Read-only diagnostic of the existing holdout runs/value_bid/confirm_t10.jsonl.
25,000 drawn rounds, seed starting 7100000, 10 ms, historical v2 lobby and
standard weighted table sizes under bowers (see learned full-deal value
entry). Reused the existing saved forecasts, outcomes and model weights.
For each of 24,511 saved positions, selected the bid with highest expected
points using the saved 50/50 simulation/network blend, then compared raw
simulation and network forecasts for that same bid with the recorded
outcome in a lobby playout. On 6,091 positions with 8-10 cards, simulation
predicted 55.1% exact bids and network 55.0%, compared with 62.8% made.
Simulation expected 2.479 tricks per hand, compared with 2.373 taken.
Across all rows, simulation predicted 65.4% exact bids versus 68.8% made.
Trick-count log loss (probability floor .001) simulation .867, network .803.
Historical setup and one rollout per option limit transfer to today's bot.
This diagnostic does not isolate hidden-deal sampling from the imagined
bidders or card policies. No strategy change or strength gate.


## Measured: bid forecasts with fixed deals and bids, varied card policy (19 November 2026)

Diagnostic `ohhell.learn.bid_play_audit`. Baseline is the current game's
native cheap card rollout and current shortlist. Two replacements use
the current game hybrid card policy for only the bidder or all seats on
non-random moves. The sampled kind-specific random-card probabilities
are held fixed. All three use the same hidden hands, opponent kinds and
locked bids (including dealer correction for each candidate bid). Earlier
public bids are generated by the standard duel field (previous hybrid x3,
skilled, casual). Bowers, standard weighted 3-8-player table draw and
down/blind/up round weights. Blind and positions with only one shortlisted
bid are skipped. No live wall-clock allowance: 128 deals per card and
endgame choice in the hybrid branches. The live-reference branch uses
the all-seat hybrid on 32 independent deals and fresh card seeds drawn
from the same observation, sampler and imagined-bid process. Each
forecast uses the other 32 deals. Recommended bids use the same network
blend as the current bot. Raw probability metrics assess the rollouts.

320 drawn positions, seed 947341000, 12 workers, 298 scored, 70 with 8-10
cards. Source dee17e8e0dfe..., bid.npz a84e9315..., play.npz 8cb5a064...
(full fingerprints in manifest). Smoke: 8 draws from seed 947340000,
4 worlds per split, 32 card deals, 2 workers, 7 scored.

Exact-bid probability bias (predicted minus reference), cheap / live
bidder only / live all: -4.72 / +1.33 / -0.21 percentage points overall,
and -10.21 / +2.95 / -0.31 on 8-10-card hands. All-live vs cheap Brier
improvement 0.01066 +/- 0.00333 overall, 0.03163 +/- 0.01104 on long
hands (paired approximate 95% margins across positions). Mean trick
bias of the cheap rollout was only -0.012 tricks overall, so much of
this discrepancy is distribution shape. All-live changes 52 of 298
blended bid recommendations, bidder-only changes 43. Held-out modeled
bid-choice score, weighted to the duel's game length: +2.71 +/- 5.62
points/game for all-live and +0.86 +/- 5.49 for bidder-only vs cheap.
Both include zero. No timed or realised strength gate. The reference
is a simulated hybrid population sharing the same sampled worlds model,
not independent human play. This isolates the effect of changing card
play in that setup, but leaves sampling and imagined bids unvalidated.

Saved runs/bid_play/audit_947341000.{meta,jsonl,json} and smoke_947340000.*
in the same directory. Seven related bid-floor and value-bid tests pass.
No change to the game bot.


## Worse in a short duel: live hybrid card play inside bid rollouts (19 November 2026)

`ohhell.learn.bid_play_duel`, fixed 800 pairs, seed 947351000, 8 workers.
Same freshly dealt cards and opponents within each pair. Bowers,
dealer restriction, standard weighted 3-8-player down/blind/up round
draw, scores weighted by number of rounds per game. Field: previous
hybrid x3 / skilled / casual. Both sides use current game hybrid
`hybrid:field=mix,inference=1,shortlist=13,touching=1,bid_sigma=0.76,bid_blend=0.25`,
250 ms wall-clock per bid, and the same 1,024-deal budgets and RNG stream
for actual card play. All identical bids led to identical card sequences.
Candidate replaces the cheap card rollout with the audited all-seat
hybrid, 128 deals per simulated card/endgame choice, on non-random moves
with the sampled kind's original noise. Uses the audit's 32-deal blocks
and locked imagined bids, with existing shortlist and blend. Incomplete
sampled worlds are discarded so each alternative gets the same complete
worlds. If no world finishes before the deadline, the bid network
supplies the choice. Fixed-size short screen with a plain paired 95%
interval, no confirmation. No promotion.

**-46.63 +/- 18.67 points per game**. 221 bids changed. Of 760 ordinary
bids, median completed worlds 2 candidate vs 21,946.5 baseline. Mean bid
times 245.85 / 251.27 ms. Network fallback on 66 positions. Median one
candidate world on 8-10-card hands (205 pairs). Even among rows with
at least one candidate world, the edge was -52.40 points/game
(descriptive, post-selection). This tests the cost of using live search
directly at equal time. The fixed-world audit had 32 worlds per forecast
and found better calibration; equal-time sampling collapsed here.

Source abc7744e74625299..., models bid a84e9315... / play 8cb5a064...
(full fingerprints in manifest). Saved
`runs/bid_play/duel_947351000.{meta,jsonl,json}`. Smoke 12 rounds at
100 ms (seed 947350000, 3 workers), completed. Fourteen related tests passed.


## Measured: cheap live-search bid calibration and compact card rollout (19 November 2026)

Opt-in hybrid setting `bid_calibration_file=ohhell/models/bid_calibration_live.json`.
Four fitted log-odds coefficients (intercept, scaled hand size, zero bid,
and bid/hand-size) adjust the exact-bid probability of the native cheap
rollout before the existing network blend. Other trick counts retain
their relative proportions. Fitted on the 298 paired positions from
seed 947341000 (saved live forecasts versus cheap odds). Same shortlist,
cheap draws, opponent models and actual card play. Live reference
uses 128 deals for its card/endgame choices. Correction costs about
17 microseconds for three rows in a local microbenchmark; equal seed
and fixed budget verified identical sample count and correct normalization.
Model bfcfca4a63ea3cc5..., bid a84e9315..., play 8cb5a064...; hashes
recorded by tournament/duel manifests. Source ce5199ebe6fae895....

Fresh validation: 480 drawn positions, 447 scored (121 with 8-10
cards), seed 947361000, 12 workers. Standard weighted 3-8-player
down/blind/up, bowers, dealer restriction and duel field earlier bidders.
Raw cheap forecast 4,096 worlds per bid, live reference 32 independently
sampled worlds with the audit's 128 card/endgame budget. No wall-clock
limit. Average exact-bid bias -4.65 -> -0.03 percentage points overall
and -9.45 -> -0.62 for 8-10 cards. Brier change -0.00438 +/- 0.00110
overall, modeled bid-choice edge **-0.74 +/- 2.96 points/game**, with
32 recommendations changed. Earlier smoke: 8 draws (7 scored), seed
947360000, 4 live worlds, 2 workers. Files
runs/bid_play/calibration_947361000.{jsonl,json,meta} and calibration_smoke_947360000.*.

Separate historical validation: saved outcomes for 24,511 v2 lobby
positions at 10 ms from runs/value_bid/confirm_t10.jsonl (25,000 drawn
from seed 7100000, historical v2 settings, 18 workers). Postprocessed
both cheap forecast variants with the same 25% saved-network blend.
Brier change -0.00255 +/- 0.00042, recorded bid-choice edge
**-3.01 +/- 1.34 points/game**. Historical forecast/outcome policy
differs from today's default. runs/bid_play/calibration_historical_holdout.json.
Better absolute probability calibration did not imply better bid ranking.
No duel or promotion. 79 relevant tests passed.

Second cheap alternative screened without wiring a new policy into the
game: existing compact `policy=learned_all` checkpoint ca5a1ab4...,
trained on old v2 teacher, bid simulation only. Same 298 positions and
saved live reference as the fixed-deal audit (seed 947341000), current
game hybrid as baseline, 50 ms real bid allowance each, 8 workers,
standard rules/field as in the audit. Median deals 108 learned vs 5,061
cheap (mean 50.77 vs 50.25 ms). Brier .17119 vs .16082; modeled bid-choice
edge **-15.93 +/- 7.23 points/game**. Files
runs/bid_play/compact_rollout_probe.{py,json}. This rules out reusing
this checkpoint unchanged for this purpose. No duel or promotion.


## Screened: paired bid-score gap correction (19 November 2026)

Opt-in `bid_rank_file=ohhell/models/bid_rank_paired.json` on the current
game hybrid (field=mix, inference=1, shortlist=13, touching=1, bid_sigma=.76,
bid_blend=.25). Ranker acts only on 8-10 card hands and among bids within
6 expected points of the current best. Fits a regularized linear correction
to points gaps rather than aggregate exact-bid probabilities. Eleven
features from the hybrid trick odds, bid, hand size, table size and bidding
position. Simulations, live card play and the existing probabilities are
unaltered. Same RNG and 4,096 sampled draws verified with opt-in and base.

Training: 3,200 round seeds from 947370000, 12 workers, 526 eligible
positions and 620 paired bid comparisons. For each position, 4,096 cheap
worlds supply baseline features. On 32 other shared worlds, locked opponent
bids and noise, score each contender with cheap rollout and live hybrid
card play (128 worlds per simulated card/endgame choice). Fit the difference
between their blended score gaps with greater weight on nearly tied bids.
Bowers, dealer restriction, standard weighted 3-8-player down/blind/up
and duel field. Fixed deal counts, no wall-clock budget. Files
runs/bid_rank/train_947370000.{jsonl,meta}; source 3ed033746dc8591f...;
model 49a043f9c8a7623f..., bid a84e9315..., play 8cb5a064....

Frozen model evaluation, no training on either file:
- Current live-search reference, seed 947361000, 447 positions including
  121 with 8-10 cards. 15 changes, all higher. -2.46 points per changed
  hand. Overall edge **-1.73 +/- 2.44 points/game**. On long hands only,
  game-length-scaled conditional edge -6.40 +/- 8.99.
- Recorded historical v2 lobby outcomes, seeds from 7100000 as in historical
  calibration validation, 24,511 positions including 6,091 long hands.
  873 changes, all higher; -2.55 points per changed hand. Overall edge
  **-1.87 +/- 1.11 points/game** with the same 25% saved network blend on
  both sides. Different lobby and bot from the current setup.
- Saved runs/bid_rank/{current_validation,historical_validation}.json.
  75 relevant tests passed, including the new zero-weight invariant check.

The fitted ranker inherited the upward shift despite changing the training
objective to paired point gaps. Does not justify a duel or promotion.
A future action-focused fit needs better labels for ranking close bids,
especially realistic continuation outcomes and enough repeated deals per
comparison to resolve small gaps.
