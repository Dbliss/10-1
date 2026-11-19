# Public Oh Hell-family bot league

This event puts five players at the same table on this repository's rules.
It is a reference comparison of **adapted implementations**, with no automatic
promotion of v2. SkatZero and BotEuchre are excluded at the user's request.

**13 November crash fix:** the first `Public_bots_v1` run stopped after ten
sets because of two native MCTS pointer defects. Upward pruning could remove
the wrong child slot, corrupting the heap; zero-valued child selection could
retain a stale pointer. Both are fixed and the captured position is covered
by a subprocess regression. Keep the old files, but start the fixed build
with a **new label**; its source and decisions differ, so it cannot resume
or merge with the old run. The examples below use `Public_bots_v2`.

| Entry | Actual implementation |
| --- | --- |
| v2 Self-play bowers | Current promoted root models and ordinary hybrid search. |
| Campbell OI neural | NumPy port of [AiStrategyModuleOI](https://github.com/campbellsoup37/OhHell), loading its original OVL and IVL text checkpoints for the table size. |
| DiminishingBridge MC | Compiled [upstream C++ Monte Carlo agent](https://github.com/thijssnelleman/DiminishingBridge), including its hidden-world sampling and random continuations. |
| DiminishingBridge MCTS | The same project's visible-history Monte Carlo tree-search agent. |
| DRL-POMDP heuristic | Executes [the upstream RuleBasedPlayer](https://github.com/jonas-schumacher/drl-pomdp) Python class. The project ships no trained DQN checkpoint, so this entry is explicitly its heuristic player. |

## Run

From the repository root:

```powershell
.venv\Scripts\python.exe -m ohhell.external_league --prepare
.venv\Scripts\python.exe -m ohhell.external_league --sets 100 --workers 4 --label Public_bots_v2
```

Dashboard: <http://localhost:8781/>. If occupied, the runner prints the next
available port. It never opens or controls your browser. The dashboard remains
available after completion until Ctrl+C; add `--exit-when-done` to exit.

The default is **100 deal sets / 500 complete games**, opening hand 10, with
all five entries at every table. Set `--sets 500` for 2,500 games. Search
players receive a nominal soft `--time-ms 2`; the neural and rule players
return directly. Actual mean and longest times are shown, so overruns and
different inference costs are visible. This is not a strict equal-CPU proof.

On first preparation the program downloads 26 files from pinned revisions,
including original neural weights, into gitignored `runs/external_bots/`.
No Java or PyTorch installation is needed. NumPy and a working `g++` compiler
are required. The local C++ DLL links its runtime statically. The normal
native Oh Hell core is built if needed.

For a short complete-game check without a strength comparison:

```powershell
.venv\Scripts\python.exe -m ohhell.external_league --check-only
```

That checks five rotated games from hand 3 through blind 0 and back up.

Results are saved as:

```text
runs/external_leagues/Public_bots_v2.meta
runs/external_leagues/Public_bots_v2.jsonl
runs/external_leagues/Public_bots_v2.json
```

The ledger contains completed deal sets, all game/round scores, clocks, legal
bid adjustments and feature saturation counts. The summary updates after
each set. Interrupted runs retain completed sets. Resume with the original
settings and label:

```powershell
.venv\Scripts\python.exe -m ohhell.external_league --sets 100 --workers 4 --label Public_bots_v2 --resume
```

Resume refuses changed source, rules, roster, root/upstream weights or search
settings. A new label is required after any implementation change. It never
overwrites an existing run without `--resume`. Native failures and illegal
cards stop the event; a different bot is never substituted.

Workers now record native fault traces and their current public seed,
rotation and round under `runs/external_leagues/LABEL.diagnostics/`. A native
crash names that context instead of exposing only `BrokenProcessPool`.
These logs contain no actual opponent hands. Completed sets stay in the
ledger even when an in-flight set fails.

Optional subsets use stable entry IDs, for example:

```powershell
.venv\Scripts\python.exe -m ohhell.external_league --roster v2,campbell,diminishing-mc --hand 10 --sets 100 --label Three_players_v1
```

## Common rules and information

- Standard 52-card deck. Right bower, left bower, then Ace down to two;
  left bower follows trump for every purpose.
- Starting hand down to 1, blind 0, then 1 back to the start. Blind 0 deals
  one hidden card to every seat; all players use the shared blind policy.
- Repository scoring: +10 per trick when exactly made, +10 for made zero,
  otherwise -10 per trick off. Dealer cannot balance all bids to the hand size.
- Scores accumulate over the entire sequence. DiminishingBridge receives
  public cumulative scores through completed-round observations.
- Adapters receive only their own hand and public bids, history, voids,
  trump and scores. They never receive actual opponent hands. Each upstream
  simulator generates its own hidden-world guesses.

For each seed, player order is shuffled, then the identical complete game is
replayed once per seat rotation. Every entry gets every hand/seat. Shuffling
between seeds prevents one fixed circular ordering of opponents throughout
the event. The seed defaults to 202611130 and can be set with `--seed`.

Standings show mean points and win share. Ties split win credit. The paired
score matrix compares players at the same mixed tables; these are not isolated
head-to-head duels. Ordinary descriptive 95% intervals use seed-level averages
over rotations. They do not treat repeated hands as independent games, account
for optional stopping, or satisfy the existing promotion gate.

## Exact adaptations and limits

**Campbell:** retained original per-table OVL/IVL checkpoints and probability
convolution. Bidding uses our score function and legal bids. Play retains the
upstream objective of maximizing probability of making the bid. Suit and
relative-rank features are made bower-aware; counts saturate at the original
13-category model width, with saturation telemetry. The source weights were
not retrained for bowers or our scoring. Stable card-order tie-breaking
replaces Java Hashtable ordering. This is a port, not an original Java runtime.

**DiminishingBridge:** preserves the original MC/MCTS search and sampling.
A 4-by-15 grid represents effective suits/bower powers, with nonexistent
rank slots marked absent. Capacity increases allow five players. Simulated
round scoring and score bounds use the shared scoring; bidding retains the
upstream rounded trick estimate. Both searches use soft elapsed deadlines.
MC root values use completed sample means, and bid estimates divide by
completed simulations instead of the original fixed budget after a cut-off.
At least one successful sample per MC root action is attempted, so soft
deadline overruns remain possible. Failed MC playouts do not count as outcomes.

The pinned source also needed stack-root deletion and use-after-free fixes,
root-move lookup correction, five-player trick-position handling, wider score
arithmetic, and removal of interactive failure prompts. The patch recipe is
in `ohhell/external_setup.py`; original source stays intact beside the
generated patched build. Its future-score heuristic still assumes descending
rounds, so it is not perfectly tailored to our down/blind/up format.

**DRL-POMDP:** upstream code executes unchanged. Its environment shim provides
the repository's bower-aware card comparison table and only the acting hand.
The dealer wrapper chooses the nearest legal bid, preferring the lower bid
on equal distance; adjustments are counted. The published DQN architecture
is not represented by random weights or a substitute policy.

The repository revisions are pinned:

| Project | Commit |
| --- | --- |
| Campbell OhHell | `2f4d3dd3600d1df1cf439258b6150aee3f50ce7d` |
| DiminishingBridge | `305af70fe0c5341aeb70d5e9d2411bb57c4e7c67` |
| DRL-POMDP | `1acb47b18959176bc654f10b9710209307ad8098` |

Upstream originals/checkpoints are local research downloads rather than
tracked vendored code. No licence file was found in these pinned repository
roots; do not interpret this setup as permission to redistribute them.
Every run fingerprints downloaded files, checkpoints, generated DLL and local
adapter/core sources. Adaptation quality and rule changes can affect results;
the event cannot establish which original project is strongest at its own game.
