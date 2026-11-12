# Bot strategy audit — 10 November 2026

The default remains **v1 Hybrid**, `hybrid:field=previous,time_ms=2`.
It approximates several requested behaviours, but does not play perfectly.
The prepared candidate targets planning card play.

## Assessment of v1

| Expectation | Assessment | Implementation and limitation |
| --- | --- | --- |
| Statistically perfect first bid | Approximate only | `SimulationBot.analyse_bids` samples hidden deals and finishes them with rule-based opponents. `HybridBot.analyse_bids` blends those trick distributions 50/50 with the trained bid network, then maximises expected **points**, rather than rounding expected tricks. A heuristic shortlist can exclude legal bids. Neither probabilities nor continuation policies are proven optimal. |
| Infer our bid assuming earlier opponents bid perfectly | Uses earlier bids, without that assumption | `Belief.log_weight` weights imagined hands by how close each opponent's bid is to a heuristic **mean trick count**, with error tolerance. This is not an inverse model of an optimal expected-score bidding policy. The bid network sees position and aggregate earlier bids, rather than every opponent's bid individually. Future bidders use heuristic bids in rollouts. |
| Scale aggression with table overcall/undercall | Partly represented | `play_matrix` includes our remaining need, others' remaining demand and players behind who want tricks. Simulated rewards penalise missing our bid. The rollout policy is mostly binary: want tricks while `need > 0`, try to lose otherwise. Its optional `react` adds only coarse table reactions; v1 leaves it off. There is no explicit smooth aggression controller. |
| Stay conservative early in a large undercall; consider a bailout | Possible through simulation, without a guarantee | Final-score simulations can expose chains of unwanted wins and the value of keeping a losing card. The continuation policy has no explicit bailout analysis, and v1 can exclude useful actions through its top-four shortlist. Playing low early can also retain dangerous winners until too late. A bailout depends on remaining opposing cards and likely leads, not just its rank. |
| Fish out cards/information and adapt | Updates information; limited proactive planning | Observations retain public history, played cards and per-seat void suits. Sampled deals respect those facts; inference also weights bids and observed plays. Each real turn uses the updated observation. Ordinary root rollouts do not optimise later choices as a policy branching on new responses, so useful probes can be undervalued. |

Blind 0 already uses a separate combinatorial single-trick win probability
from public information; all hands remain hidden. That restricted calculation
being exact does not make ordinary bidding or play exact. Similarly, the
endgame solver is exact **within a sampled complete deal**, not a proof of
optimal play with hidden cards. The current objective is expected round points;
observations do not include cumulative game scores for changing risk when ahead
or behind across the full game.

## Candidate: visible-history planning

`planning:field=previous` replaces earlier-round card search with a tree shared
across sampled hidden deals. The working tree already contained Python and
native tree implementations; this work adds validation and restores the
existing hybrid endgame path.

- Branches are publicly played cards. Hidden hands and opponent types never
  select a branch. Identical visible histories share our future decisions
  even when simulated hidden hands differ.
- Every legal root card gets a chance. The unchanged play network only orders
  first exploration; it contributes no artificial score samples.
- Opponents follow the existing population policy. Our later choices improve
  as histories are revisited. Final round points reward a probe, early
  concession or escape card through its projected consequences.
- The most visited root action wins; mean reward breaks ties. UCB exploration
  defaults to `exploration=30`; `tree_batch=8` controls sampling.
- Bidding, checkpoints, beliefs, rollout policy, blind rounds and exact-endgame
  handling retain v1 behaviour. Untimed earlier-round search uses
  `worlds * legal_cards` simulations. Timed search shares one soft deadline;
  in-progress sampling or simulation can overrun it.

This tests a replacement search algorithm, including action allocation,
model-prior treatment and later decision search. It does not isolate each
internal choice, or combine the separate adaptive-card, evidence-bid or
candidate-checkpoint experiments.

At 2 ms, a large tree may not revisit enough histories to discover a useful
probe. Opponent policies and hidden-deal sampling can be wrong. This does not
implement rational-opponent bidding inference, guaranteed bailout behaviour,
or a multiplayer equilibrium. Assuming optimal opponents needs a defined
opponent policy and a separate experiment; tournament opponents include
players who make mistakes.

## Verification and tournament

`tests/test_planning.py` includes a controlled hidden-information game where
probing is best only if the response reveals useful information. Tests also
cover shared branches across hidden states, every legal root card, completed
simulation counts, deadlines, tree limits, unchanged bidding/endgames,
checkpoint fingerprints, and complete down/blind/up games with both backends.
`tests/test_planning_deals.py` adds a concrete card-game continuation example
and checks native tree isolation between searches. Full suite: **327 passed**.
These checks establish implementation properties, not stronger Oh Hell play.

The eight-seed smoke check is inconclusive: **+2.8 +/- 7.8 points per game**.
See `BOT_EXPERIMENTS.md` for settings and saved evidence.

Run the long comparison yourself:

```powershell
.venv\Scripts\python.exe -m ohhell.tournament "planning:field=previous" --label "Planning information v1" --workers 6 --port 8770
```

Dashboard: <http://localhost:8770/>. Both sides receive 2 ms per turn. The
tournament uses down/blind/up games, rotates seats and requires unseen-deal
confirmation after a successful main gate. Use a fresh label after code or
settings change; run comparisons sequentially. No checkpoint is overwritten
and no default is promoted automatically.
