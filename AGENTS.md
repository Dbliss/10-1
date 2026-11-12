# Repository guidance

Before changing bot strategy, training models, or proposing a tournament, read
[BOT_EXPERIMENTS.md](BOT_EXPERIMENTS.md). It records attempted changes, exact
candidate settings, results, and limitations. Avoid repeating an unchanged
candidate already listed as inconclusive or unconfirmed unless the user asks
or there is a concrete new hypothesis. An inconclusive result is not proof
that the idea can never help, but it is not grounds for promotion.

Keep the current default bot unless a candidate passes the tournament's main
gate and confirmation on unseen deals. Update the experiment log after each
completed run, including unsuccessful runs. Record the rules, baseline, seed,
time allowance, model fingerprints when available, and any missing evidence.
Do not combine unproven changes and describe the result as a single-change test.

The user wants to run long bot comparisons themselves: prepare the candidate,
verify it, and provide the command and dashboard URL. Do not control the user's
desktop or browser. Saved results can be read directly from `runs/gates/`.

The actual game runs from the starting hand down to 1, then blind 0, then from
1 back to the starting hand. Blind 0 deals one card to each player; all hands
are hidden until played. Scores accumulate across the entire sequence.

Trumps rank with bowers: the Jack of trump, then the same-colour Jack (a
trump for every purpose, including following suit), then Ace down to two.
Use the trump-aware tables in `ohhell/cards.py` (`EFFECTIVE_SUIT`, `POWER`,
`SUIT_CARDS`, `OUTRANKS`, `top`/`bottom`, `beats`) for gameplay, never the
printed `SUIT_OF`/`SUIT_MASK`/`ABOVE` or raw card-integer order.
