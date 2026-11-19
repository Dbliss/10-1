import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import { api } from "../api";
import { cardImage, cardName, isRed, sortHand, SUIT_GLYPH } from "../cards";
import type { Card, GameReview as Review, Grade, ReviewMove, ReviewSummary } from "../types";

export const GRADES: { grade: Grade; label: string; mark: string }[] = [
  { grade: "best", label: "Best", mark: "★" },
  { grade: "excellent", label: "Excellent", mark: "!" },
  { grade: "good", label: "Good", mark: "✓" },
  { grade: "inaccuracy", label: "Inaccuracy", mark: "?!" },
  { grade: "mistake", label: "Mistake", mark: "?" },
  { grade: "blunder", label: "Blunder", mark: "??" },
];
const GRADE_INFO = Object.fromEntries(GRADES.map((g) => [g.grade, g])) as Record<
  Grade,
  (typeof GRADES)[number]
>;

const signed = (value: number) => (value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1));
const shortCard = (card: Card) =>
  `${card.rank === "10" ? "10" : card.rank[0]}${SUIT_GLYPH[card.suit]}`;

/** Polls the server's review of the user's decisions until it is complete. */
export function useReview(active: boolean) {
  const [review, setReview] = useState<Review | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!active) {
      setReview(null);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await api.review();
        if (cancelled) return;
        setReview(next);
        setError(null);
        if (next.complete) return;
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "The review couldn't load.");
      }
      timer = setTimeout(poll, 1000);
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [active]);

  return { review, error };
}

function GradeBadge({ grade, small }: { grade: Grade; small?: boolean }) {
  const info = GRADE_INFO[grade];
  return (
    <span className={`grade-badge grade-${grade} ${small ? "grade-badge-small" : ""}`} title={info.label}>
      {info.mark}
    </span>
  );
}

function AccuracyTile({ label, summary }: { label: string; summary: ReviewSummary }) {
  const value = summary.accuracy;
  const ring = value === null ? 0 : value;
  return (
    <div className="acc-tile">
      <div className="acc-ring" style={{ "--acc": ring } as React.CSSProperties}>
        <strong>{value === null ? "–" : value.toFixed(1)}</strong>
      </div>
      <div className="acc-text">
        <span className="label">{label}</span>
        <span className="acc-detail">
          {summary.decisions} {summary.decisions === 1 ? "choice" : "choices"} ·{" "}
          {summary.points_lost.toFixed(1)} pts lost
        </span>
      </div>
    </div>
  );
}

/** One line on the game-over dialog: accuracy, or progress while it is worked out. */
export function ReviewTeaser({ review }: { review: Review | null }) {
  if (!review) return <p className="review-teaser">Loading your game review…</p>;
  if (!review.complete)
    return (
      <p className="review-teaser">
        Analysing your choices… {review.reviewed} of {review.total}
      </p>
    );
  if (!review.total) return null;
  return (
    <p className="review-teaser">
      Accuracy <strong>{review.overall.accuracy?.toFixed(1)}</strong> · Best play would average{" "}
      <strong>{Math.round(review.expected_best_score)}</strong>
    </p>
  );
}

function moveTitle(move: ReviewMove) {
  if (move.kind === "bid") return `Bid ${move.choice}`;
  const card = move.hand.find((c) => c.id === move.choice);
  return `Trick ${move.trick_number}: ${card ? shortCard(card) : move.choice}`;
}

function roundLabel(move: ReviewMove) {
  return move.blind ? "Blind" : `${move.hand_size} ${move.hand_size === 1 ? "card" : "cards"}`;
}

function MiniCard({ card, tone }: { card: Card; tone?: "chosen" | "best" | "both" }) {
  return (
    <span className={`mini-card ${tone ? `mini-card-${tone}` : ""}`}>
      <img src={cardImage(card)} alt={cardName(card)} />
      {tone && (
        <span className="mini-card-tag">{tone === "both" ? "You · Best" : tone === "best" ? "Best" : "You"}</span>
      )}
    </span>
  );
}

function MoveDetail({ move, players }: { move: ReviewMove; players: string[] }) {
  const info = GRADE_INFO[move.grade];
  const sameChoice = String(move.choice) === String(move.best);
  const describe = (choice: string | number) => {
    if (move.kind === "bid") return `bid ${choice}`;
    const card = move.hand.find((c) => c.id === choice);
    return card ? cardName(card) : String(choice);
  };
  const hand = sortHand(move.hand, move.trump.suit);
  const n = players.length;
  // Bidding starts left of the dealer; so does the first trick.
  const order = Array.from({ length: n }, (_, i) => (move.dealer + 1 + i) % n);

  return (
    <motion.div
      key={move.index}
      className="move-detail"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
    >
      <div className={`move-verdict grade-text-${move.grade}`}>
        <GradeBadge grade={move.grade} />
        <div>
          <strong>{info.label}</strong>
          <span>
            {sameChoice
              ? `You ${move.kind === "bid" ? "bid" : "played"} the advisor's top choice.`
              : `You chose ${describe(move.choice)}; best was ${describe(move.best)}${
                  move.loss > 0.05 ? ` (${signed(-move.loss)} expected points)` : ""
                }.`}
          </span>
        </div>
      </div>

      <div className="move-context">
        <div className="move-trump">
          <img src={cardImage(move.trump)} alt={`Trump: ${cardName(move.trump)}`} />
          <span className="label">Trump</span>
          <strong className={isRed(move.trump.suit) ? "suit-red" : "suit-light"}>
            {SUIT_GLYPH[move.trump.suit]} {move.trump.suit}
          </strong>
        </div>
        <table className="move-seats">
          <thead>
            <tr>
              <th scope="col">Player</th>
              <th scope="col">Bid</th>
              <th scope="col">Won</th>
            </tr>
          </thead>
          <tbody>
            {order.map((seat) => (
              <tr key={seat} className={seat === 0 ? "row-you" : ""}>
                <th scope="row">
                  {players[seat]}
                  {seat === move.dealer && <span className="dealer-chip dealer-chip-inline">D</span>}
                </th>
                <td>{move.bids[seat] ?? "–"}</td>
                <td>{move.kind === "bid" ? "–" : move.tricks_won[seat]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {move.kind === "card" && (
        <div className="move-block">
          <span className="label">
            Trick {move.trick_number} of {move.hand_size}
            {move.trick.length === 0 ? " · you lead" : ""}
          </span>
          {move.trick.length > 0 && (
            <div className="move-trick">
              {move.trick.map(({ seat, card }) => (
                <figure key={card.id}>
                  <MiniCard card={card} />
                  <figcaption>{players[seat]}</figcaption>
                </figure>
              ))}
            </div>
          )}
        </div>
      )}

      {hand.length > 0 && (
        <div className="move-block">
          <span className="label">Your hand</span>
          <div className="move-hand">
            {hand.map((card) => {
              const chosen = move.kind === "card" && card.id === move.choice;
              const best = move.kind === "card" && card.id === move.best;
              return (
                <MiniCard
                  key={card.id}
                  card={card}
                  tone={chosen && best ? "both" : chosen ? "chosen" : best ? "best" : undefined}
                />
              );
            })}
          </div>
        </div>
      )}
      {move.blind && <p className="move-note">Blind round: you bid before seeing your card.</p>}

      <div className="move-block">
        <span className="label">Advisor's estimate of each option</span>
        <div className="move-options">
          {move.kind === "bid"
            ? move.options.map((option) => (
                <div
                  key={option.bid}
                  className={`move-option ${option.bid === move.best ? "move-option-best" : ""} ${
                    option.bid === move.choice ? "move-option-you" : ""
                  }`}
                >
                  <span>Bid {option.bid}</span>
                  <span>{Math.round(option.make_chance * 100)}% to make</span>
                  <span>{signed(option.expected_points)}</span>
                  <span className="move-option-tag">{option.bid === move.choice ? "You" : ""}</span>
                </div>
              ))
            : move.options.map((option) => (
                <div
                  key={option.card.id}
                  className={`move-option ${option.card.id === move.best ? "move-option-best" : ""} ${
                    option.card.id === move.choice ? "move-option-you" : ""
                  }`}
                >
                  <span>{cardName(option.card)}</span>
                  <span />
                  <span>{signed(option.expected_points)}</span>
                  <span className="move-option-tag">{option.card.id === move.choice ? "You" : ""}</span>
                </div>
              ))}
        </div>
        <p className="move-note">Average round points after each choice, from sampled deals.</p>
      </div>
    </motion.div>
  );
}

interface GameReviewProps {
  review: Review | null;
  error: string | null;
  onClose: () => void;
}

export function GameReviewScreen({ review, error, onClose }: GameReviewProps) {
  const [filter, setFilter] = useState<Grade | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const moves = review?.moves ?? [];
  const shown = useMemo(
    () => (filter ? moves.filter((m) => m.grade === filter) : moves),
    [moves, filter],
  );

  // Start on the costliest decision once the review is in.
  useEffect(() => {
    if (selected !== null || !review?.complete || !moves.length) return;
    const worst = moves.reduce((a, b) => (b.loss > a.loss ? b : a));
    setSelected(worst.index);
  }, [review?.complete, moves, selected]);

  const current = moves.find((m) => m.index === selected) ?? null;
  const position = current ? shown.findIndex((m) => m.index === current.index) : -1;
  const step = (delta: number) => {
    if (!shown.length) return;
    const next = position === -1 ? 0 : Math.min(shown.length - 1, Math.max(0, position + delta));
    setSelected(shown[next].index);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault();
        step(1);
      }
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        step(-1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Keep the selected row in view by scrolling the list alone, not the page.
  useEffect(() => {
    const list = listRef.current;
    const row = list?.querySelector<HTMLElement>(`[data-move="${selected}"]`);
    if (!list || !row) return;
    const top = row.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop;
    if (top < list.scrollTop + 28) list.scrollTop = Math.max(0, top - 28);
    else if (top + row.offsetHeight > list.scrollTop + list.clientHeight)
      list.scrollTop = top + row.offsetHeight - list.clientHeight + 8;
  }, [selected, filter]);

  const chooseFilter = (grade: Grade) => {
    const next = filter === grade ? null : grade;
    setFilter(next);
    const first = next ? moves.find((m) => m.grade === next) : undefined;
    if (first) setSelected(first.index);
  };

  let lastRound = -1;
  const gained = review ? review.expected_best_score - review.score : 0;

  return (
    <motion.div
      className="review-screen"
      role="dialog"
      aria-modal="true"
      aria-label="Game review"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <header className="review-top">
        <h2>Game review</h2>
        <button type="button" className="btn btn-quiet" onClick={onClose}>
          Back
        </button>
      </header>

      {error && <p className="advice-error">{error}</p>}
      {!review ? (
        <p className="review-wait">Loading your game review…</p>
      ) : !review.total ? (
        <p className="review-wait">You had no real choices to review this game.</p>
      ) : (
        <div className="review-body">
          {!review.complete && (
            <div className="review-progress" role="status">
              <span>
                Analysing your choices… {review.reviewed} of {review.total}
              </span>
              <div className="review-bar">
                <div style={{ width: `${(100 * review.reviewed) / review.total}%` }} />
              </div>
            </div>
          )}

          <section className="review-summary">
            <div className="acc-tiles">
              <AccuracyTile label="Accuracy" summary={review.overall} />
              <AccuracyTile label="Bidding" summary={review.bidding} />
              <AccuracyTile label="Playing" summary={review.playing} />
            </div>
            <div className="review-score">
              <div>
                <span className="label">Your score</span>
                <strong>{review.score}</strong>
              </div>
              <div>
                <span className="label">With the advisor's best choices</span>
                <strong>
                  ≈ {review.expected_best_score.toFixed(0)}
                  {gained > 0.5 && <small className="gain"> +{gained.toFixed(0)}</small>}
                </strong>
              </div>
              <p className="move-note">
                Your score plus the expected points each choice gave up. An estimate: the
                cards could still have fallen differently.
              </p>
            </div>
            <div className="grade-chips" role="group" aria-label="Filter by grade">
              {GRADES.map(({ grade, label }) => (
                <button
                  key={grade}
                  type="button"
                  className={`grade-chip grade-chip-${grade}`}
                  aria-pressed={filter === grade}
                  disabled={!review.overall.grades[grade]}
                  onClick={() => chooseFilter(grade)}
                >
                  <GradeBadge grade={grade} small />
                  <span>{label}</span>
                  <strong>{review.overall.grades[grade]}</strong>
                  <span className="grade-chip-split" title="Bids · plays">
                    {review.bidding.grades[grade]}·{review.playing.grades[grade]}
                  </span>
                </button>
              ))}
            </div>
          </section>

          <div className="review-main">
            <div className="move-list" ref={listRef}>
              {shown.map((move) => {
                const header = move.round !== lastRound;
                lastRound = move.round;
                return (
                  <div key={move.index}>
                    {header && (
                      <div className="move-round">
                        Round {move.round} · {roundLabel(move)}{" "}
                        <span className={isRed(move.trump.suit) ? "suit-red" : "suit-light"}>
                          {SUIT_GLYPH[move.trump.suit]}
                        </span>
                      </div>
                    )}
                    <button
                      type="button"
                      data-move={move.index}
                      className={`move-row ${move.index === selected ? "move-row-on" : ""}`}
                      onClick={() => setSelected(move.index)}
                    >
                      <GradeBadge grade={move.grade} small />
                      <span>{moveTitle(move)}</span>
                      <span className="move-row-loss">{move.loss >= 0.05 ? signed(-move.loss) : ""}</span>
                    </button>
                  </div>
                );
              })}
            </div>
            <div className="move-pane">
              {current ? (
                <>
                  <MoveDetail move={current} players={review.players} />
                  <div className="move-nav">
                    <button type="button" className="btn btn-quiet" onClick={() => step(-1)} disabled={position <= 0}>
                      ← Previous
                    </button>
                    <button
                      type="button"
                      className="btn btn-quiet"
                      onClick={() => step(1)}
                      disabled={position === -1 || position >= shown.length - 1}
                    >
                      Next →
                    </button>
                  </div>
                </>
              ) : (
                <p className="review-wait">Pick a decision to see it.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}
