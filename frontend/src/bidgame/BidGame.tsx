import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { cardImage, cardName, isRed, preloadCardImages, sortHand, SUIT_GLYPH } from "../cards";
import { fetchSituation } from "./api";
import { BarChart, LineChart, Swatch, type Series } from "./Charts";
import {
  bestOption, complete, load, option, ordinal, pct, plural, save, signed, tone, totals,
} from "./format";
import type { Player, Round, Seat, Situation } from "./types";
import "./bidgame.css";

const ROUNDS = 10;
const COLORS = ["var(--bg-p1)", "var(--bg-p2)"];

type Screen = "setup" | "table" | "results" | "final";
type Mode = "solo" | "duo";

/** Seat 0 (the user) sits at the bottom; turn order runs clockwise from there. */
const seatAngle = (seat: number, seats: number) => Math.PI / 2 + (seat * 2 * Math.PI) / seats;

/** In two-player games the players take turns bidding first. */
const biddingOrder = (players: Player[], hand: number) =>
  players.length === 2 && hand % 2 ? [1, 0] : players.map((_, i) => i);

/* ================================================================ app */

export default function BidGame() {
  const [screen, setScreen] = useState<Screen>("setup");
  const [mode, setMode] = useState<Mode>(() => load("bidGame.mode", "solo"));
  const [names, setNames] = useState<string[]>(() => load("bidGame.names", ["Player 1", "Player 2"]));
  const [players, setPlayers] = useState<Player[]>([]);
  const [rounds, setRounds] = useState<Round[]>([]);
  const [current, setCurrent] = useState(0);
  const [turn, setTurn] = useState(0);
  const [covered, setCovered] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [best, setBest] = useState<number | null>(() => load("bidGame.best", null));
  const [previousBest, setPreviousBest] = useState<number | null>(null);
  const deals = useRef<(Promise<Situation> | undefined)[]>([]);
  const gameId = useRef(0);

  useEffect(preloadCardImages, []);

  /** Each hand is requested once; a failed request may be tried again. */
  const deal = useCallback((index: number) => {
    let request = deals.current[index];
    if (!request) {
      request = fetchSituation();
      deals.current[index] = request;
      const mine = request;
      request.catch(() => {
        if (deals.current[index] === mine) deals.current[index] = undefined;
      });
    }
    return request;
  }, []);

  const startRound = useCallback(
    async (index: number, duo: boolean) => {
      const id = gameId.current;
      setCurrent(index);
      setTurn(0);
      setCovered(duo);
      setError(null);
      setScreen("table");
      try {
        const sit = await deal(index);
        if (id !== gameId.current) return;
        setRounds((prev) => {
          const next = [...prev];
          next[index] = { sit, bids: [] };
          return next;
        });
        if (index + 1 < ROUNDS) deal(index + 1).catch(() => {}); // have the next hand ready
      } catch (err) {
        if (id !== gameId.current) return;
        setError(err instanceof Error ? err.message : "Could not deal a hand.");
      }
    },
    [deal],
  );

  const startGame = () => {
    gameId.current += 1;
    deals.current = [];
    const duo = mode === "duo";
    setPlayers(
      duo
        ? names.map((name, i) => ({ name: name.trim() || `Player ${i + 1}`, color: COLORS[i] }))
        : [{ name: "You", color: COLORS[0] }],
    );
    setRounds([]);
    setPreviousBest(best);
    startRound(0, duo);
  };

  const quit = () => {
    gameId.current += 1;
    setError(null);
    setScreen("setup");
  };

  const order = biddingOrder(players, current);
  const round = rounds[current];

  const placeBid = (bid: number) => {
    const bids = [...round.bids];
    bids[order[turn]] = bid;
    setRounds((prev) => {
      const next = [...prev];
      next[current] = { ...round, bids };
      return next;
    });
    if (turn + 1 < players.length) {
      setTurn(turn + 1);
      setCovered(true);
    } else {
      setScreen("results");
    }
  };

  const nextHand = () => {
    if (current + 1 < ROUNDS) {
      startRound(current + 1, players.length === 2);
      return;
    }
    if (players.length === 1) {
      const total = Math.round(totals(rounds, players)[0] * 10) / 10;
      if (best === null || total > best) {
        setBest(total);
        save("bidGame.best", total);
      }
    }
    setScreen("final");
  };

  const chooseMode = (next: Mode) => {
    setMode(next);
    save("bidGame.mode", next);
  };
  const rename = (index: number, name: string) => {
    const next = names.map((old, i) => (i === index ? name : old));
    setNames(next);
    save("bidGame.names", next);
  };

  let body: ReactNode;
  if (screen === "setup") {
    body = (
      <Setup
        mode={mode}
        names={names}
        best={best}
        onMode={chooseMode}
        onRename={rename}
        onStart={startGame}
      />
    );
  } else if (screen === "final") {
    body = (
      <FinalScreen
        rounds={rounds}
        players={players}
        previousBest={previousBest}
        onPlayAgain={startGame}
        onSettings={quit}
      />
    );
  } else {
    const done = rounds.filter((r) => complete(r, players)).length;
    const topbar = <TopBar current={current} done={done} onQuit={quit} />;
    const sidebar = (
      <aside className="sidebar">
        <BidScoreboard rounds={rounds} players={players} current={current} />
        {players.length === 1 && best !== null && (
          <p className="bg-note bg-best">Personal best: {signed(best)}</p>
        )}
      </aside>
    );
    body =
      screen === "results" && complete(round, players) ? (
        <ResultsScreen
          key={current}
          round={round}
          index={current}
          players={players}
          topbar={topbar}
          sidebar={sidebar}
          onNext={nextHand}
        />
      ) : (
        <TableScreen
          sit={round?.sit}
          index={current}
          player={players[order[turn]]}
          other={players.length === 2 ? players[order[1 - turn]] : undefined}
          previous={turn > 0 ? players[order[turn - 1]] : undefined}
          covered={covered}
          onUncover={() => setCovered(false)}
          onBid={placeBid}
          topbar={topbar}
          sidebar={sidebar}
        />
      );
  }

  return (
    <>
      {body}
      {error && (
        <div className="toast" role="alert">
          <span>{error}</span>
          <button type="button" className="btn btn-quiet" onClick={() => startRound(current, players.length === 2)}>
            Retry
          </button>
        </div>
      )}
    </>
  );
}

/* ================================================================ setup */

interface SetupProps {
  mode: Mode;
  names: string[];
  best: number | null;
  onMode: (mode: Mode) => void;
  onRename: (index: number, name: string) => void;
  onStart: () => void;
}

function Setup({ mode, names, best, onMode, onRename, onStart }: SetupProps) {
  return (
    <main className="setup">
      <div className="setup-fan" aria-hidden="true">
        {["jack_of_hearts", "ace_of_diamonds", "jack_of_diamonds"].map((card, i) => (
          <img key={card} src={`${import.meta.env.BASE_URL}cards/${card}.png`} alt="" style={{ "--i": i - 1 } as React.CSSProperties} />
        ))}
      </div>

      <h1>Bid Game</h1>
      <p className="setup-lede">Ten random hands. Call the tricks, then see how close you got.</p>

      <fieldset className="choice">
        <legend>Players</legend>
        <div className="choice-options">
          {(
            [
              ["solo", "Solo"],
              ["duo", "Two players"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className="choice-option"
              aria-pressed={mode === value}
              onClick={() => onMode(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </fieldset>

      {mode === "duo" && (
        <div className="bg-names">
          {names.map((name, i) => (
            <label key={i} className="bg-name">
              <Swatch color={COLORS[i]} />
              <input
                type="text"
                value={name}
                maxLength={14}
                aria-label={`Player ${i + 1} name`}
                onChange={(event) => onRename(i, event.target.value)}
              />
            </label>
          ))}
        </div>
      )}

      <button type="button" className="btn btn-primary btn-wide" onClick={onStart}>
        Deal the first hand
      </button>
      {mode === "solo" && best !== null && (
        <p className="bg-note bg-center">Personal best: {signed(best)}</p>
      )}

      <p className="setup-rules">
        Each hand is a random deal: 2 to 8 players, 1 to 10 cards, sometimes the blind round, and a
        random seat in the bidding order. The seats before you have already bid. Say how many tricks
        you will take, and the results show the chance of making each bid and its expected points.
        A hand scores the expected points of your bid; the highest total after ten hands wins.
        {mode === "duo" && " Two players bid the same hands on one screen, one looking away while the other bids."}
      </p>
      <p className="setup-rules bg-center">
        <a href={import.meta.env.BASE_URL}>Play the full game</a>
      </p>
    </main>
  );
}

/* ================================================================ shared pieces */

function QuitButton({ onQuit }: { onQuit: () => void }) {
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!confirming) return;
    const timer = setTimeout(() => setConfirming(false), 3000);
    return () => clearTimeout(timer);
  }, [confirming]);

  return (
    <button
      type="button"
      className={`btn btn-quiet ${confirming ? "btn-danger" : ""}`}
      onClick={() => (confirming ? onQuit() : setConfirming(true))}
    >
      {confirming ? "Abandon this game?" : "Quit"}
    </button>
  );
}

function TopBar({ current, done, onQuit }: { current: number; done: number; onQuit: () => void }) {
  return (
    <header className="topbar">
      <span className="brand">Bid Game</span>
      <ol className="pips" aria-label={`Hand ${current + 1} of ${ROUNDS}`}>
        {Array.from({ length: ROUNDS }, (_, i) => {
          const state = i === current && done <= i ? "now" : i < done ? "done" : "next";
          return (
            <li key={i} className={`pip pip-${state}`} aria-hidden="true">
              {i + 1}
            </li>
          );
        })}
      </ol>
      <span className="round-compact">
        Hand {current + 1}/{ROUNDS}
      </span>
      <QuitButton onQuit={onQuit} />
    </header>
  );
}

function BidScoreboard({ rounds, players, current }: { rounds: Round[]; players: Player[]; current: number }) {
  const sums = totals(rounds, players);
  const perfect = rounds.reduce(
    (sum, round) => (complete(round, players) ? sum + bestOption(round.sit).expected_points : sum),
    0,
  );
  return (
    <section className="scoreboard" aria-label="Scoreboard">
      <h2>Scores</h2>
      <div className="scoreboard-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">
                <span className="visually-hidden">Hand</span>
              </th>
              {players.map((player) => (
                <th key={player.name} scope="col" className={players.length === 1 ? "col-you" : ""}>
                  <span className="bg-head">
                    {players.length > 1 && <Swatch color={player.color} />}
                    {player.name}
                  </span>
                </th>
              ))}
              <th scope="col" title="Expected points of the best bid">
                Best
              </th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: ROUNDS }, (_, i) => {
              const round = rounds[i];
              const finished = complete(round, players);
              const trump = round?.sit.trump;
              return (
                <tr key={i} className={i === current && !finished ? "row-now" : i > current ? "row-next" : ""}>
                  <th scope="row">
                    {i + 1}
                    {trump && (
                      <span className={isRed(trump.suit) ? "suit-red" : "suit-light"}> {SUIT_GLYPH[trump.suit]}</span>
                    )}
                  </th>
                  {players.map((player, p) => {
                    if (!finished) {
                      return (
                        <td key={player.name} className="cell-pending">
                          ·
                        </td>
                      );
                    }
                    const chosen = option(round.sit, round.bids[p]);
                    return (
                      <td key={player.name} className={tone(chosen.expected_points)} title={`Bid ${chosen.bid}`}>
                        {signed(chosen.expected_points)}
                      </td>
                    );
                  })}
                  <td className="cell-pending">{finished ? signed(bestOption(round.sit).expected_points) : "·"}</td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr>
              <th scope="row">Total</th>
              {players.map((player, p) => (
                <td key={player.name}>{signed(sums[p])}</td>
              ))}
              <td className="cell-pending">{signed(perfect)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}

/* ================================================================ table */

function SeatPlate({ seat, index, sit, bidder }: { seat: Seat; index: number; sit: Situation; bidder: Player }) {
  const angle = seatAngle(index, sit.players);
  return (
    <div
      className={`seat ${seat.is_you ? "seat-active" : ""}`}
      style={{ left: `${50 + Math.cos(angle) * 42}%`, top: `${50 + Math.sin(angle) * 40}%` }}
    >
      {!seat.is_you && (
        <div className="seat-cards" aria-label={`${plural(sit.tricks, "card")}`}>
          {Array.from({ length: sit.tricks }, (_, i) => (
            <span key={i} className="mini-back" />
          ))}
        </div>
      )}
      <div className="seat-plate">
        <div className="seat-head">
          <span className="seat-name">{seat.is_you ? bidder.name : seat.name}</span>
          {seat.is_dealer && (
            <span className="dealer-chip" title="Dealer">
              D
            </span>
          )}
          <span className="seat-score bg-order-tag" title="Place in the bidding">
            {ordinal(seat.order + 1)}
          </span>
        </div>
        <div className="seat-bid">
          {seat.bid !== null ? (
            <motion.span initial={{ scale: 1.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
              Bid {seat.bid}
            </motion.span>
          ) : (
            <span className={seat.is_you ? "bg-your-turn" : "seat-bid-empty"}>
              {seat.is_you ? "Your bid…" : "Bids after you"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function BidPanel({ sit, player, named, onBid }: { sit: Situation; player: Player; named: boolean; onBid: (bid: number) => void }) {
  const options = Array.from({ length: sit.tricks + 1 }, (_, i) => i);
  return (
    <motion.div
      className="bid-panel"
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.92 }}
    >
      <h2 className="bg-panel-title">
        {named && <Swatch color={player.color} />}
        {named ? `${player.name}'s bid` : "Your bid"}
      </h2>
      <div className="bid-options">
        {options.map((bid) => (
          <button
            key={bid}
            type="button"
            className="bid-option"
            disabled={!sit.legal_bids.includes(bid)}
            onClick={() => onBid(bid)}
          >
            {bid}
          </button>
        ))}
      </div>
      {sit.forbidden_bid !== null && (
        <p className="bid-note">
          As dealer you can't bid {sit.forbidden_bid}: bids can't add up to the hand size.
        </p>
      )}
    </motion.div>
  );
}

function HandFan({ sit, index }: { sit: Situation; index: number }) {
  if (sit.blind) {
    return (
      <div className="hand" key={index}>
        <div className="hand-card blind-card" aria-label="Your hidden card">
          <span aria-hidden="true">?</span>
        </div>
      </div>
    );
  }
  const cards = sortHand(sit.hand, sit.trump_suit);
  const middle = (cards.length - 1) / 2;
  return (
    <div className="hand" key={index} style={{ "--n": cards.length } as React.CSSProperties}>
      {cards.map((card, i) => {
        const offset = i - middle;
        return (
          <motion.div
            key={card.id}
            className="hand-card"
            style={{ zIndex: i + 1 }}
            role="img"
            aria-label={cardName(card)}
            initial={{ y: 70, opacity: 0, rotate: 0 }}
            animate={{ y: offset * offset * 0.6, opacity: 1, rotate: offset * 1.6 }}
            transition={{
              type: "spring",
              stiffness: 320,
              damping: 30,
              opacity: { delay: i * 0.035, duration: 0.2 },
            }}
          >
            <img src={cardImage(card)} alt="" draggable={false} />
          </motion.div>
        );
      })}
    </div>
  );
}

function Status({ sit, player, named }: { sit: Situation; player: Player; named: boolean }) {
  const soFar = sit.seats.reduce((sum, seat) => sum + (seat.bid ?? 0), 0);
  const who = named ? `${player.name}'s bid.` : "Your bid.";
  const message = sit.blind
    ? `${who} Blind round: bid without seeing your card.`
    : `${who} How many tricks will you take?`;
  return (
    <div className="status">
      <div className="trump">
        <img className="trump-card" src={cardImage(sit.trump)} alt={`Trump card: ${cardName(sit.trump)}`} />
        <div>
          <span className="label">Trump</span>
          <strong className={isRed(sit.trump.suit) ? "suit-red" : "suit-light"}>
            {SUIT_GLYPH[sit.trump.suit]} {sit.trump.suit}
          </strong>
        </div>
      </div>
      <p className="status-message status-yours" aria-live="polite">
        {message}
      </p>
      <div className="tally">
        <span className="label">
          {sit.players} players · {sit.blind ? "blind round · 1 trick" : plural(sit.tricks, "trick")}
        </span>
        <strong>
          {soFar} bid so far · you're {ordinal(sit.position + 1)}
        </strong>
      </div>
    </div>
  );
}

function CoverDialog({ player, other, previous, onReady }: { player: Player; other: Player; previous?: Player; onReady: () => void }) {
  return (
    <motion.div className="dialog-backdrop bg-cover" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <motion.div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-label={`${player.name}'s turn`}
        initial={{ y: 24, scale: 0.97 }}
        animate={{ y: 0, scale: 1 }}
      >
        <h2 className="bg-panel-title">
          <Swatch color={player.color} />
          {player.name}'s turn
        </h2>
        {previous && <p className="bg-note">{previous.name} has locked in a bid.</p>}
        <p>
          {other.name}, look away. {player.name}, tap below when only you can see the screen.
        </p>
        <button type="button" className="btn btn-primary btn-wide" onClick={onReady} autoFocus>
          Show {player.name}'s hand
        </button>
      </motion.div>
    </motion.div>
  );
}

interface TableScreenProps {
  sit?: Situation;
  index: number;
  player: Player;
  other?: Player;
  previous?: Player;
  covered: boolean;
  onUncover: () => void;
  onBid: (bid: number) => void;
  topbar: ReactNode;
  sidebar: ReactNode;
}

function TableScreen({ sit, index, player, other, previous, covered, onUncover, onBid, topbar, sidebar }: TableScreenProps) {
  const named = !!other;
  const showing = sit && !covered;
  return (
    <div className="game">
      {topbar}
      <main className="play">
        {showing ? (
          <Status sit={sit} player={player} named={named} />
        ) : (
          <div className="status">
            <span />
            <p className="status-message">{sit ? "" : "Dealing…"}</p>
          </div>
        )}
        <div className="table-wrap">
          <div className="felt">
            {sit ? (
              sit.seats.map((seat, i) => <SeatPlate key={i} seat={seat} index={i} sit={sit} bidder={player} />)
            ) : (
              <div className="bg-dealing">
                <div className="bg-spinner" />
                Dealing and working out the odds…
              </div>
            )}
            <AnimatePresence>
              {showing && <BidPanel key={`bid-${index}-${player.name}`} sit={sit} player={player} named={named} onBid={onBid} />}
            </AnimatePresence>
          </div>
        </div>
        {showing ? <HandFan key={`${index}-${player.name}`} sit={sit} index={index} /> : <div className="hand" />}
      </main>
      {sidebar}
      {covered && other && <CoverDialog player={player} other={other} previous={previous} onReady={onUncover} />}
    </div>
  );
}

/* ================================================================ results */

interface ResultsScreenProps {
  round: Round;
  index: number;
  players: Player[];
  topbar: ReactNode;
  sidebar: ReactNode;
  onNext: () => void;
}

function ResultsScreen({ round, index, players, topbar, sidebar, onNext }: ResultsScreenProps) {
  const { sit } = round;
  const best = bestOption(sit);
  const firstBid = round.bids[biddingOrder(players, index)[0]];
  const [shownBid, setShownBid] = useState(firstBid);
  const shown = option(sit, shownBid) ?? best;
  const picks = players.map((player, p) => ({ ...player, bid: round.bids[p], opt: option(sit, round.bids[p]) }));
  const top = Math.max(...picks.map((p) => p.opt.expected_points));
  const split = picks.length === 2 && picks[0].opt.expected_points !== picks[1].opt.expected_points;
  const last = index + 1 >= ROUNDS;
  const your = (p: Player) => (p.name === "You" ? "Your" : `${p.name}'s`);

  const few = shown.tricks.length <= 8;
  const peak = Math.max(...shown.tricks);

  return (
    <div className="game">
      {topbar}
      <main className="bg-report">
        <div className="bg-report-head">
          <div>
            <span className="label">Hand {index + 1} of {ROUNDS}</span>
            <h1 className="bg-title">
              {picks.every((p) => p.bid === sit.best_bid)
                ? players.length === 1
                  ? "Spot on: the best bid"
                  : "Both found the best bid"
                : `The best bid was ${sit.best_bid}`}
            </h1>
          </div>
          <button type="button" className="btn btn-primary" onClick={onNext} autoFocus>
            {last ? "See final score" : "Next hand"}
          </button>
        </div>

        <section className="bg-panel bg-recap">
          <div className="bg-recap-cards">
            <div>
              <span className="label">Trump</span>
              <img className="bg-recap-card" src={cardImage(sit.trump)} alt={cardName(sit.trump)} />
            </div>
            <div>
              <span className="label">{sit.blind ? "Hidden card" : "Your hand"}</span>
              <div className="bg-recap-hand">
                {sit.blind ? (
                  <div className="bg-recap-card blind-card" aria-label="Hidden card">
                    ?
                  </div>
                ) : (
                  sortHand(sit.hand, sit.trump_suit).map((card) => (
                    <img key={card.id} className="bg-recap-card" src={cardImage(card)} alt={cardName(card)} />
                  ))
                )}
              </div>
            </div>
          </div>
          <div className="bg-recap-bids">
            <span className="label">Bidding order</span>
            <ol className="bg-order">
              {[...sit.seats]
                .sort((a, b) => a.order - b.order)
                .map((seat) => (
                  <li key={seat.order} className={seat.is_you ? "bg-order-you" : ""}>
                    <span className="bg-order-name">
                      {seat.is_you ? (players.length === 1 ? "You" : "You two") : seat.name}
                      {seat.is_dealer && <span className="dealer-chip">D</span>}
                    </span>
                    <strong>
                      {seat.is_you
                        ? picks.map((p) => (
                            <span key={p.name} className="bg-order-pick">
                              {players.length > 1 && <Swatch color={p.color} />}
                              {p.bid}
                            </span>
                          ))
                        : seat.bid ?? "–"}
                    </strong>
                  </li>
                ))}
            </ol>
            <p className="bg-note">
              {sit.players} players · {sit.blind ? "blind round" : plural(sit.hand_size, "card")} · you bid{" "}
              {ordinal(sit.position + 1)}
              {sit.forbidden_bid !== null && ` · as dealer, ${sit.forbidden_bid} was not allowed`}
            </p>
          </div>
        </section>

        <div className="bg-tiles">
          {picks.map((p) => {
            const behind = best.expected_points - p.opt.expected_points;
            return (
              <div key={p.name} className={`bg-panel bg-tile ${split && p.opt.expected_points === top ? "bg-tile-win" : ""}`}>
                <div className="bg-tile-head">
                  <Swatch color={p.color} />
                  {p.name} bid
                </div>
                <div className="bg-hero">
                  {p.bid} <small>{p.bid === 1 ? "trick" : "tricks"}</small>
                </div>
                <dl>
                  <dt>Chance to make it</dt>
                  <dd>{pct(p.opt.make_chance)}</dd>
                  <dt>Expected points</dt>
                  <dd className={tone(p.opt.expected_points)}>{signed(p.opt.expected_points)}</dd>
                </dl>
                <p className="bg-note">
                  {p.bid === sit.best_bid
                    ? "The best bid."
                    : behind < 0.05
                      ? "As good as the best bid."
                      : `${behind.toFixed(1)} points behind the best bid.`}
                </p>
              </div>
            );
          })}
          <div className="bg-panel bg-tile bg-tile-best">
            <div className="bg-tile-head">
              <Swatch color="var(--brass)" kind="square" />
              Best bid
            </div>
            <div className="bg-hero">
              {best.bid} <small>{best.bid === 1 ? "trick" : "tricks"}</small>
            </div>
            <dl>
              <dt>Chance to make it</dt>
              <dd>{pct(best.make_chance)}</dd>
              <dt>Expected points</dt>
              <dd className={tone(best.expected_points)}>{signed(best.expected_points)}</dd>
            </dl>
            <p className="bg-note">From the advisor's simulated play-outs of this hand.</p>
          </div>
        </div>

        <section className="bg-panel bg-charts">
          <div className="bg-chart">
            <h2>Expected points for each bid</h2>
            <p className="bg-note">Average points over the simulated play-outs. Click a bar to see its tricks.</p>
            <div className="bg-legend">
              <span>
                <Swatch color="var(--brass)" kind="square" /> Best bid
              </span>
              {picks.map((p) => (
                <span key={p.name}>
                  <Swatch color={p.color} /> {your(p)} bid
                </span>
              ))}
            </div>
            <BarChart
              label="Expected points for each bid"
              axis="Bid"
              format={(v) => String(v)}
              onPick={(i) => setShownBid(sit.options[i].bid)}
              items={sit.options.map((o) => ({
                label: String(o.bid),
                value: o.expected_points,
                highlight: o.bid === sit.best_bid,
                marks: picks.filter((p) => p.bid === o.bid).map((p) => p.color),
                showValue: o.bid === sit.best_bid || picks.some((p) => p.bid === o.bid),
                text: signed(o.expected_points),
                tip: [`Bid ${o.bid}`, `Expected points ${signed(o.expected_points)}`, `Chance to make it ${pct(o.make_chance)}`],
              }))}
            />
          </div>
          <div className="bg-chart">
            <h2>
              Tricks taken when bidding {shown.bid} <small>· {pct(shown.make_chance)} to make it</small>
            </h2>
            <p className="bg-note">How often the hand ends with each number of tricks. Brass makes the bid.</p>
            <div className="bg-pills" role="group" aria-label="Choose a bid">
              {sit.options.map((o) => (
                <button
                  key={o.bid}
                  type="button"
                  className="bg-pill"
                  aria-pressed={o.bid === shown.bid}
                  onClick={() => setShownBid(o.bid)}
                >
                  {o.bid}
                </button>
              ))}
            </div>
            <BarChart
              label={`Tricks taken when bidding ${shown.bid}`}
              axis="Tricks taken"
              format={(v) => `${v}%`}
              floor={0}
              items={shown.tricks.map((p, k) => ({
                label: String(k),
                value: p * 100,
                highlight: k === shown.bid,
                marks: [],
                showValue: p > 0 && (few || k === shown.bid || p === peak),
                text: pct(p),
                tip: [
                  plural(k, "trick"),
                  `${pct(p)} of the time`,
                  k === shown.bid ? "Makes the bid" : `Misses by ${Math.abs(k - shown.bid)}`,
                ],
              }))}
            />
          </div>
          <details className="bg-numbers">
            <summary>Show the numbers</summary>
            <table className="results">
              <thead>
                <tr>
                  <th scope="col">Bid</th>
                  <th scope="col">Chance to make it</th>
                  <th scope="col">Expected points</th>
                  <th scope="col">
                    <span className="visually-hidden">Who</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {sit.options.map((o) => (
                  <tr key={o.bid} className={o.bid === sit.best_bid ? "row-you" : ""}>
                    <th scope="row">{o.bid}</th>
                    <td>{pct(o.make_chance)}</td>
                    <td className={tone(o.expected_points)}>{signed(o.expected_points)}</td>
                    <td>
                      {[o.bid === sit.best_bid ? "Best" : null, ...picks.filter((p) => p.bid === o.bid).map((p) => p.name)]
                        .filter(Boolean)
                        .join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </section>
      </main>
      {sidebar}
    </div>
  );
}

/* ================================================================ final */

interface FinalScreenProps {
  rounds: Round[];
  players: Player[];
  previousBest: number | null;
  onPlayAgain: () => void;
  onSettings: () => void;
}

function FinalScreen({ rounds, players, previousBest, onPlayAgain, onSettings }: FinalScreenProps) {
  const sums = totals(rounds, players).map((s) => Math.round(s * 10) / 10);
  const perfect = rounds.reduce((sum, round) => sum + bestOption(round.sit).expected_points, 0);
  const hits = players.map((_, p) => rounds.filter((r) => r.bids[p] === r.sit.best_bid).length);

  let headline: string;
  let detail: string;
  if (players.length === 2) {
    const [a, b] = sums;
    headline = a === b ? "It's a tie" : `${players[a > b ? 0 : 1].name} wins`;
    detail = `${players[0].name} ${signed(a)} (${hits[0]} best bids) · ${players[1].name} ${signed(b)} (${hits[1]} best bids)`;
  } else {
    const record = previousBest === null || sums[0] > previousBest;
    headline = `You scored ${signed(sums[0])}`;
    detail =
      `${hits[0]} of ${ROUNDS} best bids · ${Math.abs(perfect - sums[0]).toFixed(1)} points behind perfect bidding · ` +
      (record ? "a new personal best!" : `personal best ${signed(previousBest!)}`);
  }

  const series: Series[] = players.map((p) => ({ name: p.name, color: p.color, points: [0] }));
  series.push({ name: "Perfect", color: "var(--muted)", dashed: true, points: [0] });
  for (const round of rounds) {
    players.forEach((_, p) => {
      const points = series[p].points;
      points.push(points[points.length - 1] + option(round.sit, round.bids[p]).expected_points);
    });
    const ideal = series[series.length - 1].points;
    ideal.push(ideal[ideal.length - 1] + bestOption(round.sit).expected_points);
  }

  return (
    <main className="bg-final">
      <section className="bg-panel bg-final-head">
        <span className="label">Ten hands done</span>
        <h1 className="bg-title bg-title-big">{headline}</h1>
        <p className="bg-note">{detail}</p>
        <div className="bg-actions">
          <button type="button" className="btn btn-primary" onClick={onPlayAgain} autoFocus>
            Play again
          </button>
          <button type="button" className="btn btn-quiet" onClick={onSettings}>
            Change players
          </button>
        </div>
      </section>

      <section className="bg-panel bg-chart">
        <h2>Running score</h2>
        <p className="bg-note">Expected points added up hand by hand, against always making the best bid.</p>
        <div className="bg-legend">
          {series.map((s) => (
            <span key={s.name}>
              {s.dashed ? <Swatch kind="dash" /> : <Swatch color={s.color} />} {s.dashed ? "Perfect bidding" : s.name}
            </span>
          ))}
        </div>
        <LineChart series={series} />
      </section>

      <section className="bg-panel">
        <h2 className="bg-section-title">Hand by hand</h2>
        <div className="scoreboard-scroll">
          <table className="results bg-hands">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Hand</th>
                <th scope="col">Best bid</th>
                {players.map((p) => (
                  <th key={p.name} scope="col">
                    {p.name === "You" ? "Your bid" : p.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rounds.map((round, i) => {
                const { sit } = round;
                return (
                  <tr key={i}>
                    <th scope="row">{i + 1}</th>
                    <td className="bg-left">
                      <span className={isRed(sit.trump.suit) ? "suit-red" : "suit-light"}>{SUIT_GLYPH[sit.trump.suit]}</span>{" "}
                      {sit.players} players · {sit.blind ? "blind" : plural(sit.hand_size, "card")} ·{" "}
                      {ordinal(sit.position + 1)} to bid
                    </td>
                    <td>
                      {sit.best_bid} <span className={tone(bestOption(sit).expected_points)}>({signed(bestOption(sit).expected_points)})</span>
                    </td>
                    {players.map((p, k) => {
                      const chosen = option(sit, round.bids[k]);
                      return (
                        <td key={p.name}>
                          {chosen.bid} <span className={tone(chosen.expected_points)}>({signed(chosen.expected_points)})</span>
                          {chosen.bid === sit.best_bid && " ★"}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="bg-note">Each cell is the bid and its expected points. ★ marks the best bid.</p>
      </section>
    </main>
  );
}
