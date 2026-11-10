import { isRed, SUIT_GLYPH } from "../cards";
import type { GameState, RoundResult } from "../types";

function formatPoints(points: number) {
  return points > 0 ? `+${points}` : `${points}`;
}

function ResultCell({ result }: { result: RoundResult }) {
  const tone = result.points > 0 ? "gain" : result.points < 0 ? "loss" : "";
  return (
    <td className={tone} title={`Bid ${result.bid}, won ${result.tricks}`}>
      {formatPoints(result.points)}
    </td>
  );
}

export function Scoreboard({ game }: { game: GameState }) {
  const rounds = Array.from({ length: game.total_rounds }, (_, i) => i + 1);

  return (
    <section className="scoreboard" aria-label="Scoreboard">
      <h2>Scores</h2>
      <div className="scoreboard-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">
                <span className="visually-hidden">Cards dealt</span>
              </th>
              {game.players.map((player) => (
                <th key={player.name} scope="col" className={player.is_human ? "col-you" : ""}>
                  {player.name.replace("AI ", "A")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rounds.map((round) => {
              const history = game.history.find((entry) => entry.round === round);
              const trump = history?.trump ?? (round === game.round ? game.trump : null);
              const handSize = game.round_sizes[round - 1];
              const isCurrent = round === game.round && !history;
              return (
                <tr key={round} className={isCurrent ? "row-now" : round > game.round ? "row-next" : ""}>
                  <th scope="row" title={handSize === 0 ? "Blind round: one hidden card each" : undefined}>
                    {handSize}
                    {trump && (
                      <span className={isRed(trump.suit) ? "suit-red" : "suit-light"}>
                        {" "}
                        {SUIT_GLYPH[trump.suit]}
                      </span>
                    )}
                  </th>
                  {game.players.map((player, seat) =>
                    history ? (
                      <ResultCell key={player.name} result={history.results[seat]} />
                    ) : (
                      <td key={player.name} className="cell-pending">
                        {isCurrent && player.bid !== null ? `bid ${player.bid}` : "·"}
                      </td>
                    ),
                  )}
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr>
              <th scope="row">Total</th>
              {game.players.map((player) => (
                <td key={player.name}>{player.score}</td>
              ))}
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}
