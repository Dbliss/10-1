import { useEffect } from "react";
import { preloadCardImages } from "./cards";
import { useGame } from "./useGame";
import { GameScreen } from "./components/GameScreen";
import { SetupScreen } from "./components/SetupScreen";

export default function App() {
  const { game, loading, error, retry, newGame, endGame, bid, play, nextRound } = useGame();

  useEffect(preloadCardImages, []);

  return (
    <>
      {!loading &&
        (game ? (
          <GameScreen
            game={game}
            onBid={bid}
            onPlay={play}
            onNextRound={nextRound}
            onPlayAgain={() => newGame(game.players.length, game.total_rounds)}
            onQuit={endGame}
          />
        ) : (
          <SetupScreen onStart={newGame} />
        ))}
      {error && (
        <div className="toast" role="alert">
          <span>{error}</span>
          <button type="button" className="btn btn-quiet" onClick={retry}>
            Retry
          </button>
        </div>
      )}
    </>
  );
}
