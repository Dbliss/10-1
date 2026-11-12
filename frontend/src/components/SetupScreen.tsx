import { useState } from "react";

const PLAYER_COUNTS = [2, 3, 4, 5, 6, 7, 8];
const maxHandSize = (players: number) => Math.min(10, Math.floor(51 / players));

interface ChoiceRowProps {
  label: string;
  options: number[];
  value: number;
  onChange: (value: number) => void;
}

function ChoiceRow({ label, options, value, onChange }: ChoiceRowProps) {
  return (
    <fieldset className="choice">
      <legend>{label}</legend>
      <div className="choice-options">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            className="choice-option"
            aria-pressed={option === value}
            onClick={() => onChange(option)}
          >
            {option}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function SetupScreen({ onStart }: { onStart: (players: number, handSize: number) => void }) {
  const [players, setPlayers] = useState(4);
  const [requestedHand, setRequestedHand] = useState(10);

  const maxHand = maxHandSize(players);
  const handSize = Math.min(requestedHand, maxHand);
  const handOptions = Array.from({ length: maxHand }, (_, i) => maxHand - i);

  return (
    <main className="setup">
      <div className="setup-fan" aria-hidden="true">
        {["ace_of_spades", "king_of_hearts", "queen_of_clubs"].map((card, i) => (
          <img key={card} src={`/cards/${card}.png`} alt="" style={{ "--i": i - 1 } as React.CSSProperties} />
        ))}
      </div>

      <h1>Oh Hell</h1>
      <p className="setup-lede">
        Call exactly how many tricks you&rsquo;ll take. Then take exactly that many.
      </p>

      <ChoiceRow label="Players" options={PLAYER_COUNTS} value={players} onChange={setPlayers} />
      <ChoiceRow
        label="Starting hand"
        options={handOptions}
        value={handSize}
        onChange={setRequestedHand}
      />

      <button type="button" className="btn btn-primary btn-wide" onClick={() => onStart(players, handSize)}>
        Deal {handSize} {handSize === 1 ? "card" : "cards"}
      </button>

      <p className="setup-rules">
        Deal down to one, then a blind round with one hidden card each, then back up to your
        starting hand. Scores carry through all {2 * handSize + 1} rounds.
        Must follow suit; trumps beat everything else. The Jack of trumps is highest, then the
        other Jack of the same colour, which counts as a trump. Make your bid for 10 points per trick
        (a made zero scores 10). Miss it and lose 10 points for every trick you were off.
      </p>
    </main>
  );
}
