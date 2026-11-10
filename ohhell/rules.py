"""The configurable parts of the game."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scoring:
    """+10 per trick for a made bid, +10 for a made zero, -10 per trick off."""

    points_per_trick: int = 10
    made_zero: int = 10
    penalty_per_trick: int = 10

    def points(self, bid, tricks):
        if bid != tricks:
            return -self.penalty_per_trick * abs(bid - tricks)
        return self.made_zero if bid == 0 else self.points_per_trick * bid


@dataclass(frozen=True)
class Rules:
    num_players: int = 4
    max_hand: int = 10
    # When true, the last bidder may not make the bids add up to the hand size.
    dealer_restriction: bool = False
    scoring: Scoring = field(default_factory=Scoring)

    def __post_init__(self):
        if not 2 <= self.num_players <= 8:
            raise ValueError("Oh Hell needs 2 to 8 players.")
        # One card has to remain to turn up as trump.
        if not 1 <= self.max_hand <= 51 // self.num_players:
            raise ValueError(
                f"Hand size must be 1 to {51 // self.num_players} "
                f"for {self.num_players} players.")

    @property
    def hand_sizes(self):
        return tuple(range(self.max_hand, 0, -1))
