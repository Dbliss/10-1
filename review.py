"""A chess-style review of the user's decisions in one game.

``Game`` records each bid and card the user chooses, with the observation the
user had at that moment. ``analyse_next`` asks the advisor to value every
option of the oldest unreviewed decision; the web server calls it in the
background while the game goes on, so most of the review is ready by the end.

Each decision is scored by the expected points the user's choice gave up
against the advisor's best option. These are the advisor's estimates from a
limited number of sampled deals, so small losses are within its noise.
"""

import math
import random
import threading

from ohhell.bots import make_bot
from ohhell.cards import card_id
from ohhell.engine import BIDDING

# (name, the largest loss in expected points it covers), best to worst.
GRADES = [
    ('best', 0.25),
    ('excellent', 1.0),
    ('good', 2.0),
    ('inaccuracy', 4.0),
    ('mistake', 8.0),
    ('blunder', math.inf),
]
# Accuracy for one decision is 100 * exp(-loss / ACCURACY_SCALE): a 1-point
# loss scores 82%, a 4-point inaccuracy 45%, an 8-point blunder 20%.
ACCURACY_SCALE = 5.0


def grade(loss):
    for name, limit in GRADES:
        if loss <= limit:
            return name
    return GRADES[-1][0]


def accuracy(losses):
    if not losses:
        return None
    return round(sum(100 * math.exp(-loss / ACCURACY_SCALE) for loss in losses) / len(losses), 1)


class GameReview:
    def __init__(self, advisor_spec):
        self.advisor_spec = advisor_spec
        self.decisions = []  # in the order they were made
        self.reviewed = 0  # decisions before this index have their analysis
        self.lock = threading.Lock()  # guards the list against the request threads

    def record(self, obs, choice, describe, context):
        """Remember a decision the user made. ``context`` is what the UI shows
        and ``describe`` turns one of its cards into JSON under its trump.

        Decisions with only one legal option are not reviewed.
        """
        legal = obs.legal_actions()
        if len(legal) < 2:
            return
        with self.lock:
            self.decisions.append({'obs': obs, 'choice': choice, 'describe': describe,
                                   'context': context, 'analysis': None})

    def pending(self):
        with self.lock:
            return len(self.decisions) - self.reviewed

    def analyse_next(self):
        """Analyse the oldest unreviewed decision. Returns False if none is left.

        The caller must hold the lock that keeps the native simulator to one
        thread at a time.
        """
        with self.lock:
            if self.reviewed >= len(self.decisions):
                return False
            decision = self.decisions[self.reviewed]
        decision['analysis'] = self._analyse(decision['obs'], decision['choice'])
        with self.lock:
            self.reviewed += 1
        return True

    def _analyse(self, obs, choice):
        if obs.blind:  # only the blind bid is a choice; its odds are exact
            from ohhell.blind import bid_values
            values, probability = bid_values(obs)
            return {
                'values': values,
                'make': {bid: probability if bid else 1 - probability for bid in values},
            }
        advisor = make_bot(self.advisor_spec, random.getrandbits(32))
        if hasattr(advisor, 'start_clock'):
            advisor.start_clock()
        if obs.phase == BIDDING:
            # The advisor only analyses the bids it thinks plausible; add the
            # user's bid so it can be compared.
            shortlist = advisor._bid_options
            advisor._bid_options = lambda o: sorted(set(shortlist(o)) | {choice})
            analysis = advisor.analyse_bids(obs)
            return {
                'values': dict(analysis.values),
                'make': {bid: dist[bid] for bid, dist in analysis.distributions.items()},
            }
        return {'values': dict(advisor.evaluate_cards(obs)), 'make': None}

    def report(self, final_score):
        """The review as JSON-friendly data, with whatever is analysed so far."""
        with self.lock:
            done = self.decisions[:self.reviewed]
            total = len(self.decisions)

        moves = []
        for index, decision in enumerate(done):
            analysis, choice = decision['analysis'], decision['choice']
            values = analysis['values']
            best = max(values, key=values.get)
            loss = max(0.0, values[best] - values[choice])
            is_bid = decision['obs'].phase == BIDDING
            options = sorted(values, key=lambda option: -values[option])
            moves.append({
                'index': index,
                'kind': 'bid' if is_bid else 'card',
                **decision['context'],
                'choice': choice if is_bid else card_id(choice),
                'best': best if is_bid else card_id(best),
                'loss': round(loss, 2),
                'grade': grade(loss),
                'options': [
                    {
                        'bid' if is_bid else 'card':
                            option if is_bid else decision['describe'](option),
                        'expected_points': round(values[option], 1),
                        **({'make_chance': round(analysis['make'][option], 3)}
                           if is_bid else {}),
                    }
                    for option in options
                ],
            })

        def summary(kind=None):
            chosen = [m for m in moves if kind is None or m['kind'] == kind]
            losses = [m['loss'] for m in chosen]
            return {
                'accuracy': accuracy(losses),
                'decisions': len(chosen),
                'points_lost': round(sum(losses), 1),
                'grades': {name: sum(m['grade'] == name for m in chosen) for name, _ in GRADES},
            }

        overall = summary()
        return {
            'complete': len(done) == total,
            'reviewed': len(done),
            'total': total,
            'overall': overall,
            'bidding': summary('bid'),
            'playing': summary('card'),
            'score': final_score,
            'expected_best_score': round(final_score + overall['points_lost'], 1),
            'moves': moves,
        }
