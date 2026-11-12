"""Search a policy over visible card histories, rather than separate known deals."""

import math
import time

from ..engine import PLAYING
from ..opponents import PopulationPolicy
from ..search import playout
from .heuristic import state_card
from .learned import HybridBot


MAX_TREE_NODES = 32768


class HistoryNode:
    def __init__(self):
        self.children = {}
        self.visits = 0
        self.total = 0.0


class InformationTree:
    """One tree shared by all hidden deals sampled for this turn.

    Edges are publicly played cards, never hidden hands or opponent kinds.
    Only our seat's choices are optimised; opponents follow the existing
    population model. Expand one untried choice per simulation, then finish
    with the rollout policy. Future choices improve as histories are revisited.
    """

    def __init__(self, cards, exploration=30.0, max_nodes=MAX_TREE_NODES):
        self.root = HistoryNode()
        self.cards = list(cards)  # root model ordering, not a shortlist
        self.exploration = exploration
        self.max_nodes = max_nodes
        self.nodes = 1
        self.iterations = 0
        self.deep_visits = 0

    def child(self, node, card):
        if card not in node.children:
            if self.nodes >= self.max_nodes:
                return None
            node.children[card] = HistoryNode()
            self.nodes += 1
        return node.children[card]

    def select(self, node, cards, preferred):
        children = [(c, self.child(node, c)) for c in cards]
        if any(child is None for _, child in children):
            return None
        untried = [c for c, child in children if not child.visits]
        if untried:
            return preferred if preferred in untried else untried[0]
        log_n = math.log(1 + sum(child.visits for _, child in children))
        return max(children, key=lambda pair: (
            pair[1].total / pair[1].visits + self.exploration *
            math.sqrt(log_n / pair[1].visits)))[0]

    def run(self, state, policy, seat):
        node = self.root
        path = []
        while state.phase == PLAYING:
            if state.to_act == seat:
                cards = self.cards if node is self.root else state.legal_actions()
                if len(cards) > 1:
                    preferred = cards[0] if node is self.root else state_card(state, policy.react)
                    card = self.select(node, cards, preferred)
                    if card is None:
                        playout(state, policy)
                        break
                    child = node.children[card]
                    fresh = child.visits == 0
                    path.append(child)
                    if node is not self.root:
                        self.deep_visits += 1
                    state.play_unchecked(card)
                    node = child
                    if fresh:
                        playout(state, policy)
                        break
                    continue
                card = cards[0]
            else:
                card = policy(state)
            child = self.child(node, card)
            state.play_unchecked(card)
            if child is None:
                playout(state, policy)
                break
            node = child
        reward = state.points()[seat]
        for child in path:
            child.visits += 1
            child.total += reward
        self.iterations += 1

    def result(self):
        return {
            'visits': {c: self.root.children[c].visits if c in self.root.children else 0
                       for c in self.cards},
            'totals': {c: self.root.children[c].total if c in self.root.children else 0.0
                       for c in self.cards},
            'iterations': self.iterations, 'nodes': self.nodes,
            'deep_visits': self.deep_visits,
        }


class PlanningBot(HybridBot):
    """Hybrid bids/endgames with visible-history search for earlier card play.

    The model orders the first visit to each root action only. Every legal
    card is explored, with no model score blended into the final result.
    The most visited root action wins (mean score breaks ties).
    Untimed search uses ``worlds * legal_cards`` complete simulations;
    timed search shares a single turn deadline. Trees are rebuilt each turn.
    """

    def __init__(self, seed=0, exploration=30.0, tree_batch=8, **kwargs):
        if not math.isfinite(exploration) or exploration <= 0:
            raise ValueError('exploration must be finite and positive.')
        if not isinstance(tree_batch, int) or not 1 <= tree_batch <= 1024:
            raise ValueError('tree_batch must be an integer from 1 to 1024.')
        super().__init__(seed=seed, **kwargs)
        if not isinstance(self.worlds, int) or self.worlds < 1:
            raise ValueError('worlds must be a positive integer.')
        self.exploration = exploration
        self.tree_batch = tree_batch
        self.last_search = None

    def play(self, obs):
        self.last_search = None
        cards = obs.legal_actions()
        if len(cards) == 1:
            return cards[0]
        remaining = obs.hand.bit_count() * obs.n - obs.trick_size
        if remaining <= self.endgame_cards:
            return super().play(obs)
        predicted = dict(zip(cards, self.models.card_values(obs, obs.seat, obs.hand, cards)))
        cards.sort(key=predicted.get, reverse=True)
        tree = InformationTree(cards, self.exploration)
        budget = self.worlds * len(cards)
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            result = tree.result()
        elif self.simulator:
            result = self.simulator.plan_cards(
                obs, cards, budget, self.exploration, self.tree_batch, self._time_left())
        else:
            while tree.iterations < budget or self.deadline is not None:
                if self.deadline is not None and time.perf_counter() >= self.deadline:
                    break
                count = (self.tree_batch if self.deadline is not None else
                         min(self.tree_batch, budget - tree.iterations))
                for hands, kinds in self.belief.worlds(obs, count, self.rng):
                    policy = PopulationPolicy(kinds, self.rng, self.react)
                    tree.run(obs.determinize(hands), policy, obs.seat)
                    if self.deadline is not None and time.perf_counter() >= self.deadline:
                        break
            result = tree.result()
        self.last_search = result

        def strength(card):
            visits = result['visits'][card]
            return visits, result['totals'][card] / visits if visits else -math.inf

        return max(cards, key=strength)
