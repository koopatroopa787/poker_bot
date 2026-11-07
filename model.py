import brain
from card_utils import Card
from game_state import GameStateView
from player_action import DEALER, PlayerAction

STREETS = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}


def _cs(card):
    return card.rank + card.suit.lower()


class Model:
    """Adapter: rebuilds the hand's betting history from the engine's actions, asks
    brain.decide for an action, and keeps opponent stats per team name across hands."""

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.stats = {}  # team name -> brain stats dict (plain data, survives hands)
        self._reset()

    def _reset(self):
        self.hist, self.bets, self.street, self.nboard = [], None, 0, 0

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self._reset()

    def _opp(self, gs):
        others = [p for p in range(gs.num_players) if p != self.player_id]
        live = [p for p in others if not gs.folded[p]] or others
        return max(live, key=lambda p: gs.bets[p])

    def _key(self, gs, opp):
        teams = gs.teams
        return teams[opp] if teams else 'opp'

    def _ingest(self, actions, gs):
        if self.bets is None:  # blinds are posted without actions
            n = gs.num_players
            self.bets = [0] * n
            sb, bb = (gs.button, 1 - gs.button) if n == 2 else ((gs.button + 1) % n, (gs.button + 2) % n)
            self.bets[sb], self.bets[bb] = gs.small_blind, gs.big_blind
        for a in actions:
            if a.kind == 'reveal_card':
                if a.player_id == DEALER:
                    self.nboard += 1
                    if self.nboard in (3, 4, 5):
                        self.street += 1
                        self.bets = [0] * len(self.bets)
                continue
            if a.kind == 'win':
                continue
            p = a.player_id
            who = 'me' if p == self.player_id else 'opp'
            if a.kind == 'fold':
                self.hist.append((self.street, who, 'f', None, False))
            elif a.kind == 'call':
                top = max(self.bets)
                act = 'c' if top > self.bets[p] else 'k'
                self.bets[p] = top
                self.hist.append((self.street, who, act, None, gs.status(p) == 'all_in'))
            elif a.kind == 'raise':
                self.bets[p] = a.amount
                self.hist.append((self.street, who, 'r', a.amount, gs.status(p) == 'all_in'))

    def run(self, actions: list[PlayerAction], game_state: GameStateView) -> PlayerAction:
        gs, me = game_state, self.player_id
        try:
            self._ingest(actions, gs)
            opp = self._opp(gs)
            s = self.stats.setdefault(self._key(gs, opp), brain.new_stats())
            can = gs.can_raise
            v = dict(hole=[_cs(c) for c in self.hand], board=[_cs(c) for c in gs.board],
                     street=STREETS[gs.street], pot=gs.pot, to_call=gs.call_amount,
                     my_bet=gs.bets[me], opp_bet=gs.bets[opp], my_stack=gs.stacks[me],
                     opp_stack=gs.stacks[opp], min_raise_to=gs.min_raise_to if can else None,
                     max_raise_to=gs.max_raise_to if can else None, bb=gs.big_blind,
                     history=self.hist)
            act, amt = brain.decide(v, s)
            if act == 'r' and can:
                lo, hi = gs.raise_range
                return PlayerAction.raise_to(me, int(min(hi, max(lo, amt))))
            if act == 'f' and gs.call_amount > 0:
                return PlayerAction.fold(me)
            return PlayerAction.call(me)
        except Exception:  # never fault: check if free, otherwise fold
            return PlayerAction.call(me) if gs.call_amount == 0 else PlayerAction.fold(me)

    def end_game(self, actions: list[PlayerAction], game_state: GameStateView):
        gs = game_state
        try:
            self._ingest(actions, gs)
            opp = next(p for p in range(gs.num_players) if p != self.player_id)
            brain.learn(self.stats.setdefault(self._key(gs, opp), brain.new_stats()), self.hist)
        except Exception:
            pass
