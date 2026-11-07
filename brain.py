"""Exploitative heads-up NLHE decision core. No engine imports: plain data in, action out.

decide(view, stats) -> ('f'|'k'|'c', None) or ('r', raise_to)
learn(stats, history) updates the opponent stats dict after a hand.

view keys: hole, board (card strings like 'As'), street (0-3), pot, to_call, my_bet,
  opp_bet, my_stack, opp_stack, min_raise_to, max_raise_to (None = no raise), bb, history
history: list of (street, 'me'|'opp', act 'f'|'k'|'c'|'r', raise_to_or_None, allin)

Each decision:
1. Opponent range: preflop top-x% by equity, x from their raise/shove/call rates;
   postflop keep the top q% by made-hand strength on each street they bet or called.
2. Monte Carlo equity against that range, and against the part that continues vs a raise.
3. Highest one-step EV of fold / check-call / raise sizes, with a rough future-street term.

ponytail: one-step EV ignores multi-street plans; made-hand ranking ignores draws.
"""
import random
from fasteval import score
from preflop_table import PRE

R = '23456789TJQKA'
DECK = [r + s for r in R for s in 'shdc']


def hand_class(a, b):
    if R.index(a[0]) < R.index(b[0]):
        a, b = b, a
    return a[0] + b[0] + ('' if a[0] == b[0] else 's' if a[1] == b[1] else 'o')


COMBOS = [(DECK[i], DECK[j]) for i in range(52) for j in range(i + 1, 52)]
PREQ = {h: PRE[hand_class(*h)] for h in COMBOS}
ORDER = sorted(COMBOS, key=lambda h: -PREQ[h])
SAMPLES = 150


def new_stats():
    # faced/folds: opponent facing our bet; faced2/folds2: facing our re-raise after it raised
    return dict(hands=0, vpip=0, pfr=0, shove=0, faced=0, folds=0, faced2=0, folds2=0, post=0, agg=0)


def rate(k, n, prior, w=4):
    return (k + prior * w) / (n + w)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def learn(s, h):
    s['hands'] += 1
    pf = [x for x in h if x[0] == 0 and x[1] == 'opp']
    s['vpip'] += any(x[2] in 'cr' for x in pf)
    s['pfr'] += any(x[2] == 'r' for x in pf)
    s['shove'] += any(x[2] == 'r' and x[4] for x in pf)
    prev, opp_raised = None, set()  # streets where the opponent has raised so far
    for x in h:
        if x[1] == 'opp':
            if prev and prev[1] == 'me' and prev[2] == 'r':
                k = '2' if x[0] in opp_raised else ''
                s['faced' + k] += 1
                s['folds' + k] += x[2] == 'f'
            if x[2] == 'r':
                opp_raised.add(x[0])
            if x[0] > 0:
                s['post'] += 1
                s['agg'] += x[2] == 'r'
        prev = x


def opp_range(v, s):
    h = v['history']
    dead = set(v['hole']) | set(v['board'])
    pf = [x for x in h if x[0] == 0 and x[1] == 'opp']
    nr = sum(x[2] == 'r' for x in pf)
    if any(x[2] == 'r' and x[4] for x in pf):
        x = rate(s['shove'], s['hands'], .08)
    elif nr:
        x = rate(s['pfr'], s['hands'], .45) * .6 ** (nr - 1)
    elif any(x[2] == 'c' for x in pf):
        x = rate(s['vpip'], s['hands'], .75)
    else:
        x = 1.0
    pool = [c for c in ORDER[:max(60, int(clamp(x, .04, 1) * 1326))]
            if c[0] not in dead and c[1] not in dead]
    hands = [random.choice(pool) for _ in range(SAMPLES)]
    hands.sort(key=lambda c: -PREQ[c])
    for st in range(1, v['street'] + 1):
        acts = [x[2] for x in h if x[0] == st and x[1] == 'opp']
        if 'r' in acts:
            q = clamp(rate(s['agg'], s['post'], .35) * 1.3, .2, 1) * .7 ** (acts.count('r') - 1)
        elif 'c' in acts:
            q = clamp(1 - rate(s['folds'], s['faced'], .35), .3, 1)
        else:
            q = 1.0
        b = list(v['board'][:st + 2])
        hands.sort(key=lambda c: -score(b + list(c)))
        hands = hands[:max(12, int(len(hands) * q))]
    return hands  # strongest first


def equities(v, hands):
    me, b = list(v['hole']), list(v['board'])
    dead = set(me) | set(b)
    rest = [c for c in DECK if c not in dead]
    need, out = 5 - len(b), []
    mine = score(me + b) if not need else None
    for hh in hands:
        if not need:
            y = score(list(hh) + b)
            out.append(1 if mine > y else .5 if mine == y else 0)
            continue
        left = [c for c in rest if c != hh[0] and c != hh[1]]
        w = 0.0
        for _ in range(2):
            full = b + random.sample(left, need)
            x, y = score(me + full), score(list(hh) + full)
            w += 1 if x > y else .5 if x == y else 0
        out.append(w / 2)
    return out


def decide(v, s):
    P, c = v['pot'], v['to_call']
    res = equities(v, opp_range(v, s))
    e = sum(res) / len(res)

    def cont(frac):  # equity vs the strongest `frac` of the range
        k = max(1, int(len(res) * frac))
        return sum(res[:k]) / k

    g = (1.5, 1.0, .6, 0)[v['street']]

    def fv(eq, pot, behind):  # rough value of later streets: strong hands grow the pot
        return .4 * (2 * eq - 1) * min(behind, g * pot)

    behind = min(v['my_stack'] - c, v['opp_stack'])
    opts = [(('f', None), 0.0)] if c else [(('k', None), e * P + fv(e, P, behind))]
    if c:
        real = 1.0 if v['street'] == 3 or v['opp_stack'] == 0 or c >= v['my_stack'] else .92
        opts.append((('c', None), e * real * (P + c) - c + fv(e, P + c, max(behind, 0))))
    my_raises = sum(1 for x in v['history'] if x[0] == v['street'] and x[1] == 'me' and x[2] == 'r')
    if v['min_raise_to'] and v['opp_stack'] > 0 and my_raises < 3:
        lo, hi = v['min_raise_to'], v['max_raise_to']
        cap = clamp(v['opp_bet'] + v['opp_stack'], lo, hi)
        # Fold model: the opponent continues with (1-f) of its range against a 0.75-pot bet
        # (f = observed fold rate), and a bit less against bigger bets.
        # ponytail: one curve for all sizes; learn per-size fold rates if opponents differ by size.
        # An opponent that already raised this street folds far less to a re-raise: use its own rate.
        if any(x[0] == v['street'] and x[1] == 'opp' and x[2] == 'r' for x in v['history']):
            f = clamp(rate(s['folds2'], s['faced2'], .25), .01, .9)
        else:
            f = clamp(rate(s['folds'], s['faced'], .4), .01, .9)
        if v['street'] == 0:
            tos = {clamp(int(v['opp_bet'] * m), lo, cap) for m in (2.5, 3, 4)}
        else:
            tos = {clamp(int(v['opp_bet'] + fr * (P + c)), lo, cap) for fr in (.5, .75, 1.0)}
        if cap - v['opp_bet'] <= 1.5 * (P + c) or v['my_stack'] <= 20 * v['bb'] or f < .12:
            tos.add(cap)  # shove only when not a huge overbet, unless they never fold
        for to in tos:
            d, call_amt = to - v['my_bet'], to - v['opp_bet']
            k = clamp((1 - f) * (1.75 * P / (P + call_amt)) ** .3, .03, 1)
            es = cont(k)
            left = min(v['my_stack'] - d, v['opp_stack'] - call_amt)
            won = es * (P + d + call_amt) - d + fv(es, P + d + call_amt, max(left, 0))
            opts.append((('r', to), (1 - k) * P + k * won))
    best = max(ev for _, ev in opts)
    return random.choice([a for a, ev in opts if ev >= best - 1e-9])
