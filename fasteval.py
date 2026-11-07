"""Fast pure-Python 5-7 card evaluator (card_utils.evaluate is too slow for Monte Carlo).

Cards are strings like 'As' or 'AS'. score(cards) returns an int: higher wins, equal splits.
"""
RANKS = '23456789TJQKA'
CARD = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate('cdhs')}
CARD.update({k[0] + k[1].upper(): v for k, v in list(CARD.items())})

POP = [bin(m).count('1') for m in range(8192)]
STRAIGHT = [-1] * 8192
for _m in range(8192):
    for _hi in range(12, 3, -1):
        if (_m >> (_hi - 4)) & 31 == 31:
            STRAIGHT[_m] = _hi
            break
    else:
        if _m & 0b1000000001111 == 0b1000000001111:  # wheel A-2-3-4-5
            STRAIGHT[_m] = 3


def _top(mask, n):
    v = 0
    for r in range(12, -1, -1):
        if mask >> r & 1:
            v = v << 4 | r
            n -= 1
            if not n:
                break
    return v


def score(cards):
    sm = [0, 0, 0, 0]
    cnt = [0] * 13
    for c in cards:
        c = CARD[c]
        r = c >> 2
        cnt[r] += 1
        sm[c & 3] |= 1 << r
    for m in sm:
        if POP[m] >= 5:
            s = STRAIGHT[m]
            return (8 << 20 | s) if s >= 0 else (5 << 20 | _top(m, 5))
    mask = sm[0] | sm[1] | sm[2] | sm[3]
    q, t, p = [], [], []
    for r in range(12, -1, -1):
        n = cnt[r]
        if n == 4: q.append(r)
        elif n == 3: t.append(r)
        elif n == 2: p.append(r)
    if q:
        return 7 << 20 | q[0] << 16 | _top(mask & ~(1 << q[0]), 1) << 12
    if t and (len(t) > 1 or p):
        pair = max(t[1] if len(t) > 1 else -1, p[0] if p else -1)
        return 6 << 20 | t[0] << 16 | pair << 12
    s = STRAIGHT[mask]
    if s >= 0:
        return 4 << 20 | s
    if t:
        return 3 << 20 | t[0] << 16 | _top(mask & ~(1 << t[0]), 2) << 8
    if len(p) >= 2:
        return 2 << 20 | p[0] << 16 | p[1] << 12 | _top(mask & ~(1 << p[0]) & ~(1 << p[1]), 1) << 8
    if p:
        return 1 << 20 | p[0] << 16 | _top(mask & ~(1 << p[0]), 3) << 4
    return _top(mask, 5)
