/* Fast versions of the simulation loops in belief.py, search.py and heuristic.py.
 *
 * The Python code is the reference and is used when this library isn't built.
 * tests/test_native.py checks that the two agree on the deterministic parts.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
__declspec(dllimport) int __stdcall QueryPerformanceCounter(long long *count);
__declspec(dllimport) int __stdcall QueryPerformanceFrequency(long long *rate);
static double now_seconds(void) {
    static double tick = 0.0;
    long long count;
    if (tick == 0.0) {
        long long rate;
        QueryPerformanceFrequency(&rate);
        tick = 1.0 / (double)rate;
    }
    QueryPerformanceCounter(&count);
    return (double)count * tick;
}
#else
#include <time.h>
static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
#endif

#define EXPORT __attribute__((visibility("default")))
#define MAX_SEATS 8
#define MAX_KINDS 4
#define NO_SUIT (-1)
#define NO_CARD (-1)
#define NO_BID (-1)
#define BIDDING 0
#define PLAYING 1
#define FULL_DECK ((UINT64_C(1) << 52) - 1)
#define EST_SIDE 65  /* len(estimate.SIDE) */
#define EST_TRUMP 57 /* len(estimate.TRUMP) */

typedef uint64_t u64;

typedef struct {
    int n, hand_size, trump_card, dealer, seat, phase;
    int lead_suit, best_card, best_seat, trick_size, to_act, leader;
    int bids[MAX_SEATS], tricks_won[MAX_SEATS], voids[MAX_SEATS];
    u64 hand, played, table;
    int history_len;
    int history_seat[52], history_card[52];
} Obs;

typedef struct {
    double card_noise, bid_one, bid_two, bid_sigma, play_doubt;
} Kind;

typedef struct {
    int worlds, endgame_cards, inference, oversample;
    int points_per_trick, made_zero, penalty_per_trick;
    double weights[MAX_KINDS];
    Kind kinds[MAX_KINDS];
    double time_limit; /* seconds to keep sampling for; 0 means one block of worlds */
    int react;         /* simulated players take other seats' bids into account */
    /* Which quick trick estimate imagines and judges other seats' bids:
     * 1 is the original rule of thumb, 2 or 3 the calibrated one in
     * estimate.py, whose fitted weights follow. est_spread widens the trick
     * distribution an imagined bidder picks from (estimate.widen); version 2
     * uses {1, 0, 0}, which leaves it unchanged. */
    int estimate;
    double est_side[EST_SIDE];
    double est_trump[EST_TRUMP];
    double est_spread[3];
} Params;

typedef struct {
    u64 hands[MAX_SEATS], played, table;
    int bids[MAX_SEATS], won[MAX_SEATS];
    int to_act, lead_suit, best_card, best_seat, trick_size, leader, done;
} State;

typedef struct {
    int n, trump_suit;
    u64 trump_bit;
    int react;
} Table;

typedef struct {
    u64 hands[MAX_SEATS];
    int kinds[MAX_SEATS];
} World;

static u64 SUIT_MASK[4];
static u64 ABOVE[52];
/* Bowers: per trump suit, the Jack of trump ranks first and the other Jack of
 * the same colour second; that left bower is a trump for every purpose. */
static u64 T_SUIT[4][4];   /* cards of each effective suit */
static u64 T_OUT[4][52];   /* cards of the same effective suit that outrank a card */
static int T_EFF[4][52];   /* a card's effective suit */
static int T_POW[4][52];   /* strength within the effective suit; bowers 14 and 13 */
static int T_RIGHT[4], T_LEFT[4];
static u64 T_BOWERS[4];
static int ready = 0;

#define SUIT_OF(c) ((c) / 13)
#define RANK_OF(c) ((c) % 13)
#define POP(x) __builtin_popcountll(x)
#define HIGHEST(x) (63 - __builtin_clzll(x))
#define LOWEST(x) (__builtin_ctzll(x))

static void init_tables(void) {
    if (ready) return;
    for (int s = 0; s < 4; s++) SUIT_MASK[s] = UINT64_C(0x1FFF) << (13 * s);
    for (int c = 0; c < 52; c++)
        ABOVE[c] = SUIT_MASK[SUIT_OF(c)] & ~((UINT64_C(2) << c) - 1);
    for (int t = 0; t < 4; t++) {
        T_RIGHT[t] = 13 * t + 9;
        T_LEFT[t] = 13 * (3 - t) + 9;
        T_BOWERS[t] = (UINT64_C(1) << T_RIGHT[t]) | (UINT64_C(1) << T_LEFT[t]);
        for (int s = 0; s < 4; s++) T_SUIT[t][s] = 0;
        for (int c = 0; c < 52; c++) {
            T_EFF[t][c] = c == T_LEFT[t] ? t : SUIT_OF(c);
            T_POW[t][c] = c == T_RIGHT[t] ? 14 : c == T_LEFT[t] ? 13 : RANK_OF(c);
            T_SUIT[t][T_EFF[t][c]] |= UINT64_C(1) << c;
        }
        for (int c = 0; c < 52; c++) {
            T_OUT[t][c] = 0;
            for (int o = 0; o < 52; o++)
                if (T_EFF[t][o] == T_EFF[t][c] && T_POW[t][o] > T_POW[t][c])
                    T_OUT[t][c] |= UINT64_C(1) << o;
        }
    }
    ready = 1;
}

/* Strongest / weakest card of a non-empty mask within one effective suit. */
static inline int top_of(u64 mask, int trump) {
    if (mask & T_BOWERS[trump])
        return (mask >> T_RIGHT[trump]) & 1 ? T_RIGHT[trump] : T_LEFT[trump];
    return HIGHEST(mask);
}

static inline int bottom_of(u64 mask, int trump) {
    u64 rest = mask & ~T_BOWERS[trump];
    if (rest) return LOWEST(rest);
    return (mask >> T_LEFT[trump]) & 1 ? T_LEFT[trump] : T_RIGHT[trump];
}

/* xorshift64* */
static inline u64 next_u64(u64 *r) {
    u64 x = *r;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *r = x;
    return x * UINT64_C(2685821657736338717);
}

static inline double next_double(u64 *r) {
    return (next_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}

static inline int next_below(u64 *r, int n) {
    return (int)(((next_u64(r) >> 32) * (u64)n) >> 32);
}

static inline int points(const Params *p, int bid, int tricks) {
    if (bid != tricks) {
        int off = bid > tricks ? bid - tricks : tricks - bid;
        return -p->penalty_per_trick * off;
    }
    return bid == 0 ? p->made_zero : p->points_per_trick * bid;
}

/* ------------------------------------------------------------------ */
/* The rule-based policy (heuristic.py).                               */

static inline int is_boss(int card, u64 seen, int trump) {
    return !(T_OUT[trump][card] & ~seen);
}

static int choose_lead(u64 hand, int trump, u64 seen, int want) {
    u64 trumps = hand & T_SUIT[trump][trump];
    if (want) {
        if (trumps) {
            int top = top_of(trumps, trump);
            if (is_boss(top, seen, trump) && POP(trumps) >= 2) return top;
        }
        int best = NO_CARD;
        for (int s = 0; s < 4; s++) {
            u64 cards = hand & T_SUIT[trump][s];
            if (cards && s != trump) {
                int top = top_of(cards, trump);
                if (is_boss(top, seen, trump) && (best == NO_CARD || RANK_OF(top) > RANK_OF(best)))
                    best = top;
            }
        }
        if (best != NO_CARD) return best;
        if (trumps && is_boss(top_of(trumps, trump), seen, trump)) return top_of(trumps, trump);
        u64 shortest = 0;
        for (int s = 0; s < 4; s++) {
            u64 cards = hand & T_SUIT[trump][s];
            if (cards && s != trump && (!shortest || POP(cards) < POP(shortest)))
                shortest = cards;
        }
        return bottom_of(shortest ? shortest : trumps, trump);
    }
    int best = NO_CARD, best_score = -99;
    for (int s = 0; s < 4; s++) {
        u64 cards = hand & T_SUIT[trump][s];
        if (cards) {
            int card = bottom_of(cards, trump);
            int score = POP(T_OUT[trump][card] & ~seen) - (s == trump ? 3 : 0);
            if (score > best_score) {
                best = card;
                best_score = score;
            }
        }
    }
    return best;
}

static int highest_rank(u64 mask, int trump) {
    int best = NO_CARD, best_rank = -1, best_len = 0;
    for (int s = 0; s < 4; s++) {
        u64 cards = mask & T_SUIT[trump][s];
        if (cards) {
            int top = top_of(cards, trump), len = POP(cards);
            if (best == NO_CARD || RANK_OF(top) > best_rank
                || (RANK_OF(top) == best_rank && len < best_len)) {
                best = top;
                best_rank = RANK_OF(top);
                best_len = len;
            }
        }
    }
    return best;
}

static int lowest_rank(u64 mask, int trump) {
    int best = NO_CARD;
    for (int s = 0; s < 4; s++) {
        u64 cards = mask & T_SUIT[trump][s];
        if (cards) {
            int low = bottom_of(cards, trump);
            if (best == NO_CARD || RANK_OF(low) < RANK_OF(best)) best = low;
        }
    }
    return best;
}

static int choose_card(u64 hand, int trump, int lead_suit, int best_card, int trick_size,
                       int n, u64 seen, int need, int behind_wants, int scarce) {
    int want = need > 0;
    if (lead_suit == NO_SUIT) return choose_lead(hand, trump, seen, want);

    int last = trick_size == n - 1;
    u64 following = hand & T_SUIT[trump][lead_suit];
    int best_suit = T_EFF[trump][best_card];

    if (following) {
        u64 winners = best_suit == lead_suit ? following & T_OUT[trump][best_card] : 0;
        u64 losers = following ^ winners;
        if (want) {
            if (!winners) return bottom_of(losers, trump);
            if (last || !behind_wants) return bottom_of(winners, trump);
            int top = top_of(winners, trump);
            if (is_boss(top, seen, trump) || trick_size * 2 >= n || !losers || scarce) return top;
            return bottom_of(losers, trump);
        }
        if (losers) return top_of(losers, trump);
        return last || !behind_wants ? top_of(winners, trump) : bottom_of(winners, trump);
    }

    u64 trumps = hand & T_SUIT[trump][trump];
    u64 sides = hand ^ trumps;
    int trumped = best_suit == trump;

    if (want) {
        if (trumps) {
            if (!trumped) return bottom_of(trumps, trump);
            u64 over = trumps & T_OUT[trump][best_card];
            if (over) return bottom_of(over, trump);
        }
        return sides ? lowest_rank(sides, trump) : bottom_of(trumps, trump);
    }

    if (trumped) {
        u64 under = trumps & ~T_OUT[trump][best_card];
        if (under) return top_of(under, trump);
    }
    if (sides) return highest_rank(sides, trump);
    return last ? top_of(trumps, trump) : bottom_of(trumps, trump);
}

static inline int state_card(const State *st, const Table *t) {
    int seat = st->to_act;
    u64 hand = st->hands[seat];
    int behind_wants = 1, scarce = 0;
    if (t->react) {
        int total = 0, chasing = 0;
        for (int offset = 0; offset < t->n; offset++) {
            int other = (seat + offset) % t->n, short_by = st->bids[other] - st->won[other];
            if (short_by > 0) {
                total += short_by;
                if (offset > 0 && offset < t->n - st->trick_size) chasing = 1;
            }
        }
        behind_wants = chasing;
        scarce = total > POP(hand);
    }
    return choose_card(hand, t->trump_suit, st->lead_suit, st->best_card, st->trick_size, t->n,
                       st->played | hand | t->trump_bit, st->bids[seat] - st->won[seat],
                       behind_wants, scarce);
}

EXPORT int oh_choose_card(u64 hand, int trump, int lead_suit, int best_card, int trick_size,
                          int n, u64 seen, int need, int behind_wants, int scarce) {
    init_tables();
    return choose_card(hand, trump, lead_suit, best_card, trick_size, n, seen, need,
                       behind_wants, scarce);
}

/* ------------------------------------------------------------------ */
/* Rules (engine.py).                                                  */

static inline u64 legal_mask(u64 hand, int lead_suit, int trump) {
    if (lead_suit == NO_SUIT) return hand;
    u64 follow = hand & T_SUIT[trump][lead_suit];
    return follow ? follow : hand;
}

static inline void play_card(State *st, const Table *t, int card) {
    int seat = st->to_act;
    u64 bit = UINT64_C(1) << card;
    int trump = t->trump_suit;
    int suit = T_EFF[trump][card];
    st->hands[seat] ^= bit;
    st->played |= bit;
    st->table |= bit;

    if (st->trick_size == 0) {
        st->lead_suit = suit;
        st->best_card = card;
        st->best_seat = seat;
    } else {
        int best_suit = T_EFF[trump][st->best_card];
        if (suit == best_suit ? T_POW[trump][card] > T_POW[trump][st->best_card]
                              : suit == trump) {
            st->best_card = card;
            st->best_seat = seat;
        }
    }

    st->trick_size++;
    if (st->trick_size < t->n) {
        st->to_act = (seat + 1) % t->n;
        return;
    }
    int winner = st->best_seat;
    st->won[winner]++;
    st->leader = st->to_act = winner;
    st->trick_size = 0;
    st->lead_suit = NO_SUIT;
    st->best_card = NO_CARD;
    st->best_seat = -1;
    st->table = 0;
    if (!st->hands[winner]) st->done = 1;
}

static void playout(State *st, const Table *t, const double *noise, u64 *rng) {
    while (!st->done) {
        int seat = st->to_act;
        int card;
        if (noise[seat] > 0.0 && next_double(rng) < noise[seat]) {
            u64 mask = legal_mask(st->hands[seat], st->lead_suit, t->trump_suit);
            int pick = next_below(rng, POP(mask));
            while (pick--) mask &= mask - 1;
            card = LOWEST(mask);
        } else {
            card = state_card(st, t);
        }
        play_card(st, t, card);
    }
}

/* ------------------------------------------------------------------ */
/* Exact endgame search (search.py), with a small transposition table. */

#define MEMO_BITS 14
#define MEMO_SIZE (1 << MEMO_BITS)

typedef struct {
    u64 key;
    unsigned stamp;
    short value[MAX_SEATS];
} MemoEntry;

static MemoEntry memo[MEMO_SIZE];
static unsigned memo_stamp = 0;

static inline u64 mix64(u64 x) {
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

static inline u64 memo_key(const State *st, const Table *t) {
    u64 key = (u64)st->leader + 1;
    for (int i = 0; i < t->n; i++) {
        key = mix64(key ^ st->hands[i]) + (u64)st->won[i] * UINT64_C(0x9E3779B97F4A7C15);
    }
    return key | 1;
}

static int distinct_moves(const State *st, const Table *t, int *out) {
    int trump = t->trump_suit;
    u64 legal = legal_mask(st->hands[st->to_act], st->lead_suit, trump);
    int count = 0;
    u64 live = st->table;
    for (int i = 0; i < t->n; i++) live |= st->hands[i];
    /* Walk each effective suit weakest first, so the bowers follow the Ace. */
    for (int s = 0; s < 4; s++) {
        u64 mask = legal & T_SUIT[trump][s];
        while (mask) {
            int card = bottom_of(mask, trump);
            mask ^= UINT64_C(1) << card;
            if (count) {
                int prev = out[count - 1];
                u64 above_prev = T_OUT[trump][prev];
                int same_suit = (above_prev >> card) & 1;
                u64 between = above_prev & ~T_OUT[trump][card] & ~(UINT64_C(1) << card);
                if (same_suit && !(between & live)) {
                    out[count - 1] = card;
                    continue;
                }
            }
            out[count++] = card;
        }
    }
    return count;
}

static void solve(const State *st, const Table *t, const Params *p, int root, short *value) {
    if (st->done) {
        for (int i = 0; i < t->n; i++) value[i] = (short)points(p, st->bids[i], st->won[i]);
        return;
    }

    MemoEntry *slot = NULL;
    u64 key = 0;
    if (st->trick_size == 0) {
        key = memo_key(st, t);
        slot = &memo[key >> (64 - MEMO_BITS)];
        if (slot->stamp == memo_stamp && slot->key == key) {
            memcpy(value, slot->value, sizeof(short) * MAX_SEATS);
            return;
        }
    }

    int moves[13];
    int count = distinct_moves(st, t, moves);
    int seat = st->to_act;
    short best[MAX_SEATS], current[MAX_SEATS];
    for (int i = 0; i < count; i++) {
        State child = *st;
        play_card(&child, t, moves[i]);
        solve(&child, t, p, root, current);
        if (i == 0 || current[seat] > best[seat]
            || (current[seat] == best[seat] && current[root] < best[root]))
            memcpy(best, current, sizeof(best));
    }
    memcpy(value, best, sizeof(short) * MAX_SEATS);
    if (slot) {
        slot->key = key;
        slot->stamp = memo_stamp;
        memcpy(slot->value, best, sizeof(best));
    }
}

/* ------------------------------------------------------------------ */
/* Bidding estimates (heuristic.py).                                   */

static double poisson_at_most(int k, double mean) {
    double term = exp(-mean), total = term;
    for (int i = 1; i <= k; i++) {
        term *= mean / i;
        total += term;
    }
    return total;
}

static int card_win_probs(u64 hand, int trump_card, int n, int hand_size, int leads_first,
                          double *probs) {
    int trump = SUIT_OF(trump_card);
    u64 seen = hand | (UINT64_C(1) << trump_card);
    double unseen = 51 - hand_size;
    double miss = 1.0 - (n - 1) * hand_size / unseen;
    if (miss < 0.0) miss = 0.0;
    int trump_held = POP(hand & T_SUIT[trump][trump]);
    int short_sides = 0;
    for (int s = 0; s < 4; s++)
        if (s != trump && POP(hand & T_SUIT[trump][s]) <= 1) short_sides++;
    /* The turned-up card is always a trump: the left bower is never turned. */
    int other_trumps = POP(T_SUIT[trump][trump]) - trump_held - 1;
    double opponent_has_trump = 1.0 - exp(-(hand_size * other_trumps / unseen));
    double reach_any = 1.0 - pow(0.75, hand_size);

    int count = 0;
    for (int suit = 0; suit < 4; suit++) {
        u64 cards = hand & T_SUIT[trump][suit];
        int length = POP(cards), position = 0;
        while (cards) {
            int card = top_of(cards, trump);
            cards ^= UINT64_C(1) << card;
            int higher = POP(T_OUT[trump][card] & ~seen);
            double clear = pow(miss, higher);
            if (suit == trump) {
                double ruff = 0.0;
                if (hand_size >= 3) {
                    ruff = 0.10 * (trump_held - 1) + 0.12 * short_sides;
                    if (ruff > 0.55) ruff = 0.55;
                }
                probs[count++] = clear + (1.0 - clear) * ruff;
            } else {
                int outstanding = POP(T_SUIT[trump][suit]) - length;
                double opponent_length = hand_size * outstanding / unseen;
                double risk = poisson_at_most(position, opponent_length);
                double safe = pow(1.0 - risk * opponent_has_trump, n - 1);
                double reached = (leads_first && position == 0) ? 1.0 : reach_any;
                probs[count++] = clear * safe * reached;
            }
            position++;
        }
    }
    return count;
}

/* Version 2 (estimate.py): exact counting features and fitted logistic weights.
 * tests/test_estimate.py checks this against the Python reference. */
static double LOG_FACT[53];
static int log_fact_ready = 0;

static double log_choose(int n, int k) {
    if (!log_fact_ready) {
        LOG_FACT[0] = 0.0;
        for (int i = 1; i <= 52; i++) LOG_FACT[i] = LOG_FACT[i - 1] + log((double)i);
        log_fact_ready = 1;
    }
    if (k < 0 || k > n) return -INFINITY;
    return LOG_FACT[n] - LOG_FACT[k] - LOG_FACT[n - k];
}

static double hyper_pmf(int population, int successes, int draws, int k) {
    double l = log_choose(successes, k) + log_choose(population - successes, draws - k)
               - log_choose(population, draws);
    return isinf(l) ? 0.0 : exp(l);
}

static double hyper_at_most(int population, int successes, int draws, int k) {
    double total = 0.0;
    int last = k < draws ? k : draws;
    for (int j = 0; j <= last; j++) total += hyper_pmf(population, successes, draws, j);
    return total;
}

static double ruff_chance(int unseen, int in_suit, int trumps, int hand_size, int index) {
    int rest = unseen - in_suit;
    double chance = 0.0;
    int last = index < hand_size ? index : hand_size;
    for (int held = 0; held <= last; held++) {
        double pmf = hyper_pmf(unseen, in_suit, hand_size, held);
        if (pmf == 0.0) continue;
        int others = hand_size - held;
        double no_trump = 0.0;
        if (others <= rest) {
            double l = log_choose(rest - trumps, others) - log_choose(rest, others);
            no_trump = isinf(l) ? 0.0 : exp(l);
        }
        chance += pmf * (1.0 - no_trump);
    }
    return chance;
}

static double clamp_logit(double p) {
    if (p < 1e-4) p = 1e-4;
    if (p > 1.0 - 1e-4) p = 1.0 - 1e-4;
    return log(p / (1.0 - p));
}

/* The counting features depend only on the number of players and the hand
 * size (fixed within a turn) and a few small counts, so they are tabulated
 * once per table shape. Like the other workspaces here, not thread-safe. */
#define EST_COUNTS 15 /* 0..14 cards of one effective suit */
#define EST_INDEX 13
static int est_n = -1, est_h = -1;
static double EST_TOP[EST_COUNTS][EST_INDEX], EST_TOP_LOGIT[EST_COUNTS][EST_INDEX];
static double EST_CLEAR_LOGIT[EST_COUNTS];
static double EST_LONG[EST_COUNTS][EST_INDEX], EST_LONG_LOGIT[EST_COUNTS][EST_INDEX];
static double EST_SAFE[EST_COUNTS][EST_COUNTS][EST_INDEX];
static double EST_LOG_SAFE[EST_COUNTS][EST_COUNTS][EST_INDEX];
static unsigned char EST_SAFE_READY[EST_COUNTS][EST_COUNTS][EST_INDEX];

static void est_tables(int n, int hand_size) {
    if (n == est_n && hand_size == est_h) return;
    int unseen = 51 - hand_size, dealt = (n - 1) * hand_size;
    for (int c = 0; c < EST_COUNTS; c++) {
        EST_CLEAR_LOGIT[c] = clamp_logit(hyper_pmf(unseen, c, dealt, 0));
        for (int k = 0; k < EST_INDEX; k++) {
            double t = hyper_at_most(unseen, c, dealt, k);
            double l = pow(hyper_at_most(unseen, c, hand_size, k), n - 1);
            EST_TOP[c][k] = t;
            EST_TOP_LOGIT[c][k] = clamp_logit(t);
            EST_LONG[c][k] = l;
            EST_LONG_LOGIT[c][k] = clamp_logit(l);
        }
    }
    memset(EST_SAFE_READY, 0, sizeof(EST_SAFE_READY));
    est_n = n;
    est_h = hand_size;
}

static inline int est_safe(int n, int hand_size, int outstanding, int trumps, int index) {
    if (!EST_SAFE_READY[outstanding][trumps][index]) {
        double safe = pow(1.0 - ruff_chance(51 - hand_size, outstanding, trumps, hand_size, index),
                          n - 1);
        EST_SAFE[outstanding][trumps][index] = safe;
        EST_LOG_SAFE[outstanding][trumps][index] = log(safe > 1e-4 ? safe : 1e-4);
        EST_SAFE_READY[outstanding][trumps][index] = 1;
    }
    return 1;
}

static int card_win_probs2(u64 hand, int trump_card, int n, int hand_size, int position,
                           const Params *p, double *probs) {
    est_tables(n, hand_size);
    int trump = SUIT_OF(trump_card);
    u64 seen = hand | (UINT64_C(1) << trump_card);
    int unseen_trumps = POP(T_SUIT[trump][trump] & ~seen);
    int voids = 0, singletons = 0;
    double shortness[4] = {0.0, 0.0, 0.0, 0.0};
    for (int s = 0; s < 4; s++) {
        if (s == trump) continue;
        int length = POP(hand & T_SUIT[trump][s]);
        voids += length == 0;
        singletons += length == 1;
        shortness[s] = (length == 0) + 0.5 * (length == 1);
    }
    double leads = position == 0 ? 1.0 : 0.0;
    double last = n > 1 && position == n - 1 ? 1.0 : 0.0;

    int count = 0;
    int is_trump[13], index_of[13], length_of[13], certain[13];
    double top[13], top_logit[13], clear_logit[13], long_logit[13], log_safe[13];
    double structural[13], short_others[13], total = 0.0;
    for (int suit = 0; suit < 4; suit++) {
        double shorts = 0.0;
        for (int s = 0; s < 4; s++)
            if (s != trump && s != suit) shorts += shortness[s];
        u64 cards = hand & T_SUIT[trump][suit];
        int length = POP(cards), index = 0;
        int outstanding = POP(T_SUIT[trump][suit] & ~seen);
        while (cards) {
            int card = top_of(cards, trump);
            cards ^= UINT64_C(1) << card;
            int higher = POP(T_OUT[trump][card] & ~seen);
            double t = EST_TOP[higher][index];
            double l = EST_LONG[outstanding][index];
            double best = t > l ? t : l;
            is_trump[count] = suit == trump;
            short_others[count] = shorts;
            index_of[count] = index;
            length_of[count] = length;
            certain[count] = suit == trump && higher == 0;
            top[count] = t;
            top_logit[count] = EST_TOP_LOGIT[higher][index];
            clear_logit[count] = EST_CLEAR_LOGIT[higher];
            long_logit[count] = EST_LONG_LOGIT[outstanding][index];
            if (suit == trump) {
                log_safe[count] = 0.0;
                structural[count] = best;
            } else {
                est_safe(n, hand_size, outstanding, unseen_trumps, index);
                log_safe[count] = EST_LOG_SAFE[outstanding][unseen_trumps][index];
                structural[count] = best * EST_SAFE[outstanding][unseen_trumps][index];
            }
            total += structural[count];
            count++;
            index++;
        }
    }

    for (int i = 0; i < count; i++) {
        if (certain[i]) {
            probs[i] = 1.0;
            continue;
        }
        double first = index_of[i] == 0 ? 1.0 : 0.0;
        double others = total - structural[i];
        double f[EST_SIDE > EST_TRUMP ? EST_SIDE : EST_TRUMP];
        f[0] = 1.0;
        f[1] = top_logit[i];
        f[2] = clear_logit[i];
        f[3] = long_logit[i];
        f[4] = leads;
        f[5] = leads * first;
        f[6] = others;
        f[7] = others * first;
        f[8] = (double)position / (n > 1 ? n - 1 : 1);
        f[9] = index_of[i];
        f[10] = length_of[i];
        f[11] = hand_size / 10.0;
        f[12] = n / 8.0;
        f[13] = hand_size == 1;
        f[14] = leads * first * top[i];
        f[15] = f[8] * top[i];
        f[16] = last;
        f[17] = last * top[i];
        int width, special;
        const double *w;
        if (is_trump[i]) {
            f[18] = voids;
            f[19] = singletons;
            f[20] = (voids + 0.5 * singletons) * (1.0 - top[i]);
            special = 20;
            width = 21;
            w = p->est_trump;
        } else {
            f[18] = log_safe[i];
            f[19] = log_safe[i] * first;
            special = 18;
            width = 20;
            w = p->est_side;
        }
        /* Interactions with estimate.SCALES (hand, one_card, players) of the
         * features in estimate.SCALED, then ruff_low or log_safe. */
        static const int scaled[11] = {1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17};
        static const int scales[3] = {11, 13, 12};
        int k = width;
        for (int g = 0; g < 3; g++) {
            double scale = f[scales[g]];
            for (int j = 0; j < 11; j++) f[k++] = f[scaled[j]] * scale;
            f[k++] = f[special] * scale;
        }
        if (!is_trump[i]) {  /* estimate.DISCARDS */
            f[k++] = short_others[i];
            f[k++] = short_others[i] * top[i];
            f[k++] = short_others[i] * top[i] * f[12];
            f[k++] = last * first;
            f[k++] = first * f[12];
            f[k++] = last * first * f[11];
            f[k++] = first * f[12] * f[11];
            f[k++] = last * f[1];
            f[k++] = last * f[1] * f[11];
        }
        double z = 0.0;
        for (int j = 0; j < k; j++) z += w[j] * f[j];
        if (z > 30.0) z = 30.0;
        if (z < -30.0) z = -30.0;
        probs[i] = 1.0 / (1.0 + exp(-z));
    }
    return count;
}

/* The estimate this simulation uses for a seat's bid. */
static int bid_estimate_probs(u64 hand, int trump_card, int n, int hand_size, int seat,
                              int dealer, const Params *p, double *probs) {
    int position = ((seat - dealer - 1) % n + n) % n;
    if (p->estimate >= 2)
        return card_win_probs2(hand, trump_card, n, hand_size, position, p, probs);
    return card_win_probs(hand, trump_card, n, hand_size, position == 0, probs);
}

EXPORT double oh_expected_tricks2(u64 hand, int trump_card, int n, int hand_size, int position,
                                  const Params *p, double *probs) {
    init_tables();
    double total = 0.0;
    int count = card_win_probs2(hand, trump_card, n, hand_size, position, p, probs);
    for (int i = 0; i < count; i++) total += probs[i];
    return total;
}

EXPORT double oh_expected_tricks(u64 hand, int trump_card, int n, int hand_size, int leads_first) {
    init_tables();
    double probs[13], total = 0.0;
    int count = card_win_probs(hand, trump_card, n, hand_size, leads_first, probs);
    for (int i = 0; i < count; i++) total += probs[i];
    return total;
}

static int heuristic_bid(u64 hand, int trump_card, int n, int hand_size, int seat, int dealer,
                         const Params *p) {
    double probs[13], dist[15] = {1.0}, grown[15];
    int count = bid_estimate_probs(hand, trump_card, n, hand_size, seat, dealer, p, probs);
    for (int i = 0; i < count; i++) {
        memset(grown, 0, sizeof(grown));
        for (int k = 0; k <= i; k++) {
            grown[k] += dist[k] * (1.0 - probs[i]);
            grown[k + 1] += dist[k] * probs[i];
        }
        memcpy(dist, grown, sizeof(dist));
    }
    if (p->estimate >= 2) {  /* estimate.widen */
        double temperature = p->est_spread[0] + p->est_spread[1] * hand_size / 10.0
                             + p->est_spread[2] * n / 8.0;
        if (temperature < 0.2) temperature = 0.2;
        if (temperature != 1.0) {
            double sum = 0.0;
            for (int k = 0; k <= count; k++) {
                dist[k] = dist[k] > 0.0 ? pow(dist[k], 1.0 / temperature) : 0.0;
                sum += dist[k];
            }
            for (int k = 0; k <= count; k++) dist[k] /= sum;
        }
    }
    int best_bid = 0;
    double best_value = -1e18;
    for (int bid = 0; bid <= hand_size; bid++) {
        double value = 0.0;
        for (int k = 0; k <= count; k++) value += dist[k] * points(p, bid, k);
        if (value > best_value) {
            best_value = value;
            best_bid = bid;
        }
    }
    return best_bid;
}

/* The bid simulations imagine for a seat yet to bid, before its kind's noise. */
EXPORT int oh_heuristic_bid(u64 hand, int trump_card, int n, int hand_size, int seat,
                            int dealer, const Params *p) {
    init_tables();
    return heuristic_bid(hand, trump_card, n, hand_size, seat, dealer, p);
}

static int noisy_bid(int bid, const Kind *kind, int hand_size, u64 *rng) {
    double roll = next_double(rng);
    int size;
    if (roll < kind->bid_two) size = 2;
    else if (roll < kind->bid_two + kind->bid_one) size = 1;
    else return bid;
    int moved = bid + ((next_u64(rng) >> 63) ? size : -size);
    return (moved >= 0 && moved <= hand_size) ? moved : bid;
}

/* ------------------------------------------------------------------ */
/* Guessing the hidden cards (belief.py).                              */

#define DEAD_BITS 16
#define DEAD_SIZE (1 << DEAD_BITS)
static u64 dead_table[DEAD_SIZE];

static int place_exactly(const int *cards, int count, int index, int bins_count, int *room,
                         const int *voids, const int *suit_of, u64 *bins, u64 *rng) {
    if (index == count) return 1;
    u64 key = (u64)index + 1;
    for (int i = 0; i < bins_count; i++) key = (key << 6) | (u64)room[i];
    key = mix64(key) | 1;
    u64 *slot = &dead_table[key >> (64 - DEAD_BITS)];
    if (*slot == key) return 0;

    int order[MAX_SEATS + 1];
    for (int i = 0; i < bins_count; i++) order[i] = i;
    for (int i = bins_count - 1; i > 0; i--) {
        int j = next_below(rng, i + 1), tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
    int suit_bit = 1 << suit_of[cards[index]];
    for (int o = 0; o < bins_count; o++) {
        int i = order[o];
        if (room[i] && !(voids[i] & suit_bit)) {
            room[i]--;
            bins[i] |= UINT64_C(1) << cards[index];
            if (place_exactly(cards, count, index + 1, bins_count, room, voids, suit_of, bins, rng))
                return 1;
            room[i]++;
            bins[i] ^= UINT64_C(1) << cards[index];
        }
    }
    *slot = key;
    return 0;
}

static int sample_hands(const Obs *obs, u64 *hands, u64 *rng) {
    const int *suit_of = T_EFF[SUIT_OF(obs->trump_card)];
    u64 known = obs->hand | obs->played | (UINT64_C(1) << obs->trump_card);
    u64 unknown_mask = FULL_DECK ^ known;
    int unknown[52], unknown_count = 0;
    while (unknown_mask) {
        unknown[unknown_count++] = LOWEST(unknown_mask);
        unknown_mask &= unknown_mask - 1;
    }

    int held[MAX_SEATS];
    for (int i = 0; i < obs->n; i++) held[i] = obs->hand_size;
    for (int i = 0; i < obs->history_len; i++) held[obs->history_seat[i]]--;

    int others[MAX_SEATS], capacity[MAX_SEATS + 1], voids[MAX_SEATS + 1];
    int bins_count = 0, assigned = 0;
    for (int s = 0; s < obs->n; s++) {
        if (s != obs->seat) {
            others[bins_count] = s;
            capacity[bins_count] = held[s];
            voids[bins_count] = obs->voids[s];
            assigned += held[s];
            bins_count++;
        }
    }
    capacity[bins_count] = unknown_count - assigned;
    voids[bins_count] = 0;
    bins_count++;

    int pressure[4] = {0, 0, 0, 0};
    for (int s = 0; s < 4; s++)
        for (int b = 0; b < bins_count; b++) pressure[s] += (voids[b] >> s) & 1;

    u64 bins[MAX_SEATS + 1];
    int ordered[52], done = 0;
    for (int attempt = 0; attempt < 40 && !done; attempt++) {
        for (int i = unknown_count - 1; i > 0; i--) {
            int j = next_below(rng, i + 1), tmp = unknown[i];
            unknown[i] = unknown[j];
            unknown[j] = tmp;
        }
        int fill = 0;
        for (int level = bins_count; level >= 0; level--)
            for (int i = 0; i < unknown_count; i++)
                if (pressure[suit_of[unknown[i]]] == level) ordered[fill++] = unknown[i];

        int room[MAX_SEATS + 1];
        memcpy(room, capacity, sizeof(int) * bins_count);
        memset(bins, 0, sizeof(u64) * bins_count);
        done = 1;
        for (int i = 0; i < unknown_count; i++) {
            int suit_bit = 1 << suit_of[ordered[i]];
            int total = 0;
            for (int b = 0; b < bins_count; b++)
                if (room[b] && !(voids[b] & suit_bit)) total += room[b];
            if (!total) {
                done = 0;
                break;
            }
            int pick = next_below(rng, total);
            for (int b = 0; b < bins_count; b++) {
                if (room[b] && !(voids[b] & suit_bit)) {
                    pick -= room[b];
                    if (pick < 0) {
                        bins[b] |= UINT64_C(1) << ordered[i];
                        room[b]--;
                        break;
                    }
                }
            }
        }
    }
    if (!done) {
        int room[MAX_SEATS + 1];
        memcpy(room, capacity, sizeof(int) * bins_count);
        memset(bins, 0, sizeof(u64) * bins_count);
        memset(dead_table, 0, sizeof(dead_table));
        if (!place_exactly(ordered, unknown_count, 0, bins_count, room, voids, suit_of, bins, rng))
            return 0;
    }

    for (int i = 0; i < obs->n; i++) hands[i] = 0;
    hands[obs->seat] = obs->hand;
    for (int i = 0; i < obs->n - 1; i++) hands[others[i]] = bins[i];
    return 1;
}

static int draw_kind(const Params *p, u64 *rng) {
    double roll = next_double(rng), total = 0.0;
    for (int i = 0; i < MAX_KINDS; i++) total += p->weights[i];
    roll *= total;
    for (int i = 0; i < MAX_KINDS; i++) {
        if (roll < p->weights[i]) return i;
        roll -= p->weights[i];
    }
    return 0;
}

static double play_log_weight(const Obs *obs, const Params *p, const Table *t,
                              const u64 *opening, const int *kinds) {
    State st;
    memset(&st, 0, sizeof(st));
    memcpy(st.hands, opening, sizeof(u64) * obs->n);
    memcpy(st.bids, obs->bids, sizeof(int) * obs->n);
    st.lead_suit = NO_SUIT;
    st.best_card = NO_CARD;
    st.best_seat = -1;
    st.leader = st.to_act = (obs->dealer + 1) % obs->n;

    double total = 0.0;
    for (int i = 0; i < obs->history_len; i++) {
        int seat = obs->history_seat[i], card = obs->history_card[i];
        if (seat != obs->seat) {
            int options = POP(legal_mask(st.hands[seat], st.lead_suit, t->trump_suit));
            if (options > 1) {
                double doubt = p->kinds[kinds[seat]].play_doubt;
                double chance = doubt / options;
                if (doubt < 1.0 && state_card(&st, t) == card) chance += 1.0 - doubt;
                total += log(chance);
            }
        }
        play_card(&st, t, card);
    }
    return total;
}

static double log_weight(const Obs *obs, const Params *p, const Table *t, const World *w) {
    u64 opening[MAX_SEATS];
    memcpy(opening, w->hands, sizeof(u64) * obs->n);
    for (int i = 0; i < obs->history_len; i++)
        opening[obs->history_seat[i]] |= UINT64_C(1) << obs->history_card[i];

    double total = 0.0;
    for (int s = 0; s < obs->n; s++) {
        if (s != obs->seat && obs->bids[s] != NO_BID) {
            double probs[13], estimate = 0.0;
            int count = bid_estimate_probs(opening[s], obs->trump_card, obs->n, obs->hand_size,
                                           s, obs->dealer, p, probs);
            for (int i = 0; i < count; i++) estimate += probs[i];
            double sigma = p->kinds[w->kinds[s]].bid_sigma;
            double error = obs->bids[s] - estimate;
            total -= error * error / (2.0 * sigma * sigma) + log(sigma);
        }
    }
    if (obs->phase == PLAYING && obs->history_len)
        total += play_log_weight(obs, p, t, opening, w->kinds);
    return total;
}

#define MAX_CANDIDATES 4096
static World candidates[MAX_CANDIDATES];
static double cumulative[MAX_CANDIDATES];

static int draw_world(const Obs *obs, const Params *p, World *w, u64 *rng) {
    if (!sample_hands(obs, w->hands, rng)) return 0;
    for (int s = 0; s < obs->n; s++) w->kinds[s] = draw_kind(p, rng);
    w->kinds[obs->seat] = 0; /* slot 0 is the skilled kind: no card noise */
    return 1;
}

static int make_worlds(const Obs *obs, const Params *p, const Table *t, World *out, int count,
                       u64 *rng) {
    int evidence = 0;
    for (int s = 0; s < obs->n; s++)
        if (s != obs->seat && obs->bids[s] != NO_BID) evidence = 1;

    if (!p->inference || !evidence) {
        for (int i = 0; i < count; i++)
            if (!draw_world(obs, p, &out[i], rng)) return 0;
        return 1;
    }

    int total = count * p->oversample;
    if (total > MAX_CANDIDATES) total = MAX_CANDIDATES;
    double logs[MAX_CANDIDATES], top = -1e300;
    for (int i = 0; i < total; i++) {
        if (!draw_world(obs, p, &candidates[i], rng)) return 0;
        logs[i] = log_weight(obs, p, t, &candidates[i]);
        if (logs[i] > top) top = logs[i];
    }
    double running = 0.0;
    for (int i = 0; i < total; i++) {
        running += exp(logs[i] - top);
        cumulative[i] = running;
    }
    for (int i = 0; i < count; i++) {
        double roll = next_double(rng) * running;
        int low = 0, high = total - 1;
        while (low < high) {
            int mid = (low + high) / 2;
            if (cumulative[mid] <= roll) low = mid + 1;
            else high = mid;
        }
        out[i] = candidates[low];
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* What simulation.py asks for.                                        */

static State state_from_obs(const Obs *obs, const u64 *hands) {
    State st;
    memset(&st, 0, sizeof(st));
    memcpy(st.hands, hands, sizeof(u64) * obs->n);
    memcpy(st.bids, obs->bids, sizeof(int) * obs->n);
    memcpy(st.won, obs->tricks_won, sizeof(int) * obs->n);
    st.played = obs->played;
    st.table = obs->table;
    st.to_act = obs->to_act;
    st.lead_suit = obs->lead_suit;
    st.best_card = obs->best_card;
    st.best_seat = obs->best_seat;
    st.trick_size = obs->trick_size;
    st.leader = obs->leader;
    return st;
}

#define MAX_WORLDS 1024
static World worlds_buffer[MAX_WORLDS];

/* Fills means[i] with the average points after cards[i]. Returns 0 on failure. */
EXPORT int oh_evaluate_cards(const Obs *obs, const Params *p, const int *cards, int ncards,
                             double *means, u64 *rng) {
    init_tables();
    Table t = {obs->n, SUIT_OF(obs->trump_card), UINT64_C(1) << obs->trump_card, p->react};
    int count = p->worlds > MAX_WORLDS ? MAX_WORLDS : p->worlds;
    double stop = p->time_limit > 0.0 ? now_seconds() + p->time_limit : 0.0;
    int remaining = POP(obs->hand) * obs->n - obs->trick_size;
    int exact = remaining <= p->endgame_cards;
    long done = 0;
    for (int c = 0; c < ncards; c++) means[c] = 0.0;

    do {
        if (!make_worlds(obs, p, &t, worlds_buffer, count, rng)) return 0;
        for (int w = 0; w < count; w++) {
            const World *world = &worlds_buffer[w];
            State base = state_from_obs(obs, world->hands);
            double noise[MAX_SEATS];
            int noisy = 0;
            for (int s = 0; s < obs->n; s++) {
                noise[s] = p->kinds[world->kinds[s]].card_noise;
                if (noise[s] > 0.0) noisy = 1;
            }
            if (exact && !noisy) {
                short value[MAX_SEATS];
                if (++memo_stamp == 0) {
                    memset(memo, 0, sizeof(memo));
                    memo_stamp = 1;
                }
                for (int c = 0; c < ncards; c++) {
                    State child = base;
                    play_card(&child, &t, cards[c]);
                    solve(&child, &t, p, obs->seat, value);
                    means[c] += value[obs->seat];
                }
            } else {
                for (int c = 0; c < ncards; c++) {
                    State child = base;
                    play_card(&child, &t, cards[c]);
                    playout(&child, &t, noise, rng);
                    means[c] += points(p, child.bids[obs->seat], child.won[obs->seat]);
                }
            }
            done++;
            if (stop > 0.0 && now_seconds() >= stop) goto finished;
        }
    } while (stop > 0.0);

finished:
    for (int c = 0; c < ncards; c++) means[c] /= (double)done;
    return 1;
}

/* Fills counts[b * (hand_size + 1) + k] with deals where options[b] took k tricks. */
EXPORT int oh_analyse_bids(const Obs *obs, const Params *p, int dealer_restriction,
                           const int *options, int noptions, double *counts, u64 *rng) {
    init_tables();
    Table t = {obs->n, SUIT_OF(obs->trump_card), UINT64_C(1) << obs->trump_card, p->react};
    int count = p->worlds > MAX_WORLDS ? MAX_WORLDS : p->worlds;
    double stop = p->time_limit > 0.0 ? now_seconds() + p->time_limit : 0.0;
    int width = obs->hand_size + 1;
    int first = (obs->dealer + 1) % obs->n;
    int done = 0;
    memset(counts, 0, sizeof(double) * noptions * width);

    do {
        if (!make_worlds(obs, p, &t, worlds_buffer, count, rng)) return 0;
        for (int w = 0; w < count; w++) {
            const World *world = &worlds_buffer[w];
            State base = state_from_obs(obs, world->hands);
            base.leader = base.to_act = first;
            double noise[MAX_SEATS];
            for (int s = 0; s < obs->n; s++) {
                noise[s] = p->kinds[world->kinds[s]].card_noise;
                if (obs->bids[s] == NO_BID && s != obs->seat) {
                    int bid = heuristic_bid(world->hands[s], obs->trump_card, obs->n,
                                            obs->hand_size, s, obs->dealer, p);
                    base.bids[s] = noisy_bid(bid, &p->kinds[world->kinds[s]], obs->hand_size, rng);
                }
            }
            for (int o = 0; o < noptions; o++) {
                State child = base;
                child.bids[obs->seat] = options[o];
                if (dealer_restriction && obs->seat != obs->dealer
                        && obs->bids[obs->dealer] == NO_BID) {
                    int sum = 0;
                    for (int s = 0; s < obs->n; s++) sum += child.bids[s];
                    if (sum == obs->hand_size)
                        child.bids[obs->dealer] += child.bids[obs->dealer] == 0 ? 1 : -1;
                }
                playout(&child, &t, noise, rng);
                counts[o * width + child.won[obs->seat]] += 1.0;
            }
            done++;
            if (stop > 0.0 && now_seconds() >= stop) return done;
        }
    } while (stop > 0.0);
    return done;
}

/* Draws count worlds (at most MAX_WORLDS) as analyse_bids would, writing
 * hands[w * MAX_SEATS + s] and kinds[w * MAX_SEATS + s]. Returns the count,
 * or 0 when no deal fits. Used by bots/seat_bids.py, which picks the unbid
 * seats' bids itself before calling oh_score_bids. */
EXPORT int oh_sample_worlds(const Obs *obs, const Params *p, int count, u64 *hands,
                            int *kinds, u64 *rng) {
    init_tables();
    Table t = {obs->n, SUIT_OF(obs->trump_card), UINT64_C(1) << obs->trump_card, p->react};
    if (count > MAX_WORLDS) count = MAX_WORLDS;
    if (count < 1 || !make_worlds(obs, p, &t, worlds_buffer, count, rng)) return 0;
    for (int w = 0; w < count; w++)
        for (int s = 0; s < MAX_SEATS; s++) {
            hands[w * MAX_SEATS + s] = s < obs->n ? worlds_buffer[w].hands[s] : 0;
            kinds[w * MAX_SEATS + s] = s < obs->n ? worlds_buffer[w].kinds[s] : 0;
        }
    return count;
}

/* Plays out each world after each option, with every seat's bid given in
 * bids[(w * noptions + o) * MAX_SEATS + s], and adds to
 * counts[o * (hand_size + 1) + k] the deals where this seat took k tricks. */
EXPORT int oh_score_bids(const Obs *obs, const Params *p, int noptions, int count,
                         const u64 *hands, const int *kinds, const int *bids,
                         double *counts, u64 *rng) {
    init_tables();
    Table t = {obs->n, SUIT_OF(obs->trump_card), UINT64_C(1) << obs->trump_card, p->react};
    int width = obs->hand_size + 1;
    int first = (obs->dealer + 1) % obs->n;
    for (int w = 0; w < count; w++) {
        State base = state_from_obs(obs, hands + w * MAX_SEATS);
        base.leader = base.to_act = first;
        double noise[MAX_SEATS];
        for (int s = 0; s < obs->n; s++)
            noise[s] = p->kinds[kinds[w * MAX_SEATS + s]].card_noise;
        for (int o = 0; o < noptions; o++) {
            State child = base;
            const int *row = bids + (w * noptions + o) * MAX_SEATS;
            for (int s = 0; s < obs->n; s++) child.bids[s] = row[s];
            playout(&child, &t, noise, rng);
            counts[o * width + child.won[obs->seat]] += 1.0;
        }
    }
    return count;
}

/* ------------------------------------------------------------------ */
/* Information-set policy search (bots/planning.py).                    */

#define MAX_TREE_NODES 32768
typedef struct {
    int first_child, sibling, card, visits;
    double total;
} HistoryNode;

/* Like the other native workspaces, this is process-local, not thread-safe. */
static HistoryNode history_tree[MAX_TREE_NODES];
static int history_used;

static void history_init(int index, int card, int sibling) {
    HistoryNode *node = &history_tree[index];
    node->first_child = -1;
    node->sibling = sibling;
    node->card = card;
    node->visits = 0;
    node->total = 0.0;
}

static int history_child(int parent, int card) {
    for (int i = history_tree[parent].first_child; i >= 0; i = history_tree[i].sibling)
        if (history_tree[i].card == card) return i;
    if (history_used == MAX_TREE_NODES) return -1;
    int index = history_used++;
    history_init(index, card, history_tree[parent].first_child);
    history_tree[parent].first_child = index;
    return index;
}

static int history_select(int parent, const int *cards, int count, int preferred,
                           double exploration) {
    int children[52], first_untried = -1, visits = 0;
    for (int c = 0; c < count; c++) {
        int index = history_child(parent, cards[c]);
        if (index < 0) return -1;
        children[c] = index;
        visits += history_tree[index].visits;
        if (!history_tree[index].visits) {
            if (first_untried < 0 || cards[c] == preferred) first_untried = index;
        }
    }
    if (first_untried >= 0) return first_untried;
    double log_n = log(1.0 + visits), best = -1e300;
    int chosen = -1;
    for (int c = 0; c < count; c++) {
        HistoryNode *child = &history_tree[children[c]];
        double value = child->total / child->visits
                     + exploration * sqrt(log_n / child->visits);
        if (value > best) {
            best = value;
            chosen = children[c];
        }
    }
    return chosen;
}

static int population_card(const State *st, const Table *t, const double *noise, u64 *rng) {
    if (noise[st->to_act] > 0.0 && next_double(rng) < noise[st->to_act]) {
        u64 mask = legal_mask(st->hands[st->to_act], st->lead_suit, t->trump_suit);
        int pick = next_below(rng, POP(mask));
        while (pick--) mask &= mask - 1;
        return LOWEST(mask);
    }
    return state_card(st, t);
}

/* Returns completed simulations. Root statistics contain no model pseudo-rewards.
 * A path encodes only played cards. Concealed cards and sampled kinds never
 * choose a tree node, so an identical visible history shares future choices.
 */
EXPORT int oh_plan_cards(const Obs *obs, const Params *p, const int *cards, int ncards,
                          double exploration, int batch, double *totals, int *visits,
                          int *diagnostics, u64 *rng) {
    init_tables();
    Table t = {obs->n, SUIT_OF(obs->trump_card), UINT64_C(1) << obs->trump_card, p->react};
    double stop = p->time_limit > 0.0 ? now_seconds() + p->time_limit : 0.0;
    int done = 0, deep_visits = 0;
    history_used = 1;
    history_init(0, -1, -1);
    do {
        int count = batch;
        if (stop == 0.0 && count > p->worlds - done) count = p->worlds - done;
        if (count <= 0) break;
        if (!make_worlds(obs, p, &t, worlds_buffer, count, rng)) return 0;
        for (int w = 0; w < count; w++) {
            State state = state_from_obs(obs, worlds_buffer[w].hands);
            double noise[MAX_SEATS];
            for (int s = 0; s < obs->n; s++)
                noise[s] = p->kinds[worlds_buffer[w].kinds[s]].card_noise;
            int node = 0, path[52], depth = 0;
            while (!state.done) {
                int card, next;
                if (state.to_act == obs->seat) {
                    int legal[52], count_legal = 0;
                    if (node == 0) {
                        memcpy(legal, cards, sizeof(int) * ncards);
                        count_legal = ncards;
                    } else {
                        u64 mask = legal_mask(state.hands[obs->seat], state.lead_suit, t.trump_suit);
                        while (mask) {
                            legal[count_legal++] = LOWEST(mask);
                            mask &= mask - 1;
                        }
                    }
                    if (count_legal > 1) {
                        int preferred = node == 0 ? cards[0] : state_card(&state, &t);
                        next = history_select(node, legal, count_legal, preferred, exploration);
                        if (next < 0) { playout(&state, &t, noise, rng); break; }
                        int fresh = history_tree[next].visits == 0;
                        path[depth++] = next;
                        if (node != 0) deep_visits++;
                        play_card(&state, &t, history_tree[next].card);
                        node = next;
                        if (fresh) { playout(&state, &t, noise, rng); break; }
                        continue;
                    }
                    card = legal[0];
                } else {
                    card = population_card(&state, &t, noise, rng);
                }
                next = history_child(node, card);
                play_card(&state, &t, card);
                if (next < 0) { playout(&state, &t, noise, rng); break; }
                node = next;
            }
            double reward = points(p, state.bids[obs->seat], state.won[obs->seat]);
            for (int d = 0; d < depth; d++) {
                history_tree[path[d]].visits++;
                history_tree[path[d]].total += reward;
            }
            done++;
            if (stop > 0.0 && now_seconds() >= stop) goto tree_finished;
        }
    } while (stop > 0.0 || done < p->worlds);
tree_finished:
    for (int c = 0; c < ncards; c++) {
        int index = history_child(0, cards[c]);
        totals[c] = index >= 0 ? history_tree[index].total : 0.0;
        visits[c] = index >= 0 ? history_tree[index].visits : 0;
    }
    diagnostics[0] = history_used;
    diagnostics[1] = deep_visits;
    return done;
}
