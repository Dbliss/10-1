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
    ready = 1;
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

static inline int is_boss(int card, u64 seen) { return !(ABOVE[card] & ~seen); }

static int choose_lead(u64 hand, int trump, u64 seen, int want) {
    u64 trumps = hand & SUIT_MASK[trump];
    if (want) {
        if (trumps) {
            int top = HIGHEST(trumps);
            if (is_boss(top, seen) && POP(trumps) >= 2) return top;
        }
        int best = NO_CARD;
        for (int s = 0; s < 4; s++) {
            u64 cards = hand & SUIT_MASK[s];
            if (cards && s != trump) {
                int top = HIGHEST(cards);
                if (is_boss(top, seen) && (best == NO_CARD || RANK_OF(top) > RANK_OF(best)))
                    best = top;
            }
        }
        if (best != NO_CARD) return best;
        if (trumps && is_boss(HIGHEST(trumps), seen)) return HIGHEST(trumps);
        u64 shortest = 0;
        for (int s = 0; s < 4; s++) {
            u64 cards = hand & SUIT_MASK[s];
            if (cards && s != trump && (!shortest || POP(cards) < POP(shortest)))
                shortest = cards;
        }
        return LOWEST(shortest ? shortest : trumps);
    }
    int best = NO_CARD, best_score = -99;
    for (int s = 0; s < 4; s++) {
        u64 cards = hand & SUIT_MASK[s];
        if (cards) {
            int card = LOWEST(cards);
            int score = POP(ABOVE[card] & ~seen) - (s == trump ? 3 : 0);
            if (score > best_score) {
                best = card;
                best_score = score;
            }
        }
    }
    return best;
}

static int highest_rank(u64 mask) {
    int best = NO_CARD, best_rank = -1, best_len = 0;
    for (int s = 0; s < 4; s++) {
        u64 cards = mask & SUIT_MASK[s];
        if (cards) {
            int top = HIGHEST(cards), len = POP(cards);
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

static int lowest_rank(u64 mask) {
    int best = NO_CARD;
    for (int s = 0; s < 4; s++) {
        u64 cards = mask & SUIT_MASK[s];
        if (cards) {
            int low = LOWEST(cards);
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
    u64 following = hand & SUIT_MASK[lead_suit];
    int best_suit = SUIT_OF(best_card);

    if (following) {
        u64 winners = best_suit == lead_suit ? following & ABOVE[best_card] : 0;
        u64 losers = following ^ winners;
        if (want) {
            if (!winners) return LOWEST(losers);
            if (last || !behind_wants) return LOWEST(winners);
            int top = HIGHEST(winners);
            if (is_boss(top, seen) || trick_size * 2 >= n || !losers || scarce) return top;
            return LOWEST(losers);
        }
        if (losers) return HIGHEST(losers);
        return last || !behind_wants ? HIGHEST(winners) : LOWEST(winners);
    }

    u64 trumps = hand & SUIT_MASK[trump];
    u64 sides = hand ^ trumps;
    int trumped = best_suit == trump;

    if (want) {
        if (trumps) {
            if (!trumped) return LOWEST(trumps);
            u64 over = trumps & ABOVE[best_card];
            if (over) return LOWEST(over);
        }
        return sides ? lowest_rank(sides) : LOWEST(trumps);
    }

    if (trumped) {
        u64 under = trumps & ~ABOVE[best_card];
        if (under) return HIGHEST(under);
    }
    if (sides) return highest_rank(sides);
    return last ? HIGHEST(trumps) : LOWEST(trumps);
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

static inline u64 legal_mask(u64 hand, int lead_suit) {
    if (lead_suit == NO_SUIT) return hand;
    u64 follow = hand & SUIT_MASK[lead_suit];
    return follow ? follow : hand;
}

static inline void play_card(State *st, const Table *t, int card) {
    int seat = st->to_act;
    u64 bit = UINT64_C(1) << card;
    int suit = SUIT_OF(card);
    st->hands[seat] ^= bit;
    st->played |= bit;
    st->table |= bit;

    if (st->trick_size == 0) {
        st->lead_suit = suit;
        st->best_card = card;
        st->best_seat = seat;
    } else {
        int best_suit = SUIT_OF(st->best_card);
        if (suit == best_suit ? card > st->best_card : suit == t->trump_suit) {
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
            u64 mask = legal_mask(st->hands[seat], st->lead_suit);
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
    u64 mask = legal_mask(st->hands[st->to_act], st->lead_suit);
    int count = 0;
    u64 live = st->table;
    for (int i = 0; i < t->n; i++) live |= st->hands[i];
    while (mask) {
        int card = LOWEST(mask);
        mask &= mask - 1;
        if (count) {
            int prev = out[count - 1];
            u64 above_prev = ABOVE[prev];
            int same_suit = (above_prev >> card) & 1;
            u64 between = above_prev & ~ABOVE[card] & ~(UINT64_C(1) << card);
            if (same_suit && !(between & live)) {
                out[count - 1] = card;
                continue;
            }
        }
        out[count++] = card;
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
    int trump_held = POP(hand & SUIT_MASK[trump]);
    int short_sides = 0;
    for (int s = 0; s < 4; s++)
        if (s != trump && POP(hand & SUIT_MASK[s]) <= 1) short_sides++;
    int other_trumps = 13 - trump_held - 1;
    double opponent_has_trump = 1.0 - exp(-(hand_size * other_trumps / unseen));
    double reach_any = 1.0 - pow(0.75, hand_size);

    int count = 0;
    for (int suit = 0; suit < 4; suit++) {
        u64 cards = hand & SUIT_MASK[suit];
        int length = POP(cards), position = 0;
        while (cards) {
            int card = HIGHEST(cards);
            cards ^= UINT64_C(1) << card;
            int higher = POP(ABOVE[card] & ~seen);
            double clear = pow(miss, higher);
            if (suit == trump) {
                double ruff = 0.0;
                if (hand_size >= 3) {
                    ruff = 0.10 * (trump_held - 1) + 0.12 * short_sides;
                    if (ruff > 0.55) ruff = 0.55;
                }
                probs[count++] = clear + (1.0 - clear) * ruff;
            } else {
                int outstanding = 13 - length - (SUIT_OF(trump_card) == suit);
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

EXPORT double oh_expected_tricks(u64 hand, int trump_card, int n, int hand_size, int leads_first) {
    init_tables();
    double probs[13], total = 0.0;
    int count = card_win_probs(hand, trump_card, n, hand_size, leads_first, probs);
    for (int i = 0; i < count; i++) total += probs[i];
    return total;
}

static int heuristic_bid(u64 hand, int trump_card, int n, int hand_size, int leads_first,
                         const Params *p) {
    double probs[13], dist[15] = {1.0}, grown[15];
    int count = card_win_probs(hand, trump_card, n, hand_size, leads_first, probs);
    for (int i = 0; i < count; i++) {
        memset(grown, 0, sizeof(grown));
        for (int k = 0; k <= i; k++) {
            grown[k] += dist[k] * (1.0 - probs[i]);
            grown[k + 1] += dist[k] * probs[i];
        }
        memcpy(dist, grown, sizeof(dist));
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
                         const int *voids, u64 *bins, u64 *rng) {
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
    int suit_bit = 1 << SUIT_OF(cards[index]);
    for (int o = 0; o < bins_count; o++) {
        int i = order[o];
        if (room[i] && !(voids[i] & suit_bit)) {
            room[i]--;
            bins[i] |= UINT64_C(1) << cards[index];
            if (place_exactly(cards, count, index + 1, bins_count, room, voids, bins, rng))
                return 1;
            room[i]++;
            bins[i] ^= UINT64_C(1) << cards[index];
        }
    }
    *slot = key;
    return 0;
}

static int sample_hands(const Obs *obs, u64 *hands, u64 *rng) {
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
                if (pressure[SUIT_OF(unknown[i])] == level) ordered[fill++] = unknown[i];

        int room[MAX_SEATS + 1];
        memcpy(room, capacity, sizeof(int) * bins_count);
        memset(bins, 0, sizeof(u64) * bins_count);
        done = 1;
        for (int i = 0; i < unknown_count; i++) {
            int suit_bit = 1 << SUIT_OF(ordered[i]);
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
        if (!place_exactly(ordered, unknown_count, 0, bins_count, room, voids, bins, rng))
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
            int options = POP(legal_mask(st.hands[seat], st.lead_suit));
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
    int first = (obs->dealer + 1) % obs->n;
    for (int s = 0; s < obs->n; s++) {
        if (s != obs->seat && obs->bids[s] != NO_BID) {
            double probs[13], estimate = 0.0;
            int count = card_win_probs(opening[s], obs->trump_card, obs->n, obs->hand_size,
                                       s == first, probs);
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
                                            obs->hand_size, s == first, p);
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
