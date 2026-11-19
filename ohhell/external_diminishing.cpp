/* Included after the pinned, compatibility-patched upstream main.cc.
 * Only own cards, played cards, public bids/voids/scores enter this ABI.
 */
extern "C" __attribute__((visibility("default"))) int league_dim_choose(
    int n, int hand_size, int seat, int trump, int bidding, int forbidden,
    int algorithm, int budget, double seconds, unsigned int seed,
    const int *grid_in, const int *trick_in, const int *order_in,
    const int *bids_in, const int *won_in, const int *scores_in, const int *voids_in) {
    numberOfPlayers = n;
    numberOfRounds = 15;
    numberOfSuits = 4;
    numberOfCards = 60; // rank-grid capacity; eight absent slots are marked -2
    orderSize = 15;
    currentTrump = trump;
    mcBudget = std::max(budget, 15);
    srand(seed);
    int grid[4][MAX_ROUNDS], played[MAX_PLAYERS][CARD_DIM], order[MAX_PLAYERS];
    int bids[MAX_PLAYERS], won[MAX_PLAYERS], scores[MAX_PLAYERS];
    int moves[MAX_ROUNDS][CARD_DIM];
    std::fill(&moves[0][0], &moves[0][0] + MAX_ROUNDS * CARD_DIM, -1);
    std::copy(grid_in, grid_in + 60, &grid[0][0]);
    std::copy(trick_in, trick_in + n * 2, &played[0][0]);
    std::copy(order_in, order_in + n, order);
    std::copy(bids_in, bids_in + n, bids);
    std::copy(won_in, won_in + n, won);
    std::copy(scores_in, scores_in + n, scores);
    memset(suitPlayer, 0, sizeof(suitPlayer));
    for(int s = 0; s < 4; s++) for(int p = 0; p < n; p++)
        suitPlayer[s][p] = !!voids_in[s * n + p];
    league_seconds = seconds;
    league_start = std::chrono::steady_clock::now();
    if(bidding) return evalBidMC(grid, seat, forbidden, order, hand_size);
    int count = 0;
    possibleMoves(moves, grid, count, played, seat, hand_size);
    if(count < 1) return -999;
    int choice = algorithm == 0
        ? monteCarloAgent(mcBudget, seat, hand_size, scores, order, played, moves, grid, bids, won)
        : monteCarloTreeSearchAgent(mcBudget, seat, hand_size, scores, order, played, moves, grid, bids, won);
    if(choice < 0 || choice >= count) return -999;
    return moves[choice][0] * 15 + moves[choice][1];
}

extern "C" __attribute__((visibility("default"))) int league_dim_winner(
    int n, int trump, const int *trick) {
    numberOfPlayers=n; currentTrump=trump;
    int played[MAX_PLAYERS][CARD_DIM];
    std::copy(trick, trick + n*2, &played[0][0]);
    return trickWinner(played);
}
