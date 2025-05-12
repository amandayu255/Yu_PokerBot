import random
import time
import itertools
import math

# card ranks and suits for a standard 52-card deck
RANKS = '23456789TJQKA'
SUITS = 'CDHS'

# create a 52-card deck
def create_deck():
    return [r + s for r in RANKS for s in SUITS]

# shuffle the deck
def shuffle_deck(deck):
    random.shuffle(deck)

# determine rank of a 5-card poker hand
# higher numbers = stronger hands
def hand_rank(hand):
    # convert ranks to numerical values for comparison
    ranks = sorted(['--23456789TJQKA'.index(r) for r, s in hand], reverse=True)
    suits = [s for r, s in hand]

    flush = len(set(suits)) == 1  # all suits the same?
    straight = ranks == list(range(ranks[0], ranks[0]-5, -1))  # consecutive rank values?

    # count how many times each rank appears
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    counts = sorted(rank_counts.values(), reverse=True)

    # assign score based on hand type
    if straight and flush:
        return (8, ranks) # straight flush
    if counts[0] == 4:
        return (7, ranks) # four of a kind
    if counts[0] == 3 and counts[1] == 2:
        return (6, ranks) # full house
    if flush:
        return (5, ranks)
    if straight:
        return (4, ranks)
    if counts[0] == 3:
        return (3, ranks)
    if counts[0] == 2 and counts[1] == 2:
        return (2, ranks)  # two pair
    if counts[0] == 2:
        return (1, ranks)  # one pair
    return (0, ranks)  # high card

# get the best 5-card hand from 7 cards
def best_hand(cards):
    return max(itertools.combinations(cards, 5), key=hand_rank)

# compare two hands
# return 1 if hand1 wins, -1 if hand2 wins, 0 for tie
def compare_hands(hand1, hand2):
    return (hand_rank(hand1) > hand_rank(hand2)) - (hand_rank(hand1) < hand_rank(hand2))

# node used in MCTS to represent a possible opponent hand
class Node:
    def __init__(self, opp_cards):
        self.opp_cards = opp_cards  # opponent's 2 cards
        self.wins = 0
        self.simulations = 0

    # UCB1 to balance between exploration and exploitation
    def ucb1(self, total_simulations):
        if self.simulations == 0:
            return float('inf')  # ensure each node is visited at least once
        win_rate = self.wins / self.simulations
        exploration = math.sqrt(2 * math.log(total_simulations) / self.simulations)
        return win_rate + exploration

# Monte Carlo simulation to decide fold or stay
class PokerBot:
    def __init__(self):
        self.deck = []

    # remove known cards from deck
    def reset_deck(self, known_cards):
        self.deck = create_deck()
        for card in known_cards:
            self.deck.remove(card)
        shuffle_deck(self.deck)

    # Monte Carlo simulation to calculate how often we win
    def simulate(self, my_cards, community_cards, start_time, time_limit=9.5):
        possible_opp_hands = []
        seen = set()

        # get all valid 2-card combinations for opponent
        for i in range(len(self.deck)):
            for j in range(i+1, len(self.deck)):
                hand = tuple(sorted((self.deck[i], self.deck[j])))
                if hand not in seen:
                    seen.add(hand)
                    possible_opp_hands.append(Node(hand))

        total_simulations = 0

        # run simulations until time runs out
        while time.time() - start_time < time_limit:
            # pick opponent hand using UCB1
            node = max(possible_opp_hands, key=lambda n: n.ucb1(total_simulations + 1))

            # make a temporary deck without those opponent cards
            unknown = self.deck.copy()
            unknown.remove(node.opp_cards[0])
            unknown.remove(node.opp_cards[1])
            random.shuffle(unknown)

            # build the full board (5 cards total)
            board = community_cards.copy()
            while len(board) < 5:
                board.append(unknown.pop())

            # compare best hands for us and opponent
            full_hand_my = my_cards + board
            full_hand_opp = list(node.opp_cards) + board
            result = compare_hands(best_hand(full_hand_my), best_hand(full_hand_opp))

            # Record win/loss
            node.simulations += 1
            if result == 1:
                node.wins += 1
            total_simulations += 1

        # calculate win rate overall
        wins = sum(n.wins for n in possible_opp_hands)
        sims = sum(n.simulations for n in possible_opp_hands)
        return wins / sims if sims > 0 else 0.0

    # make a decision if fold or stay
    def decide(self, my_cards, community_cards):
        known_cards = my_cards + community_cards
        self.reset_deck(known_cards)
        start_time = time.time()
        win_rate = self.simulate(my_cards, community_cards, start_time)
        print(f"Estimated win probability: {win_rate:.2f}")
        return 'STAY' if win_rate >= 0.5 else 'FOLD'

if __name__ == '__main__':
    bot = PokerBot()
    deck = create_deck()
    shuffle_deck(deck)

    # deal two cards to us and opponent
    my_cards = [deck.pop(), deck.pop()]
    opp_hidden = [deck.pop(), deck.pop()]

    print(f"My cards: {my_cards}")

    # pre-flop decision
    community_cards = []
    print("Pre-Flop decision:")
    print(bot.decide(my_cards, community_cards))

    # deal flop
    community_cards += [deck.pop(), deck.pop(), deck.pop()]
    print(f"\nFlop: {community_cards}")
    print("Pre-Turn decision:")
    print(bot.decide(my_cards, community_cards))

    # deal turn
    community_cards += [deck.pop()]
    print(f"\nTurn: {community_cards}")
    print("Pre-River decision:")
    print(bot.decide(my_cards, community_cards))

    # deal river
    community_cards += [deck.pop()]
    print(f"\nRiver: {community_cards}")
    print("Final showdown decision:")
    print(bot.decide(my_cards, community_cards))

    # show opponent hand and determine the winner
    print(f"\nOpponent's cards: {opp_hidden}")
    my_best = best_hand(my_cards + community_cards)
    opp_best = best_hand(opp_hidden + community_cards)

    result = compare_hands(my_best, opp_best)
    if result > 0:
        print("You WIN!")
    elif result < 0:
        print("You LOSE.")
    else:
        print("It's a TIE.")