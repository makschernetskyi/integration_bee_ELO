import numpy as np
from typing import Callable, List
from player import Player
import pandas as pd

def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate the expected score for a player."""
    return 1 / (1 + np.exp((rating_b - rating_a) / 100))

def update_elo(winner: Player, loser: Player, k: float = 32, elo_formula: Callable = expected_score):
    """Update ratings for winner and loser based on the Elo formula."""
    expected_w = elo_formula(winner.rating, loser.rating)
    expected_l = elo_formula(loser.rating, winner.rating)

    winner.rating += np.floor(k * (1 - expected_w))
    loser.rating += np.floor(k * (0 - expected_l))

    winner.matches_played +=1
    loser.matches_played +=1

def apply_decay(players: List[Player], decay_factor: float = 0.3):
    """
    Apply decay to player ratings, bringing them closer to the mean.

    Args:
        players (List[Player]): List of Player objects.
        decay_factor (float): The fraction by which ratings decay toward the mean (default is 0.1).
    """
    mean_rating = np.mean([player.rating for player in players])
    for player in players:
        player.rating += decay_factor * (mean_rating - player.rating)

def run_tournament(players: List[Player], k: float, p: int,  elo_formula: Callable, history):
    """Simulate a tournament with p players randomly chosen."""
    rng = np.random.default_rng()
    participants = rng.choice(players, size=p, replace=False).tolist()

    # Randomly shuffle and match players for a single elimination tournament
    current_participants = participants.copy()
    next_participants = []

    while len(current_participants) > 1:
        for i in range(0, len(current_participants), 2):
            player_1, player_2 = current_participants[i], current_participants[i+1]

            # Retrieve historical match outcome
            outcome = history[((history['Winner'] == player_1.id) & (history['Loser'] == player_2.id)) |
                              ((history['Winner'] == player_2.id) & (history['Loser'] == player_1.id))]

            if outcome.shape[0] > 0:
                is_1_winner = outcome.sample(1).iloc[0]['Winner'] == player_1.id
            else:
                is_1_winner = rng.integers(2) == 0  # Random outcome if no history exists

            if is_1_winner:
                update_elo(player_1, player_2, k, elo_formula)
                next_participants.append(player_1)
            else:
                update_elo(player_2, player_1, k, elo_formula)
                next_participants.append(player_2)

        current_participants = next_participants
        next_participants = []


def load_players_and_history(n_players, n_tournaments, games_per_tournament=15):
    n_matches = n_tournaments*games_per_tournament

    df = pd.read_csv('./processed_matches.csv')

    players_names = list(set(list(df['Winner'].unique()) + list(df['Loser'].unique())))[:n_players]
    df = df[(df["Winner"].isin(players_names)) | (df["Loser"].isin(players_names))]
    players = [Player(player_name) for player_name in players_names]

    return players, df


def simulate_elo(m: int, p: int, n: int, k: float, elo_formula: Callable = expected_score):
    """Simulate n tournaments with m players and return the final ratings."""
    players, history = load_players_and_history(m, n, p-1)


    for i in range(n):
        run_tournament(players, k, p, elo_formula, history)

        if i%5 == 4:
            apply_decay(players)

    return sorted(players, key=lambda x: x.rating, reverse=True)
