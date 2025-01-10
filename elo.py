import numpy as np
from typing import Callable, List
from player import Player
from cython_run_tournament import run_tournament
import pandas as pd



def get_expected_score(tau):
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Calculate the expected score for a player."""
        return 1 / (1 + np.exp((rating_b - rating_a) / tau))

    return expected_score




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


def preprocess_history(history_df):
    """
    Preprocess history DataFrame into a dictionary for fast lookups.
    Args:
        history_df (pd.DataFrame): Match history with Winner, Loser, Winner_Score, Loser_Score.

    Returns:
        dict: Preprocessed match data for fast lookups.
    """
    history_dict = {}
    for _, row in history_df.iterrows():
        key = (row['Winner'], row['Loser'])
        history_dict[key] = {'Winner_Score': row['Winner_Score'], 'Loser_Score': row['Loser_Score']}
    return history_dict


def load_players_and_history(n_players, n_tournaments, games_per_tournament=15, initial_score=500):
    n_matches = n_tournaments*games_per_tournament

    df = pd.read_csv('./processed_matches.csv')

    players_names = list(set(list(df['Winner'].unique()) + list(df['Loser'].unique())))[:n_players]
    df = df[(df["Winner"].isin(players_names)) | (df["Loser"].isin(players_names))]
    players = [Player(player_name, initial_score) for player_name in players_names]

    history_dict = preprocess_history(df)

    return players, history_dict


def simulate_elo(m: int, p: int, n: int, k: float, elo_formula: Callable, decay_factor: float = 0.3, initial_score: int = 500):
    """
    Simulate n tournaments with m players and return the final ratings and snapshots.

    Args:
        initial_score (int): score players start with. by default equal to 500
        decay_factor (float): decay factor to apply
        m (int): Number of players.
        p (int): Players per tournament.
        n (int): Number of tournaments.
        k (float): K-factor for Elo updates.
        elo_formula (Callable): Function to calculate expected score.

    Returns:
        List[Player]: Final sorted ratings.
        np.ndarray: Snapshots of rankings (shape: m x (n // d)).
    """
    players, history = load_players_and_history(m, n, p - 1, initial_score)

    d = 5  # Interval for decay and snapshot
    num_snapshots = n  # Number of snapshots
    snapshots = np.zeros((m, num_snapshots))  # Preallocate memory for snapshots

    snapshot_index = 0  # To track snapshot columns

    for i in range(n):
        # Simulate one tournament
        run_tournament(players, k, p, history, elo_formula)
        snapshots[:, snapshot_index] = [player.rating for player in players]
        snapshot_index += 1
        # Apply decay and take a snapshot at intervals
        if i % d == d - 1:
            apply_decay(players, decay_factor)

    return sorted(players, key=lambda x: x.rating, reverse=True), snapshots, [player.id for player in players]
