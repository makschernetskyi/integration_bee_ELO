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


def load_players_and_history(n_players, n_tournaments, games_per_tournament=15, initial_score=500, v=3):
    """
    Load players and match history, normalizing scores to the best-of-v format and rounding to integers.

    Args:
        n_players (int): Number of players to include.
        n_tournaments (int): Number of tournaments.
        games_per_tournament (int): Number of games per tournament.
        initial_score (int): Initial Elo rating for players.
        v (int): Number of games in the best-of-v format.

    Returns:
        List[Player]: List of Player objects.
        dict: Preprocessed match data with normalized scores.
    """
    n_matches = n_tournaments * games_per_tournament

    # Load match history
    df = pd.read_csv('./processed_matches.csv')

    # Filter players
    players_names = list(set(list(df['Winner'].unique()) + list(df['Loser'].unique())))[:n_players]
    df = df[(df["Winner"].isin(players_names)) | (df["Loser"].isin(players_names))]

    # Normalize scores to best-of-v and round to integers
    max_score = df[['Winner_Score', 'Loser_Score']].max().max()  # Find the maximum possible score in the dataset
    df['Winner_Score'] = ((df['Winner_Score'] / max_score) * v).round().astype(int)
    df['Loser_Score'] = ((df['Loser_Score'] / max_score) * v).round().astype(int)

    # Create Player objects
    players = [Player(player_name, initial_score) for player_name in players_names]

    # Preprocess history into a dictionary
    history_dict = preprocess_history(df)

    return players, history_dict



def simulate_elo(
    m: int,
    p: int,
    n: int,
    elo_formula: Callable,
    decay_factor: float = 0.3,
    initial_score: int = 500,
    k_scaling: str = "sqrt",  # Scaling type for dynamic K-factor
    k_min: float = None,      # Min K-factor for scaling
    k_max: float = None,
    k: float = None,    # Max K-factor for scaling
    custom_k: list = None,    # Custom K-factor array
    v: int = 5                # Number of games in the "best-of-v" series
):
    """
    Simulate n tournaments with m players and return the final ratings and snapshots.

    Args:
        m (int): Number of players.
        p (int): Players per tournament.
        n (int): Number of tournaments.
        k (float): K-factor for Elo updates.
        elo_formula (Callable): Function to calculate expected score.
        decay_factor (float): Decay factor to apply.
        initial_score (int): Initial Elo rating for players.
        k_scaling (str): Scaling type for K-factor ("log", "sqrt", "linear", "static", "custom").
        k_min (float): Min K-factor for scaling (used for "log", "sqrt", "linear").
        k_max (float): Max K-factor for scaling.
        custom_k (list): Custom K-factor array for each round.
        v (int): Number of games in the "best-of-v" series.

    Returns:
        List[Player]: Final sorted ratings.
        np.ndarray: Snapshots of rankings (shape: m x n).
        List[str]: Player IDs in order.
    """
    players, history = load_players_and_history(m, n, p - 1, initial_score)

    d = 5  # Interval for decay
    snapshots = np.zeros((m, n))  # Preallocate memory for snapshots

    # Dynamically build the arguments for run_tournament
    run_args = {
        "players": players,
        "p": p,
        "history": history,
        "elo_formula": elo_formula,
        "v": v
    }

    # Add optional arguments if they are not None
    if k is not None:
        run_args["k"] = k
    if k_scaling:
        run_args["k_scaling"] = k_scaling
    if k_min is not None:
        run_args["k_min"] = k_min
    if k_max is not None:
        run_args["k_max"] = k_max
    if custom_k is not None:
        run_args["custom_k"] = custom_k

    for i in range(n):
        run_tournament(**run_args)

        # Store player ratings for snapshots
        snapshots[:, i] = [player.rating for player in players]

        # Apply decay at intervals
        if i % d == d - 1:
            apply_decay(players, decay_factor)

    return sorted(players, key=lambda x: x.rating, reverse=True), snapshots, [player.id for player in players]


