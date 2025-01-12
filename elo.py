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
    num_players: int,
    players_per_tournament: int,
    num_tournaments: int,
    elo_formula: Callable,
    decay_factor: float = 0.3,
    initial_rating: int = 500,
    k_scaling: str = "sqrt",  # Scaling type for dynamic K-factor
    k_min: float = None,      # Min K-factor for scaling
    k_max: float = None,      # Max K-factor for scaling
    base_k: float = None,     # Base K-factor for static scaling
    custom_k_factors: list = None,  # Custom K-factor array
    games_per_series: int = 5,      # Number of games in the "best-of-v" series
    max_deviation_multiplier: float = 2.0,  # Maximum multiplier for deviation adjustment (h)
    deviation_scaling_factor: float = 200.0,  # Scaling factor for deviation adjustment (w)
    base_multiplier_factor: float = 0.5  # Base multiplier factor (p)
):
    """
    Simulate num_tournaments with num_players and return the final ratings and snapshots.

    Args:
        num_players (int): Number of players.
        players_per_tournament (int): Players per tournament.
        num_tournaments (int): Number of tournaments.
        elo_formula (Callable): Function to calculate expected score.
        decay_factor (float): Decay factor to apply.
        initial_rating (int): Initial Elo rating for players.
        k_scaling (str): Scaling type for K-factor ("log", "sqrt", "linear", "static", "custom").
        k_min (float): Min K-factor for scaling (used for "log", "sqrt", "linear").
        k_max (float): Max K-factor for scaling.
        base_k (float): K-factor for Elo updates (used for static scaling).
        custom_k_factors (list): Custom K-factor array for each round.
        games_per_series (int): Number of games in the "best-of-v" series.
        max_deviation_multiplier (float): Maximum multiplier for deviation adjustment (h).
        deviation_scaling_factor (float): Scaling factor for deviation adjustment (w).
        base_multiplier_factor (float): Base multiplier factor (p).

    Returns:
        List[Player]: Final sorted ratings.
        np.ndarray: Snapshots of rankings (shape: num_players x num_tournaments).
        List[str]: Player IDs in order.
    """
    # Load players and history
    players, history = load_players_and_history(
        num_players, num_tournaments, players_per_tournament - 1, initial_rating
    )

    interval_for_decay = 5  # Interval for applying decay
    snapshots = np.zeros((num_players, num_tournaments))  # Preallocate memory for snapshots

    # Dynamically build the arguments for run_tournament
    run_tournament_args = {
        "players": players,
        "num_participants": players_per_tournament,
        "history": history,
        "elo_formula": elo_formula,
        "games_per_series": games_per_series,
        "h": max_deviation_multiplier,
        "w": deviation_scaling_factor,
        "base_multiplier": base_multiplier_factor,
    }

    # Add optional arguments if they are not None
    if base_k is not None:
        run_tournament_args["k"] = base_k
    if k_scaling:
        run_tournament_args["k_scaling"] = k_scaling
    if k_min is not None:
        run_tournament_args["k_min"] = k_min
    if k_max is not None:
        run_tournament_args["k_max"] = k_max
    if custom_k_factors is not None:
        run_tournament_args["custom_k"] = custom_k_factors

    # Run tournaments
    for tournament_index in range(num_tournaments):
        run_tournament(**run_tournament_args)

        # Store player ratings for snapshots
        snapshots[:, tournament_index] = [player.rating for player in players]

        # Apply decay at intervals
        if tournament_index % interval_for_decay == interval_for_decay - 1:
            apply_decay(players, decay_factor)

    return sorted(players, key=lambda x: x.rating, reverse=True), snapshots, [player.id for player in players]




