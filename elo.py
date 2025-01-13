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


def f_factory(mu: float, lam: float):
    """
    Create a piecewise linear transformation function f and its inverse f⁻¹.

    Args:
        mu (float): Mean rating.
        lam (float): Deviation scaling factor (>1).

    Returns:
        tuple: A tuple (f, f_inverse) of callable functions.
    """
    def f(x: float) -> float:
        if x > mu:
            return (x - mu) * lam + mu
        elif x < mu:
            return (x - mu) / lam + mu
        else:
            return x

    def f_inverse(x: float) -> float:
        if x > mu:
            return (x - mu) / lam + mu
        elif x < mu:
            return (x - mu) * lam + mu
        else:
            return x

    return f, f_inverse


def apply_decay(players: List[Player], decay_factor: float, f: Callable, f_inverse: Callable):
    """
    Apply decay to player ratings, bringing them closer to the mean.

    Args:
        f: transformation function to adjust ratings
        players (List[Player]): List of Player objects.
        decay_factor (float): The fraction by which ratings decay toward the mean.
        f_inverse (Callable): Inverse transformation function to adjust ratings.
    """
    mean_rating = np.mean([f_inverse(player.rating) for player in players])  # Use transformed ratings
    for player in players:
        actual_rating = f_inverse(player.rating)
        actual_rating += decay_factor * (mean_rating - actual_rating)
        player.rating = f(actual_rating)  # Transform back



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
    decay_factor: float,
    initial_rating: int,
    k_scaling: str,
    k_min: float = None,
    k_max: float = None,
    base_k: float = None,
    custom_k_factors: list = None,
    games_per_series: int = 5,
    f: Callable = None,
    f_inverse: Callable = None,
):
    """
    Simulate tournaments with the f transformation applied.

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
        base_k (float): Base K-factor for static scaling.
        custom_k_factors (list): Custom K-factor array for each round.
        games_per_series (int): Number of games in the "best-of-v" series.
        f (Callable): Transformation function for displayed ratings.
        f_inverse (Callable): Inverse transformation function for actual ratings.

    Returns:
        List[Player]: Final sorted ratings.
        np.ndarray: Snapshots of rankings (shape: num_players x num_tournaments).
        List[str]: Player IDs in order.
    """
    # Load players and history
    players, history = load_players_and_history(
        n_players=num_players,
        n_tournaments=num_tournaments,
        games_per_tournament=players_per_tournament - 1,
        initial_score=initial_rating,
    )

    # Preallocate snapshots for ratings
    snapshots = np.zeros((num_players, num_tournaments))

    for tournament_index in range(num_tournaments):
        # Build arguments for run_tournament dynamically
        tournament_args = {
            "players": players,
            "num_participants": players_per_tournament,
            "history": history,
            "elo_formula": lambda a, b: elo_formula(f_inverse(a), f_inverse(b)),  # Use true ratings
            "games_per_series": games_per_series,
            "k_scaling": k_scaling,
            "f": f,
            "f_inverse": f_inverse,
            "initial_rating": initial_rating,
        }

        # Add K-related arguments conditionally
        if k_scaling in ["log", "sqrt", "linear"]:
            tournament_args["k_min"] = k_min
            tournament_args["k_max"] = k_max
        elif k_scaling in ["static"]:
            tournament_args["k"] = base_k
        elif k_scaling == "custom":
            tournament_args["custom_k"] = custom_k_factors

        # Run the tournament
        run_tournament(**tournament_args)

        # Store transformed (displayed) ratings for snapshots
        snapshots[:, tournament_index] = [player.rating for player in players]

        # Apply decay every 5 tournaments
        if tournament_index % 5 == 4:
            apply_decay(players, decay_factor, f, f_inverse)

    # Return final sorted ratings, snapshots, and player IDs
    return (
        sorted(players, key=lambda x: x.rating, reverse=True),
        snapshots,
        [player.id for player in players],
    )






