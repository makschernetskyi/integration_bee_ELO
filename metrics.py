import numpy as np
from scipy.stats import kurtosis

from scipy.stats import skew


def calculate_skew(final_ratings):
    """
    Calculate the skewness of final Elo ratings.

    Args:
        final_ratings (ndarray): List of final Elo ratings.

    Returns:
        float: Skewness value.
    """
    return skew(final_ratings)


def calculate_kurtosis(final_ratings):
    """
    Calculate the kurtosis of final Elo ratings.

    Args:
        final_ratings (list): List of final Elo ratings.

    Returns:
        float: Kurtosis value.
    """
    return kurtosis(final_ratings, fisher=True)  # Fisher=True for normal kurtosis = 0


def calculate_drift(snapshots):
    return np.mean(np.abs(np.diff(snapshots, axis=1)), axis=1)


def time_to_convergence(snapshots, epsilon=0.01):
    diff = np.abs(np.diff(snapshots, axis=1))
    time_steps = np.argmax(diff < epsilon, axis=1)
    time_steps[time_steps == 0] = snapshots.shape[1]  # For players never converging
    return time_steps


def calculate_convergence_rate(snapshots, window=5):
    """
    Calculate convergence rate based on the trend of differences over time.

    Args:
        snapshots (ndarray): Snapshots of ratings over time (shape: m x n).
        window (int): Number of rounds to consider for measuring stabilization.

    Returns:
        float: Average convergence trend (negative values indicate stabilization).
    """
    # Limit the evaluation to the last `window` rounds
    if snapshots.shape[1] < window:
        window = snapshots.shape[1]

    # Compute absolute differences for all players over the last `window` rounds
    diffs = np.abs(np.diff(snapshots[:, -window:], axis=1))  # Shape: (m, window-1)

    # Time indices (x values for regression) replicated for all players
    x = np.arange(diffs.shape[1])
    x_mean = np.mean(x)

    # Vectorized calculation of slopes (m = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)^2)
    y = diffs  # Differences are the "y" values
    y_mean = np.mean(y, axis=1, keepdims=True)
    numerator = np.sum((x - x_mean) * (y - y_mean), axis=1)
    denominator = np.sum((x - x_mean) ** 2)
    slopes = numerator / denominator  # Slopes for all players

    # Return the average slope across all players
    return np.mean(slopes)


def calculate_mac(snapshots):
    return np.mean(np.abs(np.diff(snapshots, axis=1)), axis=1)


def combined_metric_with_constraints(
    final_ratings, snapshots, target_skew=1.0, target_kurtosis=4.0, weights=None, convergence_window=5
):
    """
    Calculate the combined metric as an objective function with constraints.

    Args:
        final_ratings (ndarray): Final Elo ratings (already sorted in descending order).
        snapshots (ndarray): Snapshots of ratings over time.
        target_skew (float): Desired skewness value.
        target_kurtosis (float): Desired kurtosis value.
        weights (dict): Weights for each metric (keys: skew, kurtosis, drift, convergence, top_performance, out_of_bounds).
        convergence_window (int): Number of rounds to consider for convergence evaluation.

    Returns:
        float: Combined metric score.
    """
    if weights is None:
        weights = {
            "skew": 10.0,
            "kurtosis": 0,
            "drift": 0,
            "convergence": 0,
            "top_performance": 4.0,
            "out_of_bounds": 50.0,  # Penalty weight for out-of-bounds ratings
        }

    final_ratings = np.array(final_ratings)


    # Calculate scalar metrics
    skewness = calculate_skew(final_ratings)
    kurt = calculate_kurtosis(final_ratings)

    # Aggregate vector metrics
    drift = np.mean(calculate_drift(snapshots))  # Mean drift
    convergence_rate = calculate_convergence_rate(snapshots, window=convergence_window)  # Trend of stabilization

    # Mean and standard deviation for the ratings
    mean_rating = np.mean(final_ratings)
    std_dev = np.std(final_ratings)

    # Top performance metric: Penalize deviation from target
    target_top_score = mean_rating + 2.2 * std_dev
    top_performers = np.mean(final_ratings[:10])  # Average of top 10 players
    top_performance_penalty = abs(top_performers - target_top_score)  # Penalize deviation

    # Constraint penalties
    skew_penalty = (
        0 if 0.5 <= skewness <= 1.5 else weights["skew"] * abs(skewness - target_skew)
    )
    kurt_penalty = (
        0 if 3 <= kurt <= 5 else weights["kurtosis"] * abs(kurt - target_kurtosis)
    )

    # Efficient penalty for out-of-bounds ratings using NumPy
    out_of_bounds_count = np.sum((final_ratings > 2500) | (final_ratings < 0))
    out_of_bounds_penalty = out_of_bounds_count * weights["out_of_bounds"]

    # Combined metric
    combined_score = (
        skew_penalty
        + kurt_penalty
        - weights["drift"] * drift  # Higher drift is better
        - weights["convergence"] * convergence_rate  # Higher stabilization trend is better
        + weights["top_performance"] * top_performance_penalty  # Align top performers
        + out_of_bounds_penalty  # Penalize out-of-bounds ratings
    )

    return combined_score


