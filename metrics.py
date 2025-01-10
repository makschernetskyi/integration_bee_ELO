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


def calculate_convergence_rate(snapshots, threshold=0.01):
    # Compute differences
    diffs = np.abs(np.diff(snapshots[:, -5:], axis=1))
    return np.mean(diffs < threshold, axis=1)


def calculate_mac(snapshots):
    return np.mean(np.abs(np.diff(snapshots, axis=1)), axis=1)


def combined_metric_with_constraints(
    final_ratings, snapshots, target_skew=1.0, target_kurtosis=4.0, weights=None
):
    """
    Calculate the combined metric as an objective function with constraints.

    Args:
        final_ratings (ndarray): Final Elo ratings.
        snapshots (ndarray): Snapshots of ratings over time.
        target_skew (float): Desired skewness value.
        target_kurtosis (float): Desired kurtosis value.
        weights (dict): Weights for each metric (keys: skew, kurtosis, drift, mac).

    Returns:
        float: Combined metric score.
    """
    if weights is None:
        weights = {
            "skew": 1.0,
            "kurtosis": 1.0,
            "drift": 1.5,
            "mac": 2.0,
            "penalty": 10.0,  # Penalty weight for violating constraints
        }

    # Calculate scalar metrics
    skewness = calculate_skew(final_ratings)
    kurt = calculate_kurtosis(final_ratings)

    # Aggregate vector metrics
    drift = np.mean(calculate_drift(snapshots))  # Mean drift
    mac = np.mean(calculate_mac(snapshots))      # Mean MAC

    # Constraint penalties
    skew_penalty = (
        0 if 0 <= skewness <= 1.5 else weights["penalty"] * abs(skewness - target_skew)
    )
    kurt_penalty = (
        0 if 3 <= kurt <= 5 else weights["penalty"] * abs(kurt - target_kurtosis)
    )

    # Combined metric
    combined_score = (
        weights["skew"] * skew_penalty
        + weights["kurtosis"] * kurt_penalty
        - weights["drift"] * drift  # Higher drift is better
        + weights["mac"] * mac  # Lower MAC is better
    )

    return combined_score