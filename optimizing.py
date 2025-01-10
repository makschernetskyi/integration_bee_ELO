import optuna
from metrics import combined_metric_with_constraints  # Import your combined metric function
from elo import simulate_elo, get_expected_score
import concurrent.futures
import numpy as np


def objective_function(trial):
    """
    Objective function for Bayesian optimization with parallelized simulations.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Average metric value across multiple simulation runs.
    """
    # Suggest parameters to optimize
    k = trial.suggest_float("k", 50, 400)  # Range for K-factor
    decay_factor = trial.suggest_float("decay_factor", 0.1, 0.5)  # Range for decay factor
    initial_score = trial.suggest_int("initial_score", 300, 800)  # Range for initial score
    tau = trial.suggest_float("tau", 50, 200)  # Range for tau

    num_simulations = 5  # Number of runs to average

    def run_simulation():
        """Run a single Elo simulation and calculate the metric."""
        final_ratings, snapshots, player_ids = simulate_elo(
            m=100,  # Number of players
            p=16,   # Players per tournament
            n=100,  # Number of tournaments
            k=k,
            elo_formula=get_expected_score(tau),
            decay_factor=decay_factor,
            initial_score=initial_score
        )
        ratings = [player.rating for player in final_ratings]
        return combined_metric_with_constraints(ratings, snapshots)

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        metric_values = list(executor.map(lambda _: run_simulation(), range(num_simulations)))

    # Return the average metric value
    return np.mean(metric_values)


if __name__ == "__main__":
    # Create and run the optimization study
    study = optuna.create_study(direction="minimize")  # We aim to minimize the metric
    study.optimize(objective_function, n_trials=50)  # Adjust n_trials as needed

    # Output the best parameters and metric score
    print("Optimal Parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best Metric Score: {study.best_value:.4f}")
