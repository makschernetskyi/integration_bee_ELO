import optuna
from metrics import combined_metric_with_constraints  # Import your combined metric function
from elo import simulate_elo, get_expected_score
import concurrent.futures
import numpy as np


def create_objective_function(mode="dynamic"):
    """
    Factory function to create an objective function for optimization.

    Args:
        mode (str): Optimization mode ("dynamic", "static", "custom").

    Returns:
        Callable: Objective function for Optuna.
    """
    def objective_function(trial):
        """
        Generalized objective function based on mode.
        """
        if mode == "dynamic":
            k_min = trial.suggest_float("k_min", 10, 100)
            k_max = trial.suggest_float("k_max", 100, 800)
            k_scaling = "sqrt"  # Fixed scaling type
            custom_k = None
            k = None
        elif mode == "static":
            k = trial.suggest_float("k", 10, 800)
            k_min, k_max, k_scaling, custom_k = None, None, None, None
        elif mode == "custom":
            custom_k = trial.suggest_categorical("custom_k", [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
            k, k_min, k_max, k_scaling = None, None, None, None
        else:
            raise ValueError("Invalid mode. Choose from 'dynamic', 'static', or 'custom'.")

        # New parameters for deviation multiplier
        max_deviation_multiplier = trial.suggest_float("max_deviation_multiplier", 1.0, 20.0)
        deviation_scaling_factor = trial.suggest_float("deviation_scaling_factor", 10.0, 200.0)
        base_multiplier_factor = trial.suggest_float("base_multiplier_factor", 0.0, 1.0)

        # Other parameters
        decay_factor = trial.suggest_float("decay_factor", 0.0, 0.5)
        initial_score = trial.suggest_int("initial_score", 300, 800)
        tau = trial.suggest_float("tau", 20, 200)

        num_simulations = 10  # Number of simulations for averaging

        def run_simulation():
            final_ratings, snapshots, player_ids = simulate_elo(
                num_players=100,
                players_per_tournament=16,
                num_tournaments=1000,
                base_k=k,
                elo_formula=get_expected_score(tau),
                decay_factor=decay_factor,
                initial_rating=initial_score,
                k_min=k_min,
                k_max=k_max,
                k_scaling=k_scaling,
                custom_k_factors=custom_k,
                max_deviation_multiplier=max_deviation_multiplier,
                deviation_scaling_factor=deviation_scaling_factor,
                base_multiplier_factor=base_multiplier_factor,
            )
            ratings = [player.rating for player in final_ratings]
            return combined_metric_with_constraints(ratings, snapshots)

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            metric_values = list(executor.map(lambda _: run_simulation(), range(num_simulations)))

        return np.mean(metric_values)

    return objective_function


if __name__ == "__main__":
    # Prompt user for optimization mode
    optimization_mode = input("Enter optimization mode (dynamic/static/custom): ").strip().lower()

    # Create the objective function based on the mode
    objective_function = create_objective_function(mode=optimization_mode)

    # Create and run the optimization study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_function, n_trials=50)

    # Output the best parameters and metric score
    print("Optimal Parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best Metric Score: {study.best_value:.4f}")
