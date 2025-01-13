import time  # Import time for measuring execution duration
from elo import simulate_elo, get_expected_score, f_factory
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy import stats
from metrics import calculate_convergence_rate
from tqdm import tqdm
import seaborn as sns
import pandas as pd


def run_single_simulation():
    num_players = 100  # Number of players <328
    players_per_tournament = 16  # Players per tournament
    num_tournaments = 5000  # Number of tournaments
    base_k = 512  # K-factor
    k_min = 20  # Increased to amplify dynamic K-factors
    k_max = 80  # Increased to reward top performers more in later rounds
    decay_factor = 0.2  # Added a small decay factor to stabilize ratings over time
    initial_rating = 700  # Reduced slightly to create more spread in the ratings
    tau = 400  # Reduced to increase the gap impact in the expected score calculation
    k_scaling = "static"  # "sqrt" scaling tends to favor higher-rated players
    lam = 1.3  # Deviation scaling factor

    # Generate f and f_inverse using f_factory
    f, f_inverse = f_factory(mu=initial_rating, lam=lam)

    # Start timing
    start_time = time.time()

    # Run the simulation
    final_ratings, snapshots, player_ids = simulate_elo(
        num_players=num_players,
        players_per_tournament=players_per_tournament,
        num_tournaments=num_tournaments,
        elo_formula=get_expected_score(tau),
        decay_factor=decay_factor,
        initial_rating=initial_rating,
        k_scaling=k_scaling,
        k_min=k_min,
        k_max=k_max,
        base_k=base_k,
        custom_k_factors=None,
        games_per_series=5,
        f=f,
        f_inverse=f_inverse,
    )

    # End timing
    end_time = time.time()
    duration = end_time - start_time

    # Save players to a file
    with open("player_ratings.txt", "w") as file:
        for player in final_ratings:
            file.write(f"{player}\n")

    # Gather ratings for statistics
    ratings = [player.rating for player in final_ratings]

    # Print statistics to console
    print(f"Execution Time: {duration:.2f} seconds")
    print(f"Variance: {statistics.variance(ratings):.2f}")
    print(f"Deviation: {np.sqrt(statistics.variance(ratings)):.2f}")
    print(f"Skew: {stats.skew(np.array(ratings)):.2f}")
    print(f"Highest Rating: {max(ratings):.2f}")
    print(f"Lowest Rating: {min(ratings):.2f}")
    print(f"Mean Rating: {statistics.mean(ratings):.2f}")
    print(f"Median Rating: {statistics.median(ratings):.2f}")

    # Plot and save distribution
    plt.hist(ratings, bins=10, edgecolor="black")
    plt.title("Elo Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.savefig("ratings_distribution.png")
    plt.show()


def run_n_simulations(num_simulations=10):
    num_players = 100  # Number of players <328
    players_per_tournament = 16  # Players per tournament
    num_tournaments = 1000  # Number of tournaments
    base_k = 64  # K-factor
    k_min = 20  # Increased to amplify dynamic K-factors
    k_max = 80  # Increased to reward top performers more in later rounds
    decay_factor = 0  # Added a small decay factor to stabilize ratings over time
    initial_rating = 700  # Reduced slightly to create more spread in the ratings
    tau = 80  # Reduced to increase the gap impact in the expected score calculation
    k_scaling = "linear"  # "sqrt" scaling tends to favor higher-rated players
    lam = 1.3  # Deviation scaling factor

    # Generate f and f_inverse using f_factory
    f, f_inverse = f_factory(mu=initial_rating, lam=lam)

    metrics = {
        "variance": [],
        "deviation": [],
        "skew": [],
        "highest_rating": [],
        "lowest_rating": [],
        "mean_rating": [],
        "median_rating": [],
        "mean_drift": [],
        "convergence": [],
    }

    print("Running multiple Elo simulations and collecting metrics...")

    # Use tqdm for progress bar
    with tqdm(total=num_simulations, desc="Simulations Progress") as progress_bar:
        for sim in range(num_simulations):
            start_time = time.time()

            # Run the simulation
            final_ratings, snapshots, player_ids = simulate_elo(
                num_players=num_players,
                players_per_tournament=players_per_tournament,
                num_tournaments=num_tournaments,
                elo_formula=get_expected_score(tau),
                decay_factor=decay_factor,
                initial_rating=initial_rating,
                k_scaling=k_scaling,
                k_min=k_min,
                k_max=k_max,
                base_k=base_k,
                custom_k_factors=None,
                games_per_series=5,
                f=f,
                f_inverse=f_inverse,
            )

            end_time = time.time()
            progress_bar.set_postfix({"Simulation Time (s)": f"{end_time - start_time:.2f}"})
            progress_bar.update(1)

            # Gather ratings and metrics
            ratings = [player.rating for player in final_ratings]
            metrics["variance"].append(statistics.variance(ratings))
            metrics["deviation"].append(np.sqrt(statistics.variance(ratings)))
            metrics["skew"].append(stats.skew(np.array(ratings)))
            metrics["highest_rating"].append(max(ratings))
            metrics["lowest_rating"].append(min(ratings))
            metrics["mean_rating"].append(statistics.mean(ratings))
            metrics["median_rating"].append(statistics.median(ratings))
            metrics["mean_drift"].append(np.mean(np.abs(np.diff(snapshots, axis=1))))  # Drift
            metrics["convergence"].append(calculate_convergence_rate(snapshots, window=5))  # Convergence rate

    # Compute mean values
    mean_metrics = {key: np.mean(value) for key, value in metrics.items()}

    # Output mean values to console
    print("\nAverage Metrics Across Simulations:")
    for key, value in mean_metrics.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")

    # Prepare data for correlation matrix
    metrics_df = pd.DataFrame(metrics)

    # Compute correlation matrix
    correlation_matrix = metrics_df.corr()

    # Plot correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    # plt.title("Correlation Matrix of Metrics")
    # plt.savefig("correlation_matrix.png")
    # plt.show()

    # Plot individual metrics
    num_metrics = len(metrics)
    fig, axes = plt.subplots(nrows=(num_metrics // 3) + 1, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(range(1, num_simulations + 1), metric_values, label=metric_name, marker="o")
        ax.set_title(metric_name.replace("_", " ").capitalize())
        ax.set_xlabel("Simulation Number")
        ax.set_ylabel("Metric Value")
        ax.legend()

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("metrics_individual_plots.png")
    plt.show()


if __name__ == "__main__":
    run_single_simulation()
