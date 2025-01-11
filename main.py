import time  # Import time for measuring execution duration
from elo import simulate_elo, get_expected_score
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy import stats


if __name__ == "__main__":
    m = 100  # Number of players <328
    p = 16   # Players per tournament
    n = 100  # Number of tournaments
    k = 298  # K-factor
    k_min = 77.89065013918406
    k_max = 239.38881917970338
    decay_factor = 0.46
    initial_score = 458
    tau = 134.96906361018003
    k_scaling = "sqrt"

    # Optimal
    # Parameters:
    # k_min: 77.2462689286173
    # k_max: 239.10325082909344
    # decay_factor: 0.4616222104959444
    # initial_score: 458
    # tau: 134.20290416221096
    # Best
    # Metric
    # Score: 75.4955

    # Start timing
    start_time = time.time()

    # Run the simulation
    final_ratings, snapshots, player_ids = simulate_elo(m=m, p=p, n=n, elo_formula=get_expected_score(tau), decay_factor=decay_factor, initial_score=initial_score, k_min=k_min, k_max=k_max, k_scaling=k_scaling)

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
    plt.hist(ratings, bins=10, edgecolor='black')
    plt.title("Elo Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.savefig("ratings_distribution.png")
    plt.show()
