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
    k = 63.7  # K-factor
    decay_factor = 0.48
    initial_score = 536
    tau = 122.54

    # Optimal Parameters:
    # k: 63.70079792992623
    # decay_factor: 0.48427076541341285
    # initial_score: 536
    # tau: 122.54110256115277

    # Start timing
    start_time = time.time()

    # Run the simulation
    final_ratings, snapshots, player_ids = simulate_elo(m, p, n, k, get_expected_score(tau), decay_factor, initial_score)

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
    print(f"Skew: {np.sqrt(stats.skew(np.array(ratings))):.2f}")
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
