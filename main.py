from player import Player
from elo import simulate_elo, expected_score
import matplotlib.pyplot as plt
import statistics



if __name__ == "__main__":
    m = 100  # Number of players <328
    p = 16   # Players per tournament
    n = 100  # Number of tournaments
    k = 200  # K-factor

    final_ratings = simulate_elo(m, p, n, k)

    # Save players to a file
    with open("player_ratings.txt", "w") as file:
        for player in final_ratings:
            file.write(f"{player}\n")

    # Gather ratings for statistics
    ratings = [player.rating for player in final_ratings]

    # Print statistics to console
    print(f"Variance: {statistics.variance(ratings):.2f}")
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
