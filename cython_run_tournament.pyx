# cython_run_tournament.pyx
import numpy as np
cimport numpy as cnp

from player cimport Player  # Import Player class from an external file

cdef double dynamic_k(int round_num, int total_rounds, double k_min, double k_max, str scaling):
    """
    Calculate the dynamic K value based on the round number and scaling method.

    Args:
        round_num (int): Current round number (1-indexed).
        total_rounds (int): Total number of rounds in the tournament.
        k_min (double): Minimum K-factor (for the first round).
        k_max (double): Maximum K-factor (for the final round).
        scaling (str): Scaling method ("linear", "log", "sqrt").

    Returns:
        double: K-factor for the current round.
    """
    if scaling == "linear":
        return k_min + (round_num - 1) / (total_rounds - 1) * (k_max - k_min)
    elif scaling == "log":
        return k_min + np.log2(round_num + 1) / np.log2(total_rounds + 1) * (k_max - k_min)
    elif scaling == "sqrt":
        return k_min + np.sqrt(round_num) / np.sqrt(total_rounds) * (k_max - k_min)
    else:
        raise ValueError("Invalid scaling method for dynamic K. Choose 'linear', 'log', or 'sqrt'.")


cdef double apply_momentum(
    Player player,
    double k,
    double expected_score_w,
    double expected_score_l,
    double streak_multiplier=0.1,
    double max_gap_adjustment=10.0  # Set default value for max_gap_adjustment
):
    """
    Adjust the player's K-factor based on their current streak and the Elo gap.

    Args:
        player (Player): The player whose rating is being adjusted.
        k (double): The base K-factor.
        expected_score_w (double): Expected score of the winner.
        expected_score_l (double): Expected score of the loser.
        streak_multiplier (double): The scaling factor for streak effects (default: 0.4).
        max_gap_adjustment (double): Maximum allowed value for the gap adjustment (default: 10.0).

    Returns:
        double: Adjusted K-factor based on momentum.
    """
    cdef int streak = max(0, player.streak)  # Ensure streak is non-negative
    cdef double adjustment = streak_multiplier * streak

    # Apply streak-based and gap-based adjustments
    return k * adjustment


cdef double deviation_multiplier(double rating, double mean_rating, double h, double w, double p):
    """
    Calculate the multiplier based on a player's deviation from the mean rating.

    Args:
        rating (double): The player's current Elo rating.
        mean_rating (double): The mean Elo rating of all players.
        h (double): Maximum value of the multiplier.
        w (double): Scaling factor for deviation.
        p (double): Base multiplier for all players.

    Returns:
        double: Multiplier for Elo adjustment.
    """
    cdef double deviation = max(0, rating - mean_rating)  # Absolute deviation from the mean
    return h * (1 - p) * (1 - np.exp(-((deviation / w) ** 2))) + h * p + 1



cdef void update_elo(
    Player winner,
    Player loser,
    double k,
    object elo_formula,
    double match_outcome,
    double mean_rating,
    double h=2.0,
    double w=200.0,
    double p=0.5
):
    """
    Update Elo ratings for the winner and loser based on the Elo formula.

    Args:
        winner (Player): The player who won the match.
        loser (Player): The player who lost the match.
        k (double): The base K-factor.
        elo_formula (callable): Function to calculate expected score.
        match_outcome (double): Scaled outcome (range [0, 1]) based on match results.
        mean_rating (double): The mean Elo rating of all players.
        h (double): Maximum multiplier for deviation adjustment.
        w (double): Scaling factor for deviation adjustment.
        p (double): Base multiplier factor.
    """
    cdef double expected_w = elo_formula(winner.rating, loser.rating)
    cdef double expected_l = elo_formula(loser.rating, winner.rating)

    # Compute deviation-based multiplier for the winner
    cdef double dev_multiplier = deviation_multiplier(winner.rating, mean_rating, h, w, p)

    # Apply deviation multiplier to the K-factor
    cdef double adjusted_k = k * dev_multiplier

    # Update ratings using the adjusted K-factor
    winner.update_rating(winner.rating + adjusted_k * (match_outcome - expected_w))
    loser.update_rating(loser.rating + k * ((1 - match_outcome) - expected_l))

    # Increment matches played
    winner.matches_played += 1
    loser.matches_played += 1

    # Update streaks
    winner.update_streak(won=True)
    loser.update_streak(won=False)



def run_tournament(
    list players,
    int num_participants,
    dict history,
    object elo_formula,
    int games_per_series,
    str k_scaling="sqrt",
    double k=32,
    double k_min=100,
    double k_max=400,
    list custom_k=None,
    double h=2.0,  # Maximum multiplier for deviation adjustment
    double w=200.0,  # Scaling factor for deviation adjustment
    double base_multiplier=0.5  # Base multiplier factor for deviation adjustment
):
    """
    Simulate a tournament with a specified number of participants, supporting dynamic and static K.

    Args:
        players (List[Player]): List of players.
        num_participants (int): Number of participants in the tournament.
        history (dict): Preprocessed match data.
        elo_formula (Callable): Function to calculate expected score.
        games_per_series (int): Number of games in the "best-of-v" series format.
        k_scaling (str): Scaling method for dynamic K ("sqrt", "log", "linear", "static", "custom").
        k (double): Base K-factor for Elo updates (used for static K).
        k_min (double): Minimum K-factor (for the first round).
        k_max (double): Maximum K-factor (for the final round).
        custom_k (list): Custom list of K values, one for each round (only used if k_scaling="custom").
        h (double): Maximum multiplier for deviation adjustment.
        w (double): Scaling factor for deviation adjustment.
        base_multiplier (double): Base multiplier factor for deviation adjustment.
    """
    cdef int len_participants
    cdef Player player_1, player_2
    cdef list next_participants
    cdef int round_number = 1
    cdef int total_rounds = int(np.ceil(np.log2(num_participants)))  # Total rounds in the tournament
    cdef double current_k  # Declare here at the top of the function
    rng = np.random.default_rng()

    # Compute mean Elo rating
    cdef double mean_rating = np.mean([player.rating for player in players])

    # Randomly select participants
    participants = rng.choice(players, size=num_participants, replace=False).tolist()
    len_participants = len(participants)

    while len_participants > 1:
        next_participants = []

        # Determine K for the current round
        if k_scaling == "static":
            current_k = k
        elif k_scaling == "custom":
            if round_number <= len(custom_k):
                current_k = custom_k[round_number - 1]
            else:
                raise ValueError("Custom K array is smaller than the total number of rounds.")
        else:
            current_k = dynamic_k(round_number, total_rounds, k_min, k_max, k_scaling)

        # Simulate matches for the current round
        for i in range(0, len_participants, 2):
            player_1 = participants[i]
            player_2 = participants[i + 1]

            # Lookup historical outcome
            if (player_1.id, player_2.id) in history:
                # Retrieve scores
                winner_score = history[(player_1.id, player_2.id)]['Winner_Score']
                loser_score = history[(player_1.id, player_2.id)]['Loser_Score']

                # Determine winner and match outcome
                is_1_winner = winner_score > loser_score
                match_outcome = winner_score / games_per_series  # Scaled outcome
            else:
                # Random outcome if no history exists
                is_1_winner = rng.integers(2) == 0
                match_outcome = rng.random()  # Simulate a random outcome scale

            # Update Elo ratings based on winner
            if is_1_winner:
                update_elo(
                    player_1, player_2, current_k, elo_formula, match_outcome,
                    mean_rating, h, w, base_multiplier
                )
                next_participants.append(player_1)
            else:
                update_elo(
                    player_2, player_1, current_k, elo_formula, 1 - match_outcome,
                    mean_rating, h, w, base_multiplier
                )
                next_participants.append(player_2)

        # Prepare for the next round
        participants = next_participants
        len_participants = len(participants)
        round_number += 1




