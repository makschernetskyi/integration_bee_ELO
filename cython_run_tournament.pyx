import numpy as np
cimport numpy as cnp

from player cimport Player  # Import Player class from an external file

cdef double dynamic_k(int round_num, int total_rounds, double k_min, double k_max, str scaling):
    """
    Calculate the dynamic K value based on the round number and scaling method.
    """
    if scaling == "linear":
        return k_min + (round_num - 1) / (total_rounds - 1) * (k_max - k_min)
    elif scaling == "log":
        return k_min + np.log2(round_num + 1) / np.log2(total_rounds + 1) * (k_max - k_min)
    elif scaling == "sqrt":
        return k_min + np.sqrt(round_num) / np.sqrt(total_rounds) * (k_max - k_min)
    else:
        raise ValueError("Invalid scaling method for dynamic K. Choose 'linear', 'log', or 'sqrt'.")


cdef void update_elo(
    Player winner,
    Player loser,
    double k,
    object elo_formula,
    double match_outcome,
    double mean_rating,
    object f=None,
    object f_inverse=None
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
        f (object): Transformation function for displayed ratings.
        f_inverse (object): Inverse transformation function for actual ratings.
    """
    cdef double expected_w = elo_formula(f_inverse(winner.rating), f_inverse(loser.rating))
    cdef double expected_l = elo_formula(f_inverse(loser.rating), f_inverse(winner.rating))

    # Transform to actual ratings, update, and transform back
    winner_actual = f_inverse(winner.rating) + k * (match_outcome - expected_w)
    loser_actual = f_inverse(loser.rating) + k * ((1 - match_outcome) - expected_l)

    winner.rating = f(winner_actual)
    loser.rating = f(loser_actual)

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
    double h=2.0,
    double w=200.0,
    double base_multiplier=0.5,
    object f=None,
    object f_inverse=None,
    double initial_rating=500
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
        f (object): Transformation function for displayed ratings.
        f_inverse (object): Inverse transformation function for actual ratings.
        initial_rating (double): The initial rating used as the static mean for f⁻¹ transformations.
    """
    cdef int len_participants
    cdef Player player_1, player_2
    cdef list next_participants
    cdef int round_number = 1
    cdef int total_rounds = int(np.ceil(np.log2(num_participants)))  # Total rounds in the tournament
    cdef double current_k  # Declare here at the top of the function
    rng = np.random.default_rng()

    # Compute mean Elo rating (static mean, equal to initial rating)
    cdef double mean_rating = initial_rating

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
                    mean_rating, f, f_inverse
                )
                next_participants.append(player_1)
            else:
                update_elo(
                    player_2, player_1, current_k, elo_formula, 1 - match_outcome,
                    mean_rating, f, f_inverse
                )
                next_participants.append(player_2)

        # Prepare for the next round
        participants = next_participants
        len_participants = len(participants)
        round_number += 1
