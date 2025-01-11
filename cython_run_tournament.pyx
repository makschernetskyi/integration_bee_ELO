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

cdef void update_elo(Player winner, Player loser, double k, object elo_formula, double match_outcome):
    """
    Update Elo ratings for the winner and loser based on the Elo formula.

    Args:
        winner (Player): The player who won the match.
        loser (Player): The player who lost the match.
        k (double): The K-factor.
        elo_formula (callable): Function to calculate expected score.
        match_outcome (double): Scaled outcome (range [0, 1]) based on match results.
    """
    cdef double expected_w = elo_formula(winner.rating, loser.rating)
    cdef double expected_l = elo_formula(loser.rating, winner.rating)

    # Update ratings using the scaled match outcome
    winner.update_rating(winner.rating + k * (match_outcome - expected_w))
    loser.update_rating(loser.rating + k * ((1 - match_outcome) - expected_l))

    # Increment matches played
    winner.matches_played += 1
    loser.matches_played += 1


def run_tournament(
    list players,
    int p,
    dict history,
    object elo_formula,
    int v,
    str k_scaling="sqrt",
    double k=32,
    double k_min=100,
    double k_max=400,
    list custom_k=None
):
    """
    Simulate a tournament with p players randomly chosen, supporting dynamic and static K.

    Args:
        players (List[Player]): List of players.
        k (float): Base K-factor for Elo updates (used for static K).
        p (int): Number of participants in the tournament.
        history (dict): Preprocessed match data.
        elo_formula (Callable): Function to calculate expected score.
        v (int): Number of games in the "best-of-v" format.
        k_scaling (str): Scaling method for dynamic K ("sqrt", "log", "linear", "static", "custom").
        k_min (float): Minimum K-factor (for the first round).
        k_max (float): Maximum K-factor (for the final round).
        custom_k (list): Custom list of K values, one for each round (only used if k_scaling="custom").
    """
    cdef int len_participants
    cdef Player player_1, player_2
    cdef list next_participants
    cdef int round_number = 1
    cdef int total_rounds = int(np.ceil(np.log2(p)))  # Total rounds in the tournament
    cdef double current_k  # Declare here at the top of the function
    rng = np.random.default_rng()

    # Randomly select participants
    participants = rng.choice(players, size=p, replace=False).tolist()
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
                match_outcome = winner_score / v  # Scaled outcome
            else:
                # Random outcome if no history exists
                is_1_winner = rng.integers(2) == 0
                match_outcome = rng.random()  # Simulate a random outcome scale

            # Update Elo ratings based on winner
            if is_1_winner:
                update_elo(player_1, player_2, current_k, elo_formula, match_outcome)
                next_participants.append(player_1)
            else:
                update_elo(player_2, player_1, current_k, elo_formula, 1 - match_outcome)
                next_participants.append(player_2)

        # Prepare for the next round
        participants = next_participants
        len_participants = len(participants)
        round_number += 1



