# cython_run_tournament.pyx
import numpy as np
cimport numpy as cnp

from player cimport Player  # Import Player class from an external file


cdef void update_elo(Player winner, Player loser, double k, object elo_formula):
    """
    Update Elo ratings for the winner and loser based on the Elo formula.

    Args:
        winner (Player): The player who won the match.
        loser (Player): The player who lost the match.
        k (double): The K-factor.
        elo_formula (callable): Function to calculate expected score.
    """
    cdef double expected_w = elo_formula(winner.rating, loser.rating)
    cdef double expected_l = elo_formula(loser.rating, winner.rating)

    winner.update_rating(winner.rating + k * (1 - expected_w))
    loser.update_rating(loser.rating + k * (0 - expected_l))

    winner.matches_played += 1
    loser.matches_played += 1

def run_tournament(list players, double k, int p, dict history, object elo_formula):
    """
    Simulate a tournament with p players randomly chosen.

    Args:
        players (List[Player]): List of players.
        k (float): K-factor for Elo updates.
        p (int): Number of participants in the tournament.
        history (dict): Preprocessed match data.
        elo_formula (Callable): Function to calculate expected score.
    """
    cdef int len_participants
    cdef Player player_1, player_2
    cdef list next_participants
    rng = np.random.default_rng()

    # Randomly select participants
    participants = rng.choice(players, size=p, replace=False).tolist()
    len_participants = len(participants)

    while len_participants > 1:
        next_participants = []
        for i in range(0, len_participants, 2):
            player_1 = participants[i]
            player_2 = participants[i + 1]

            # Lookup historical outcome
            if (player_1.id, player_2.id) in history:
                is_1_winner = history[(player_1.id, player_2.id)]['Winner_Score'] > history[(player_1.id, player_2.id)]['Loser_Score']
            else:
                is_1_winner = rng.integers(2) == 0  # Random outcome if no history exists

            if is_1_winner:
                update_elo(player_1, player_2, k, elo_formula)
                next_participants.append(player_1)
            else:
                update_elo(player_2, player_1, k, elo_formula)
                next_participants.append(player_2)

        participants = next_participants
        len_participants = len(participants)