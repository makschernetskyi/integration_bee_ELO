cdef class Player:
    def __init__(self, str id, double initial_rating=500):
        self.id = id
        self.rating = initial_rating
        self.matches_played = 0
        self.rating_history = [initial_rating]
        self.streak = 0

    def update_rating(self, double new_rating):
        """
        Update the player's rating and store it in the history.
        """
        self.rating = new_rating
        self.rating_history.append(new_rating)

    def update_streak(self, won: bool):
        """
        Update the player's streak based on the outcome.

        Args:
            won (bool): True if the player won, False otherwise.
        """
        if won:
            self.streak = self.streak + 1  # Increment for wins
        else:
            self.streak = max(0, 0)  # Decrement for losses

    def __repr__(self):
        return f"Player({self.id}, Rating: {self.rating:.2f}, Matches: {self.matches_played})"