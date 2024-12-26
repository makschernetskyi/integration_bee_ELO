class Player:
    def __init__(self, id, initial_rating=500):
        self.id = id
        self.rating = initial_rating
        self.matches_played = 0

    def __repr__(self):
        return f"Player({self.id}, Rating: {self.rating:.2f}, Matches: {self.matches_played})"
