import pandas as pd
from player import Player

def load_and_transform_matches(file_path: str, start_year: int = 1965, output_path: str = "processed_matches.csv"):
    """
    Load match data from a CSV file, create Player objects, and save processed match history.

    Args:
        file_path (str): Path to the original CSV file.
        start_year (int): Only include matches starting from this year.
        output_path (str): Path to save the processed match history CSV.

    Returns:
        players (dict): Dictionary of player names to Player objects.
        match_history (list): List of tuples (winner, loser, winner_score, loser_score).
    """
    # Load CSV data
    df = pd.read_csv(file_path, delimiter=",")


    # Filter data: Exclude draws and matches before the start_year
    df = df[(df['home_score'] != df['away_score']) & (pd.to_datetime(df['date']).dt.year >= start_year)]

    # Initialize players and match history
    players = {}
    match_history = []

    # Transform matches
    for _, row in df.iterrows():
        # Determine winner and loser
        if row['home_score'] > row['away_score']:
            winner_name = f"plyr. {row['home_team']}"
            loser_name = f"plyr. {row['away_team']}"
            winner_score, loser_score = row['home_score'], row['away_score']
        else:
            winner_name = f"plyr. {row['away_team']}"
            loser_name = f"plyr. {row['home_team']}"
            winner_score, loser_score = row['away_score'], row['home_score']

        # Add players to the dictionary if not already added
        if winner_name not in players:
            players[winner_name] = Player(winner_name)
        if loser_name not in players:
            players[loser_name] = Player(loser_name)

        # Record the match result with scores
        match_history.append((winner_name, loser_name, winner_score, loser_score))

    # Save processed match history to a CSV file
    processed_df = pd.DataFrame(match_history, columns=["Winner", "Loser", "Winner_Score", "Loser_Score"])
    processed_df.to_csv(output_path, index=False)

    return players, match_history



if __name__ == "__main__":
    load_and_transform_matches(file_path="./results.csv")
