import pandas as pd
from player import Player

import pandas as pd


def load_and_transform_matches(file_path: str, start_year: int = 1965, output_path: str = "processed_matches.csv", v: int = 5):
    """
    Load match data from a CSV file, filter it, and save processed match history.

    Args:
        file_path (str): Path to the original CSV file.
        start_year (int): Only include matches starting from this year.
        output_path (str): Path to save the processed match history CSV.
        v (int): Maximum score for best-of-v matches. Only keep entries where scores are <= v//2 + 1.

    Returns:
        None
    """
    # Load CSV data
    df = pd.read_csv(file_path, delimiter=",")

    # Define max allowed score
    max_allowed_score = v // 2 + 1

    # Filter data: Exclude draws, matches before start_year, and scores exceeding max_allowed_score
    df = df[(df['home_score'] != df['away_score']) &
            (pd.to_datetime(df['date']).dt.year >= start_year) &
            (df['home_score'] <= max_allowed_score) &
            (df['away_score'] <= max_allowed_score)]

    # Transform matches into Winner/Loser format
    match_history = []
    for _, row in df.iterrows():
        if row['home_score'] > row['away_score']:
            winner_name = f"plyr. {row['home_team']}"
            loser_name = f"plyr. {row['away_team']}"
            winner_score, loser_score = row['home_score'], row['away_score']
        else:
            winner_name = f"plyr. {row['away_team']}"
            loser_name = f"plyr. {row['home_team']}"
            winner_score, loser_score = row['away_score'], row['home_score']

        match_history.append((winner_name, loser_name, winner_score, loser_score))

    # Save the processed match history to a CSV file
    processed_df = pd.DataFrame(match_history, columns=["Winner", "Loser", "Winner_Score", "Loser_Score"])
    processed_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    load_and_transform_matches(file_path="./results.csv")
