# Elo Simulation System

This project implements an Elo-based rating system simulation designed to explore suitable configurations for ranking players in the Integration Bee competition. It provides tools to simulate single-elimination tournaments, calculate dynamic player ratings, and periodically decay ratings toward the mean to reflect long-term stability.

## Introduction

The Integration Bee is a mathematical competition requiring a precise, dynamic, and fair ranking system to evaluate participants' skill levels. This project adapts the Elo rating system to address the unique challenges posed by such competitions, ensuring meaningful updates to player rankings after each event while maintaining a balance between long-term stability and responsiveness to recent performances.

## The Elo Rating System

Elo is a method for calculating the relative skill levels of players in competitive games. Originally developed for chess, it assigns a numerical rating to each player, dynamically updated after every match based on expected and actual outcomes. The expected score uses a logistic curve to model the probability of one player defeating another, where the difference in ratings determines the skewness of the probability. Matches between equally rated players have a 50% expected score for each, while a significant rating gap results in a much higher expectation for the stronger player.

The update mechanism compares this expected score with the actual outcome and adjusts ratings proportionally. A parameter called the K-factor determines the sensitivity of the rating system. Higher K-factors allow ratings to shift more dramatically after a single match, while lower values prioritize stability over time.

## Challenges in the Integration Bee Context

### Infrequent Competitions

Integration Bee events occur infrequently, which makes every match disproportionately impactful compared to systems where games happen regularly. This requires careful calibration of the K-factor. While a higher K-factor ensures that each match meaningfully affects the player's rating, overly large changes can destabilize the system. To address this, we use a moderately high K-factor while keeping the starting ratings at a controlled level, ensuring that changes are impactful yet stable.

### Complexity of Rounds

The complexity of integrals varies significantly between rounds. Early rounds often feature straightforward problems, while later rounds demand more advanced techniques. This discrepancy in difficulty necessitates weighting matches differently based on the round. Higher-stakes matches, such as semifinals and finals, have a greater impact on ratings by scaling the K-factor to reflect the significance of the round. This adjustment ensures that the ratings better represent the players' abilities in solving progressively difficult integrals.

### Infrequent Participation

Some players may compete only sporadically, making their ratings less reliable over time. To address this, the system incorporates a decay mechanism that periodically shifts ratings closer to the mean. This decay ensures that inactive players do not retain inflated ratings indefinitely, while active players can continually adjust their ratings based on recent performance. The decay rate is proportional to the gap between a player's rating and the mean rating of all participants, balancing fairness and accuracy.

### Handling Disqualifications

Disqualifications present a unique challenge as they do not involve an actual match. Penalizing a disqualified player heavily could unfairly distort their rating, while ignoring the event entirely might fail to reflect the disqualification's impact. The system applies minimal adjustments in these cases, ensuring that the absence of a match does not disproportionately affect rankings while maintaining the integrity of the competition.

### Margin-Based Outcomes

The magnitude of victory or defeat can provide additional nuance to rating adjustments. By incorporating margin-based outcomes, the system can reward dominant performances more heavily and reduce the impact of narrow losses. This approach refines the traditional win/loss binary outcome, making the updates more representative of actual match dynamics.

### Addressing Bias in Simulations

One of the challenges in simulating an Elo system for the Integration Bee is the potential bias introduced when generating random match outcomes. To overcome this, we integrated real-world data from football matches as a proxy. By transforming football teams into players and match results into outcomes, we achieved a realistic distribution of win probabilities and skill levels. This approach ensured that the simulations reflected genuine competitive dynamics, avoiding artificially generated randomness that could skew the results. This data-driven approach also allows for better validation of the system's parameters and adjustments.

## Implementation

The simulation is implemented in Python and leverages the numpy and pandas libraries for numerical computations and data management. Players and match data are loaded from a CSV file, enabling seamless integration of historical data. The system simulates single-elimination tournaments with configurable parameters to allow experimentation with different configurations and scenarios.

Configurations 

The simulation supports the following configurable parameters: 

- Initial Ratings: All players start with the same base rating, typically 400.

- K-Factor: Controls the sensitivity of rating changes. Higher values cause ratings to change more significantly after each match.

- Decay Interval: Specifies how often ratings decay toward the mean.

- Decay Factor: Determines the proportion of decay applied at each interval.

- Round-Dependent Weighting: Allows scaling the K-factor based on the round's significance.

- Steps to Run the Simulation

### Prepare the Environment: 
Clone the repository and install the required dependencies:

`git clone <repository_url>` 
`cd <repository_name>` 
`pip install numpy pandas` 

### Configure Simulation Parameters:
Edit the simulate_elo function in the script to adjust parameters such as the number of players (m), participants per tournament (p), number of tournaments (n), K-factor, and decay settings.

### Run the Simulation:
Execute the script to simulate tournaments and update player ratings:

`python elo_simulation.py`

### Analyze Results:
The simulation generates an output file named player_results.txt, containing final ratings and match counts for all players.

## Applications

This project is specifically designed for the Integration Bee but has broader applications in ranking systems for competitions. By simulating tournaments, users can evaluate the effectiveness of various configurations and refine the system for accuracy and fairness. The flexibility of the code makes it suitable for adapting to different competitive scenarios beyond mathematics.

## Future Directions

Future improvements to the system will focus on integrating machine learning techniques to optimize the parameters automatically, ensuring that the system adapts to data-driven insights. Additionally, the code requires optimization to improve its efficiency, as current simulations can be slow for larger datasets. Enhancing performance will allow for more extensive experimentation and analysis.

## License

This project is licensed under the MIT License. For more details, refer to the LICENSE file in the repository.
