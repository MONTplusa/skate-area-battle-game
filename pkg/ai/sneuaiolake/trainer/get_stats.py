from collections import defaultdict
import json
import sys


results = []
for file in sys.argv[1:]:
    with open(file) as f:
        results.append(json.load(f))

wins = defaultdict(int)
for result in results:
    state = result['finalState']
    board = state['board']
    colors = state['colors']

    scores = [0, 0, 0]
    for i in range(len(board)):
        for j in range(len(board[i])):
            scores[colors[i][j]] += board[i][j]

    winner = None
    if scores[0] > scores[1]:
        winner = state["player0Name"]
    elif scores[0] < scores[1]:
        winner = state["player1Name"]

    if winner:
        wins[winner] += 1

total_games = len(results)
draws = total_games - sum(wins.values())
for winner, count in wins.items():
    print(f"{winner} wins: {count}")
print(f"Draws: {draws}")
print(f"Total games: {total_games}")
