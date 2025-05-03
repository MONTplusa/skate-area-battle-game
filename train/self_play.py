import os
import pickle
import random
from game_state import GameState, Position

# Configuration
BOARD_SIZE = 20
N_GAMES = 100
MAX_MOVES = 1000

# Initialize a random game state
def init_game():
    N = BOARD_SIZE
    board = [[random.randint(0, 100) for _ in range(N)] for _ in range(N)]
    colors = [[-1] * N for _ in range(N)]
    rocks = [[False] * N for _ in range(N)]
    x0, y0 = random.randrange(N), random.randrange(N)
    x1, y1 = N - 1 - x0, N - 1 - y0
    return GameState(board, colors, rocks, Position(x0, y0), Position(x1, y1), 0)

# Generate legal moves for current player
def legal_moves(st):
    N = BOARD_SIZE
    p = st.turn
    pos = st.player0 if p == 0 else st.player1
    opp = st.player1 if p == 0 else st.player0
    moves = []
    for dir_idx, (dx, dy) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):
        x, y = pos.x, pos.y
        for step in range(1, N):
            x += dx
            y += dy
            if x < 0 or x >= N or y < 0 or y >= N:
                break
            if st.rocks[y][x] or (x == opp.x and y == opp.y):
                break
            moves.append((dir_idx, step))
    return moves

# Apply move to state
def apply_move(st, mv):
    board = [row[:] for row in st.board]
    colors = [row[:] for row in st.colors]
    rocks = [row[:] for row in st.rocks]
    p0, p1 = st.player0, st.player1
    turn = st.turn
    p = turn
    pos = p0 if p == 0 else p1
    # place rock
    rocks[pos.y][pos.x] = True
    dx, dy = [(1,0),(-1,0),(0,1),(0,-1)][mv[0]]
    x, y = pos.x, pos.y
    for _ in range(mv[1]):
        x += dx
        y += dy
        colors[y][x] = p
    if p == 0:
        p0 = Position(x, y)
    else:
        p1 = Position(x, y)
    return GameState(board, colors, rocks, p0, p1, 1 - p)

# Uniform policy over legal moves
def uniform_policy(moves):
    n = len(moves)
    if n == 0:
        return []
    return [1.0 / n] * n

# Final value normalized
def final_value(st):
    score0 = sum(st.board[y][x] for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if st.colors[y][x] == 0)
    score1 = sum(st.board[y][x] for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if st.colors[y][x] == 1)
    print(score0,score1)
    if score0-score1>0:
        return 1
    if score0-score1<0:
        return -1
    return float(score0 - score1) / (100 * BOARD_SIZE * BOARD_SIZE)

# Self-play data generation
all_data = []
for g in range(N_GAMES):
    st = init_game()
    history = []
    for _ in range(MAX_MOVES):
        moves = legal_moves(st)
        if not moves:
            break
        policy = uniform_policy(moves)
        history.append((st, policy))
        mv = random.choice(moves)
        st = apply_move(st, mv)
    val = final_value(st)
    for state_hist, pol in history:
        all_data.append((state_hist, pol, val))

# Save dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
with open(os.path.join(data_dir, 'dataset.pkl'), 'wb') as f:
    pickle.dump(all_data, f)

print(f"Generated {len(all_data)} data points from {N_GAMES} games.")
