class Position:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class GameState:
    def __init__(self, board, colors, rocks, player0: Position, player1: Position, turn: int):
        self.board = board      # List[List[int]]
        self.colors = colors    # List[List[int]]
        self.rocks = rocks      # List[List[bool]]
        self.player0 = player0  # Position for player 0
        self.player1 = player1  # Position for player 1
        self.turn = turn        # Current turn (0 or 1)
