from abc import abstractmethod
import argparse
import json
import os
import random
import re

import numpy as np
from tensorflow import keras

# 定数
BOARD_SIZE = 20  # 盤面のサイズ
NUM_CHANNELS = 6  # 入力チャンネル数
NUM_SAMPLE = 10  # 初期盤面サンプル数

class Position:
    """プレイヤーの位置を表すクラス"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

class Move:
    """1手の移動を表すクラス"""
    def __init__(self, from_x, from_y, to_x, to_y, state=None):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.state = state

    def from_pos(self):
        return Position(self.from_x, self.from_y)

    def to_pos(self):
        return Position(self.to_x, self.to_y)

    def to_dict(self):
        return {
            "fromX": self.from_x,
            "fromY": self.from_y,
            "toX": self.to_x,
            "toY": self.to_y,
            "state": self.state.to_dict() if self.state else None
        }

class GameState:
    """ゲームの状態を表すクラス"""
    def __init__(self, board_size=BOARD_SIZE):
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.colors = [[-1 for _ in range(board_size)] for _ in range(board_size)]
        self.rocks = [[False for _ in range(board_size)] for _ in range(board_size)]
        self.player0 = Position(0, 0)
        self.player1 = Position(0, 0)
        self.turn = 0
        self.player0_name = "Player0"
        self.player1_name = "Player1"

    def clone(self):
        """GameStateの深いコピーを返す"""
        new_state = GameState(len(self.board))

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                new_state.board[y][x] = self.board[y][x]
                new_state.colors[y][x] = self.colors[y][x]
                new_state.rocks[y][x] = self.rocks[y][x]

        new_state.player0 = Position(self.player0.x, self.player0.y)
        new_state.player1 = Position(self.player1.x, self.player1.y)
        new_state.turn = self.turn
        new_state.player0_name = self.player0_name
        new_state.player1_name = self.player1_name

        return new_state

    def to_dict(self):
        """GameStateをJSON変換可能な辞書に変換"""
        return {
            "board": self.board,
            "colors": self.colors,
            "rocks": self.rocks,
            "player0": {"x": self.player0.x, "y": self.player0.y},
            "player1": {"x": self.player1.x, "y": self.player1.y},
            "turn": self.turn,
            "player0Name": self.player0_name,
            "player1Name": self.player1_name
        }

    def legal_moves(self, player):
        """指定されたプレイヤーの合法手をすべて列挙"""
        moves = []
        pos = self.player0 if player == 0 else self.player1

        # 上下左右の移動方向
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        board_size = len(self.board)

        # 各方向について移動可能な最大距離を計算
        for dir_x, dir_y in dirs:
            max_dist = 0
            for dist in range(1, board_size):
                x = pos.x + dir_x * dist
                y = pos.y + dir_y * dist

                # 盤面外または岩石にぶつかる場合は終了
                if x < 0 or x >= board_size or y < 0 or y >= board_size:
                    break
                if self.rocks[y][x]:
                    break

                # 相手プレイヤーの位置なら終了
                if player == 0 and x == self.player1.x and y == self.player1.y:
                    break
                if player == 1 and x == self.player0.x and y == self.player0.y:
                    break

                max_dist = dist

            # その方向に移動可能なすべての距離について合法手を追加
            for dist in range(1, max_dist + 1):
                x = pos.x + dir_x * dist
                y = pos.y + dir_y * dist

                new_state = self.clone()
                if player == 0:
                    new_state.player0.x = x
                    new_state.player0.y = y
                else:
                    new_state.player1.x = x
                    new_state.player1.y = y

                # 移動元を岩石に変更
                new_state.rocks[pos.y][pos.x] = True

                # 移動経路を自分の色で塗る
                for d in range(dist + 1):
                    path_x = pos.x + dir_x * d
                    path_y = pos.y + dir_y * d
                    new_state.colors[path_y][path_x] = player

                new_state.turn = 1 - player  # 手番交代
                moves.append(Move(pos.x, pos.y, x, y, new_state))

        # 合法手をシャッフル
        random.shuffle(moves)
        return moves

def generate_initial_states(num_samples, board_size):
    """初期盤面をnum_samples個生成"""
    states = []

    for _ in range(num_samples):
        state = GameState(board_size)

        # 盤面の初期化
        for y in range(board_size):
            for x in range(board_size):
                state.board[y][x] = random.randint(0, 100)  # 0-100の値
                state.colors[y][x] = -1  # 未着色

        # プレイヤーの初期位置をランダムに設定
        while True:
            state.player0.x = random.randint(0, board_size - 1)
            state.player0.y = random.randint(0, board_size - 1)
            state.player1.x = board_size - 1 - state.player0.x
            state.player1.y = board_size - 1 - state.player0.y

            # プレイヤーが十分離れているか確認
            dx = state.player1.x - state.player0.x
            dy = state.player1.y - state.player0.y
            if dx * dx + dy * dy >= board_size:  # 一定以上の距離を確保
                break

        state.turn = 0  # 先手から開始
        states.append(state)

    return states

def create_input_data(state):
    """
    GameState型のデータから入力データを作成

    Args:
        state: GameState型のデータ

    Returns:
        input_data: (BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS) の形状の入力データ
    """
    # 入力データの初期化
    input_data = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)

    # 次のプレイヤーが1となるように入力を作成する
    # つまり、next_player == 0の場合はプレイヤー0とプレイヤー1を逆に読み替える必要がある
    next_player = state.turn
    player0 = 1 - next_player
    player1 = next_player

    # ボードの最大値を取得して正規化
    board_max = 0
    for row in state.board:
        row_max = max(row)
        if row_max > board_max:
            board_max = row_max

    # チャンネル0: ボードの数値を0-1に正規化
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board_max > 0:
                input_data[y, x, 0] = state.board[y][x] / board_max

    # チャンネル1: プレイヤー0の色
    # チャンネル2: プレイヤー1の色
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = state.colors[y][x]
            if color == player0:
                input_data[y, x, 1] = 1.0
            elif color == player1:
                input_data[y, x, 2] = 1.0

    # チャンネル3: 岩の位置
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if state.rocks[y][x]:
                input_data[y, x, 3] = 1.0

    # チャンネル4: プレイヤー0の位置
    player0_pos = state.player0 if player0 == 0 else state.player1
    input_data[player0_pos.y, player0_pos.x, 4] = 1.0

    # チャンネル5: プレイヤー1の位置
    player1_pos = state.player1 if player1 == 1 else state.player0
    input_data[player1_pos.y, player1_pos.x, 5] = 1.0

    return input_data

# モデルは既に存在することを前提とするため、create_model関数は削除

def load_model(model_path):
    """Kerasモデルを読み込む"""
    # 拡張子が.kerasでない場合は追加
    if not model_path.endswith('.keras'):
        model_path = f"{model_path}.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    try:
        print(f"Kerasモデルを読み込みます: {model_path}")
        model = keras.models.load_model(model_path)
        print(f"Kerasモデルを正常に読み込みました: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"モデルの読み込みに失敗しました: {e}")

class AI:
    """AIの基底クラス"""
    @abstractmethod
    def select_board(self, states):
        """初期盤面を選択"""
        ...

    @abstractmethod
    def select_turn(self, states):
        """先手/後手を選択"""
        ...

    @abstractmethod
    def evaluate(self, state, player):
        """状態を評価"""
        ...

    def batch_evaluate(self, states, player):
        """複数の状態を評価"""
        evals = []
        for state in states:
            evals.append(self.evaluate(state, player))
        return evals


class NeuralNetworkAI(AI):
    """ニューラルネットワークを使用したAI"""
    def __init__(self, model, name="NeuralNetworkAI"):
        self.model = model
        self.name = name

    def select_board(self, states):
        """初期盤面を選択"""
        # ランダムに選択
        return random.randint(0, len(states) - 1)

    def select_turn(self, states):
        """先手/後手を選択"""
        # ランダムに選択
        return random.randint(0, 1)

    def evaluate(self, state, player):
        """状態を評価"""
        # 入力データを作成
        input_data = create_input_data(state)
        input_data = np.expand_dims(input_data, axis=0)  # バッチ次元を追加

        # Kerasモデルで評価
        value = self.model.predict(input_data, verbose=0)[0][0]

        return value

    def batch_evaluate(self, states, player):
        """複数の状態を評価"""
        input_data = np.zeros((len(states), BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)

        for i, state in enumerate(states):
            input_data[i] = create_input_data(state)

        # Kerasモデルで評価
        values = self.model.predict(input_data, verbose=0)[:, 0]

        return values

class RandomAI(AI):
    """ランダムに手を選択するAI"""
    def __init__(self, model, name="RandomAI"):
        self.model = model
        self.name = name

    def select_board(self, states):
        """初期盤面を選択"""
        # ランダムに選択
        return random.randint(0, len(states) - 1)

    def select_turn(self, states):
        """先手/後手を選択"""
        # ランダムに選択
        return random.randint(0, 1)

    def evaluate(self, state, player):
        """状態を評価"""
        # ランダムな評価値を返す
        return random.uniform(-1.0, 1.0)

class GameRunner:
    """対戦を管理するクラス"""
    def __init__(self, agent0, agent1, *, show_log=False):
        self.agents = [agent0, agent1]
        self.show_log = show_log

    def run(self):
        """対戦を実行して結果を返す"""
        # 1) 初期盤面サンプル生成 & 盤面選択
        states = generate_initial_states(NUM_SAMPLE, BOARD_SIZE)
        chooser = random.randint(0, 1)
        other = 1 - chooser
        idx = self.agents[other].select_board(states)
        state = states[idx].clone()

        # 2) 先後選択
        first = self.agents[chooser].select_turn(states)
        state.turn = (first + chooser) % 2
        if first != chooser:
            state.player0, state.player1 = state.player1, state.player0

        # 3) AIの名前を設定
        state.player0_name = self.agents[0].name
        state.player1_name = self.agents[1].name

        # 4) 結果オブジェクトの初期化
        result = {
            "initialState": state.to_dict(),
            "moves": [],
            "finalState": None
        }

        # 5) ゲームループ
        number = 0
        skips = 0
        while skips < 2:
            player = state.turn

            moves = state.legal_moves(player)
            random.shuffle(moves)  # 手をランダムにシャッフル

            if len(moves) == 0:
                if self.show_log:
                    print(f"プレイヤー{player}の合法手がありません。スキップします。")
                skips += 1
                state.turn = 1 - player
                continue

            skips = 0
            best_eval = float('-inf')
            best_move = None

            evals = self.agents[player].batch_evaluate([move.state for move in moves], player)
            for move, eval in zip(moves, evals):
                if self.show_log:
                    print(f"| プレイヤー{player}の手: ({move.from_y}, {move.from_x})) -> ({move.to_y}, {move.to_x})), 評価値: {eval}")
                if eval > best_eval:
                    best_eval = eval
                    best_move = move

            # 有効な手が見つかった場合のみ進める
            if best_move is not None:
                if self.show_log:
                    print(f"-> #{number:<3} プレイヤー{player}の最良手 ({best_eval}): ({best_move.from_y}, {best_move.from_x})) -> ({best_move.to_y}, {best_move.to_x}))")
                state = best_move.state
                result["moves"].append(best_move.to_dict())
                number += 1
            else:
                # 有効な手がない場合はスキップ
                if self.show_log:
                    print(f"プレイヤー{player}の有効な手が見つかりません。スキップします。")
                state.turn = 1 - player
                skips += 1

        # 最終状態を保存
        result["finalState"] = state.to_dict()

        return result

def find_next_sequence_number(directory, prefix):
    """指定されたディレクトリ内の同じプレフィックスを持つファイルの最大連番を取得する"""
    # ディレクトリが存在しない場合は0を返す
    if not os.path.exists(directory):
        return 0

    # プレフィックス_NNNNN.json の形式にマッチする正規表現
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d{{5}})\.json$')
    max_seq = 0

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            try:
                seq = int(match.group(1))
                if seq > max_seq:
                    max_seq = seq
            except ValueError:
                continue

    return max_seq + 1

def self_play(model_path, output_dir, num_games, prefix, show_log):
    """自己対戦を実行"""
    # モデルの読み込み
    model = load_model(model_path)

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # AIの作成
    p0_ai = NeuralNetworkAI(model, f"NeuralNetworkAI ({model_path})")
    p1_ai = NeuralNetworkAI(model, f"NeuralNetworkAI ({model_path})")

    # 既存ファイルの最大連番を取得
    start_seq = find_next_sequence_number(output_dir, prefix)
    print(f"連番 {start_seq:05d} から開始します")

    # 自己対戦の実行
    wins = [0, 0]
    for i in range(num_games):
        # 自己対戦
        runner = GameRunner(p0_ai, p1_ai, show_log=show_log)
        result = runner.run()

        # 勝者を判定
        scores = [0, 0]
        state = result["finalState"]
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                color = state["colors"][y][x]
                if color == -1:
                    continue
                scores[color] += state["board"][y][x]

        print(f"対戦 #{i}: スコア - {state['player0Name']}: {scores[0]}, {state['player1Name']}: {scores[1]}")
        if scores[0] > scores[1]:
            wins[0]+=1
        elif scores[0] < scores[1]:
            wins[1] += 1
        # input("Enterキーを押して続行...")

        # 結果の保存
        seq_num = start_seq + i
        filename = f"{output_dir}/{prefix}_{seq_num:05d}.json"
        with open(filename, 'w') as f:
            json.dump(result, f)

    draw = num_games - sum(wins)
    print(f"勝利数: P0: {wins[0]}, P1: {wins[1]}, 引き分け: {draw}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='自己対戦によるトレーニングデータ生成')
    parser.add_argument('--model', type=str, required=True, help='使用するKerasモデルのパス（.keras形式）')
    parser.add_argument('--output', type=str, default='output', help='対戦結果の保存先ディレクトリ')
    parser.add_argument('--games', type=int, default=10, help='自己対戦の回数')
    parser.add_argument('--prefix', type=str, required=True, help='出力ファイル名のプレフィックス')
    parser.add_argument('--show-log', action='store_true', help='ログを表示する')
    args = parser.parse_args()

    # 自己対戦の実行
    self_play(args.model, args.output, args.games, args.prefix, args.show_log)

if __name__ == '__main__':
    main()
