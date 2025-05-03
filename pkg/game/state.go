package game

import (
	"math/rand"
	"time"
)

func init() {
	// 乱数生成器の初期化
	rand.Seed(time.Now().UnixNano())
}

// Position はプレイヤーの位置を表す構造体です
type Position struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// Move は1手の移動を表す構造体です
type Move struct {
	FromX int        `json:"fromX"` // 移動元X座標
	FromY int        `json:"fromY"` // 移動元Y座標
	ToX   int        `json:"toX"`   // 移動先X座標
	ToY   int        `json:"toY"`   // 移動先Y座標
	State *GameState `json:"state"` // 移動後の状態
}

func (m *Move) From() Position {
	return Position{X: m.FromX, Y: m.FromY}
}

func (m *Move) To() Position {
	return Position{X: m.ToX, Y: m.ToY}
}

// GameState はゲームの状態を表す構造体です
type GameState struct {
	Board       [][]int  `json:"board"`       // 各マスの数値
	Colors      [][]int  `json:"colors"`      // 各マスの色（-1:なし、0:プレイヤー0、1:プレイヤー1）
	Rocks       [][]bool `json:"rocks"`       // 各マスの岩石の有無
	Player0     Position `json:"player0"`     // プレイヤー0の位置
	Player1     Position `json:"player1"`     // プレイヤー1の位置
	Turn        int      `json:"turn"`        // 現在の手番（0または1）
	Player0Name string   `json:"player0Name"` // プレイヤー0の名前
	Player1Name string   `json:"player1Name"` // プレイヤー1の名前
}

// Clone はGameStateの深いコピーを返します
func (s *GameState) Clone() *GameState {
	newState := &GameState{
		Board:       make([][]int, len(s.Board)),
		Colors:      make([][]int, len(s.Colors)),
		Rocks:       make([][]bool, len(s.Rocks)),
		Player0:     s.Player0,
		Player1:     s.Player1,
		Turn:        s.Turn,
		Player0Name: s.Player0Name,
		Player1Name: s.Player1Name,
	}

	for i := range s.Board {
		newState.Board[i] = make([]int, len(s.Board[i]))
		copy(newState.Board[i], s.Board[i])

		newState.Colors[i] = make([]int, len(s.Colors[i]))
		copy(newState.Colors[i], s.Colors[i])

		newState.Rocks[i] = make([]bool, len(s.Rocks[i]))
		copy(newState.Rocks[i], s.Rocks[i])
	}

	return newState
}

// GenerateInitialStates は初期盤面をnumSamples個生成します
// GenerateInitialStates は初期盤面をnumSamples個生成します
func GenerateInitialStates(numSamples, boardSize int) []*GameState {
	// シード値をリセット（重複防止）
	rand.Seed(time.Now().UnixNano())
	states := make([]*GameState, numSamples)

	for i := 0; i < numSamples; i++ {
		state := &GameState{
			Board:  make([][]int, boardSize),
			Colors: make([][]int, boardSize),
			Rocks:  make([][]bool, boardSize),
		}

		// 盤面の初期化
		for y := 0; y < boardSize; y++ {
			state.Board[y] = make([]int, boardSize)
			state.Colors[y] = make([]int, boardSize)
			state.Rocks[y] = make([]bool, boardSize)

			for x := 0; x < boardSize; x++ {
				state.Board[y][x] = rand.Intn(101) // 0-100の値
				state.Colors[y][x] = -1            // 未着色
			}
		}

		// プレイヤーの初期位置をランダムに設定
		for {
			state.Player0.X = rand.Intn(boardSize)
			state.Player0.Y = rand.Intn(boardSize)
			state.Player1.X = boardSize - 1 - state.Player0.X
			state.Player1.Y = boardSize - 1 - state.Player0.Y

			// プレイヤーが十分離れているか確認
			dx := state.Player1.X - state.Player0.X
			dy := state.Player1.Y - state.Player0.Y
			if dx*dx+dy*dy >= boardSize { // 一定以上の距離を確保
				break
			}
		}

		state.Turn = 0 // 先手から開始
		states[i] = state
	}

	return states
}

// LegalMoves は指定されたプレイヤーの合法手をすべて列挙します
func (s *GameState) LegalMoves(player int) []Move {
	var moves []Move
	pos := s.Player0
	if player == 1 {
		pos = s.Player1
	}

	// 上下左右の移動方向
	dirs := [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	boardSize := len(s.Board)

	// 各方向について移動可能な最大距離を計算
	for _, dir := range dirs {
		maxDist := 0
		for dist := 1; dist < boardSize; dist++ {
			x := pos.X + dir[0]*dist
			y := pos.Y + dir[1]*dist

			// 盤面外または岩石にぶつかる場合は終了
			if x < 0 || x >= boardSize || y < 0 || y >= boardSize {
				break
			}
			if s.Rocks[y][x] {
				break
			}

			// 相手プレイヤーの位置なら終了
			if player == 0 && x == s.Player1.X && y == s.Player1.Y {
				break
			}
			if player == 1 && x == s.Player0.X && y == s.Player0.Y {
				break
			}

			maxDist = dist
		}

		// その方向に移動可能なすべての距離について合法手を追加
		for dist := 1; dist <= maxDist; dist++ {
			x := pos.X + dir[0]*dist
			y := pos.Y + dir[1]*dist

			newState := s.Clone()
			if player == 0 {
				newState.Player0.X = x
				newState.Player0.Y = y
			} else {
				newState.Player1.X = x
				newState.Player1.Y = y
			}

			// 移動元を岩石に変更
			newState.Rocks[pos.Y][pos.X] = true

			// 移動経路を自分の色で塗る
			for d := 0; d <= dist; d++ {
				pathX := pos.X + dir[0]*d
				pathY := pos.Y + dir[1]*d
				newState.Colors[pathY][pathX] = player
			}

			newState.Turn = 1 - player // 手番交代
			moves = append(moves, Move{pos.X, pos.Y, x, y, newState})
		}
	}

	// 合法手をシャッフル
	rand.Shuffle(len(moves), func(i, j int) {
		moves[i], moves[j] = moves[j], moves[i]
	})
	return moves
}

// ApplyMove は指定された移動を適用して新しい盤面を返します
func (s *GameState) ApplyMove(move Move) *GameState {
	return move.State
}
