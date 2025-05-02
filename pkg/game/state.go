package game

import (
	"math/rand"
)

// Position はプレイヤーの位置を表す構造体です
type Position struct {
	X, Y int
}

// Move は1手の移動を表す構造体です
type Move struct {
	FromX, FromY int // 移動元
	ToX, ToY     int // 移動先
	State        *GameState
}

// GameState はゲームの状態を表す構造体です
type GameState struct {
	Board   [][]int  // 各マスの数値
	Colors  [][]int  // 各マスの色（-1:なし、0:プレイヤー0、1:プレイヤー1）
	Rocks   [][]bool // 各マスの岩石の有無
	Player0 Position // プレイヤー0の位置
	Player1 Position // プレイヤー1の位置
	Turn    int      // 現在の手番（0または1）
}

// Clone はGameStateの深いコピーを返します
func (s *GameState) Clone() *GameState {
	newState := &GameState{
		Board:   make([][]int, len(s.Board)),
		Colors:  make([][]int, len(s.Colors)),
		Rocks:   make([][]bool, len(s.Rocks)),
		Player0: s.Player0,
		Player1: s.Player1,
		Turn:    s.Turn,
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
func GenerateInitialStates(numSamples, boardSize int) []*GameState {
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

		// プレイヤーの初期位置（点対称）
		state.Player0.X = rand.Intn(boardSize)
		state.Player0.Y = rand.Intn(boardSize)
		state.Player1.X = boardSize - 1 - state.Player0.X
		state.Player1.Y = boardSize - 1 - state.Player0.Y

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

		// その方向に移動可能な場合のみ合法手を追加
		if maxDist > 0 {
			// 最大距離の位置に移動する手を追加
			x := pos.X + dir[0]*maxDist
			y := pos.Y + dir[1]*maxDist

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
			for d := 0; d <= maxDist; d++ {
				pathX := pos.X + dir[0]*d
				pathY := pos.Y + dir[1]*d
				newState.Colors[pathY][pathX] = player
			}

			newState.Turn = 1 - player // 手番交代
			moves = append(moves, Move{pos.X, pos.Y, x, y, newState})
		}
	}

	return moves
}

// ApplyMove は指定された移動を適用して新しい盤面を返します
func (s *GameState) ApplyMove(move Move) *GameState {
	return move.State
}
