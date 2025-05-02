package game

import (
	"math/rand"
	"time"
)

// GameState は盤面情報を保持
type GameState struct {
	N     int      // グリッドのサイズ
	Value [][]int  // マス上の点数
	Owner [][]int  // 塗り状態: -1=未,0/1=プレイヤー
	Rock  [][]bool // 岩石設置状態
	Px    [2]int   // プレイヤー X 座標
	Py    [2]int   // プレイヤー Y 座標
	Turn  int      // 現手番: 0 or 1
}

// Clone は GameState のディープコピーを返す
func (gs *GameState) Clone() *GameState {
	val := make([][]int, gs.N)
	own := make([][]int, gs.N)
	rock := make([][]bool, gs.N)
	for y := 0; y < gs.N; y++ {
		val[y] = make([]int, gs.N)
		own[y] = make([]int, gs.N)
		rock[y] = make([]bool, gs.N)
		copy(val[y], gs.Value[y])
		copy(own[y], gs.Owner[y])
		copy(rock[y], gs.Rock[y])
	}
	return &GameState{
		N:     gs.N,
		Value: val,
		Owner: own,
		Rock:  rock,
		Px:    gs.Px,
		Py:    gs.Py,
		Turn:  gs.Turn,
	}
}

// Move は一手の構造体
type Move struct {
	Player int
	FromX  int
	FromY  int
	ToX    int
	ToY    int
}

// LegalMoves は player の合法手を返す
func (gs *GameState) LegalMoves(player int) []Move {
	x0, y0 := gs.Px[player], gs.Py[player]
	oppX, oppY := gs.Px[1-player], gs.Py[1-player]
	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	var moves []Move
	for _, d := range dirs {
		dx, dy := d[0], d[1]
		for step := 1; step < gs.N; step++ {
			x := x0 + dx*step
			y := y0 + dy*step
			if x < 0 || x >= gs.N || y < 0 || y >= gs.N {
				break
			}
			if gs.Rock[y][x] || (x == oppX && y == oppY) {
				break
			}
			moves = append(moves, Move{player, x0, y0, x, y})
		}
	}
	return moves
}

func sign(d int) int {
	switch {
	case d < 0:
		return -1
	case d > 0:
		return 1
	default:
		return 0
	}
}

// ApplyMove は一手を適用し手番を交替
func (gs *GameState) ApplyMove(m Move) {
	p := m.Player
	x0, y0 := gs.Px[p], gs.Py[p]
	gs.Rock[y0][x0] = true
	dx, dy := sign(m.ToX-x0), sign(m.ToY-y0)
	x, y := x0, y0
	for {
		x += dx
		y += dy
		gs.Owner[y][x] = p
		if x == m.ToX && y == m.ToY {
			break
		}
	}
	gs.Px[p], gs.Py[p] = m.ToX, m.ToY
	gs.Turn = 1 - p
}

// GenerateInitialStates はランダム初期盤面を返す
func GenerateInitialStates(numSample, N int) []*GameState {
	rand.Seed(time.Now().UnixNano())
	samples := make([]*GameState, numSample)
	for i := 0; i < numSample; i++ {
		gs := &GameState{N: N, Value: make([][]int, N), Owner: make([][]int, N), Rock: make([][]bool, N)}
		for y := 0; y < N; y++ {
			gs.Value[y] = make([]int, N)
			gs.Owner[y] = make([]int, N)
			gs.Rock[y] = make([]bool, N)
			for x := 0; x < N; x++ {
				gs.Value[y][x] = rand.Intn(101)
				gs.Owner[y][x] = -1
			}
		}
		x := rand.Intn(N)
		y := rand.Intn(N)
		gs.Px[0], gs.Py[0] = x, y
		gs.Px[1], gs.Py[1] = N-1-x, N-1-y
		gs.Turn = 0
		samples[i] = gs
	}
	return samples
}
