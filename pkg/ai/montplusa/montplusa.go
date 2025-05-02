package montplusa

import (
	"math"
	"sort"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

const (
	// 争奪領域における重み (先手優位を若干織り込む)
	contestedBias = 0.55
	// 高価値セルボーナス用パラメータ
	highValueK     = 5
	highValueBonus = 0.1
	// 評価関数拡張用重み
	mobWeight  = 0.05 // モビリティ(可動性)差分重み
	turnWeight = 0.1  // 手番優位バイアス
)

// MontplusaAI は Territory ベースかつ強化評価を行う AI です。
type MontplusaAI struct{}

// New は MontplusaAI のコンストラクタです。
func New() *MontplusaAI { return &MontplusaAI{} }

// SelectBoard は複数初期盤面から最適なものを選択します。
func (ai *MontplusaAI) SelectBoard(states []*game.GameState) int {
	bestIdx := 0
	bestScore := math.Inf(-1)
	for i, st := range states {
		s0 := ai.Evaluate(st, 0)
		s1 := ai.Evaluate(st, 1)
		score := math.Max(s0, s1)
		if score > bestScore {
			bestScore, bestIdx = score, i
		}
	}
	return bestIdx
}

// SelectTurn は先手(0)か後手(1)かを選択します。
func (ai *MontplusaAI) SelectTurn(states []*game.GameState) int {
	st := states[0]
	s0 := ai.Evaluate(st, 0)
	s1 := ai.Evaluate(st, 1)
	if s0 >= s1 {
		return 0
	}
	return 1
}

// Evaluate は Territory ベースの予測スコア差に高価値セルボーナス
// とモビリティ、手番バイアスを加えた評価値を返します。
func (ai *MontplusaAI) Evaluate(st *game.GameState, player int) float64 {
	opp := 1 - player
	// 領土予測
	myScore, oppScore := territoryEstimation(st, player)
	// 高価値ボーナス
	bonus := highValueClusterBonus(st, player)
	// モビリティ差分
	mobDiff := float64(mobilityCount(st, getPos(st, player), getPos(st, opp)) -
		mobilityCount(st, getPos(st, opp), getPos(st, player)))
	// 手番優位バイアス
	turnBias := 0.0
	if st.Turn == player {
		turnBias = turnWeight
	} else {
		turnBias = -turnWeight
	}
	// 総合評価
	return (myScore - oppScore) + bonus + mobWeight*mobDiff + turnBias
}

// territoryEstimation は各マスの最短スライド手数から領土を推定し、
// 自分(player)視点での現在領域と予測獲得領域のスコアを返します。
func territoryEstimation(st *game.GameState, player int) (myScore, oppScore float64) {
	nop := 1 - player
	// 距離マップ計算
	distMe := computeSlideDistances(st, getPos(st, player), getPos(st, nop))
	distOp := computeSlideDistances(st, getPos(st, nop), getPos(st, player))

	N := len(st.Board)
	const INF = math.MaxInt32
	for y := 0; y < N; y++ {
		for x := 0; x < N; x++ {
			v := float64(st.Board[y][x])
			switch c := st.Colors[y][x]; {
			case c == player:
				myScore += v
			case c == nop:
				oppScore += v
			default:
				dm, do := distMe[y][x], distOp[y][x]
				switch {
				case dm < do:
					myScore += v
				case do < dm:
					oppScore += v
				case dm < INF: // 同距離: contestedBias で分配
					myScore += contestedBias * v
					oppScore += (1 - contestedBias) * v
				}
			}
		}
	}
	return
}

// highValueClusterBonus は上位 highValueK マスのうち、自分が領土と予測できるセルに
// highValueBonus 倍のボーナスを返します。
func highValueClusterBonus(st *game.GameState, player int) float64 {
	opp := 1 - player
	type cell struct {
		v    float64
		x, y int
	}
	cells := make([]cell, 0, len(st.Board)*len(st.Board))
	for y := range st.Board {
		for x, val := range st.Board[y] {
			if st.Colors[y][x] == -1 {
				cells = append(cells, cell{float64(val), x, y})
			}
		}
	}
	// 値順にソート
	sort.Slice(cells, func(i, j int) bool { return cells[i].v > cells[j].v })
	// 距離マップ再利用
	distMe := computeSlideDistances(st, getPos(st, player), getPos(st, opp))
	distOp := computeSlideDistances(st, getPos(st, opp), getPos(st, player))
	bonus := 0.0
	maxK := highValueK
	if len(cells) < highValueK {
		maxK = len(cells)
	}
	for i := 0; i < maxK; i++ {
		dm := distMe[cells[i].y][cells[i].x]
		do := distOp[cells[i].y][cells[i].x]
		switch {
		case dm < do:
			bonus += highValueBonus * cells[i].v
		case do < dm:
			bonus -= highValueBonus * cells[i].v
		}
	}
	return bonus
}

// mobilityCount は与えられた位置と相手位置から、一手で到達可能な停止マスの数を返します。
func mobilityCount(st *game.GameState, me, other game.Position) int {
	N := len(st.Board)
	count := 0
	dirs := []struct{ dx, dy int }{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for _, dir := range dirs {
		x, y := me.X, me.Y
		for {
			x += dir.dx
			y += dir.dy
			if x < 0 || x >= N || y < 0 || y >= N {
				break
			}
			if st.Rocks[y][x] || (other.X == x && other.Y == y) {
				break
			}
			count++
		}
	}
	return count
}

// computeSlideDistances は start からの最短スライド手数を全セルについて計算します。
func computeSlideDistances(
	st *game.GameState, start, other game.Position,
) [][]int {
	N := len(st.Board)
	const INF = math.MaxInt32
	dist := make([][]int, N)
	for i := range dist {
		dist[i] = make([]int, N)
		for j := range dist[i] {
			dist[i][j] = INF
		}
	}
	type node struct{ x, y, d int }
	queue := []node{{start.X, start.Y, 0}}
	dist[start.Y][start.X] = 0
	dirs := []struct{ dx, dy int }{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for i := 0; i < len(queue); i++ {
		n := queue[i]
		for _, dir := range dirs {
			for step := 1; ; step++ {
				x := n.x + dir.dx*step
				y := n.y + dir.dy*step
				if x < 0 || x >= N || y < 0 || y >= N {
					break
				}
				if st.Rocks[y][x] || (other.X == x && other.Y == y) {
					break
				}
				if dist[y][x] > n.d+1 {
					dist[y][x] = n.d + 1
					queue = append(queue, node{x, y, n.d + 1})
				}
			}
		}
	}
	return dist
}

// getPos はプレイヤーインデックスから対応する位置を返します。
func getPos(st *game.GameState, player int) game.Position {
	if player == 0 {
		return st.Player0
	}
	return st.Player1
}
