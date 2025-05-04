package montplusa

import (
	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

const (
	BOARD_SIZE = 20
)

type MontplusaAI struct{}

func New() *MontplusaAI { return &MontplusaAI{} }

func (ai *MontplusaAI) Name() string {
	return "montplusa"
}

func (ai *MontplusaAI) SelectBoard(states []*game.GameState) int {
	bestIdx := 0
	return bestIdx
}

func (ai *MontplusaAI) SelectTurn(states []*game.GameState) int {
	return 0
}

func (ai *MontplusaAI) Evaluate(st *game.GameState, player int) float64 {
	poss := [2][2]int{}
	poss[0][0] = st.Player0.Y
	poss[0][1] = st.Player0.X
	poss[1][0] = st.Player1.Y
	poss[1][1] = st.Player1.X
	minDiff := 1000000000.0
	for firstDir := 0; firstDir < 4; firstDir++ {
		maxDiff := -1000000000.0
		for secondDir := 0; secondDir < 4; secondDir++ {
			minDiff2 := 1000000000.0
			for dir3 := 0; dir3 < 4; dir3++ {
				territory := judgeTerritory(st, poss, firstDir, secondDir, dir3, st.Turn)
				scores := [2]float64{}
				for i := range territory {
					for j := range territory[i] {
						if territory[i][j] != -1 {
							scores[territory[i][j]] += float64(st.Board[i][j])
						}
					}
				}
				for i := range st.Colors {
					for j := range st.Colors[i] {
						if st.Colors[i][j] != -1 {
							scores[st.Colors[i][j]] += float64(st.Board[i][j]) * 0.1
						}
					}
				}
				diff := scores[player] - scores[1-player]
				if diff < minDiff2 {
					minDiff2 = diff
				}
			}
			if minDiff2 > maxDiff {
				maxDiff = minDiff2
			}
		}
		if maxDiff < minDiff {
			minDiff = maxDiff
		}
	}
	dirs := [4][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
	value := minDiff
	c := 0
	for d := 0; d < 4; d++ {
		y := poss[player][0] + dirs[d][0]
		x := poss[player][1] + dirs[d][1]
		if x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE {
			continue
		}
		if x == poss[1-player][1] && y == poss[1-player][0] {
			continue
		}
		if st.Rocks[y][x] {
			continue
		}
		c++
	}
	if c == 1 {
		value += 0.01
	}
	return value
}

func judgeTerritory(state *game.GameState, poss [2][2]int, firstDir int, secondDir int, dir3 int, turn int) [][]int {
	dirs := [4][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
	results := make([][]int, len(state.Board))
	for i := range results {
		results[i] = make([]int, len(state.Board[i]))
		for j := range results[i] {
			results[i][j] = -1
		}
	}
	queue := NewBFSQueue(len(state.Board) * len(state.Board))
	queue.Init()
	p1 := from2dTo1d(poss[turn])
	p2 := from2dTo1d(poss[1-turn])
	queue.Push(p1)
	results[poss[turn][0]][poss[turn][1]] = turn
	queue.Push(p2)
	results[poss[1-turn][0]][poss[1-turn][1]] = 1 - turn
	bfscount := 0
	end3 := 2
	for queue.Len() > 0 {
		p1d := queue.Pop()
		p2d := from1dTo2d(p1d)
		pl := results[p2d[0]][p2d[1]]
		for d := 0; d < 4; d++ {
			if bfscount == 0 {
				if d != firstDir {
					continue
				}
			} else if bfscount == 1 {
				if d != secondDir {
					continue
				}
			} else if bfscount < end3 {
				if d != dir3 {
					continue
				}
			}
			y := p2d[0]
			x := p2d[1]
			for {
				y += dirs[d][0]
				x += dirs[d][1]
				if x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE {
					break
				}
				if state.Rocks[y][x] {
					break
				}
				if results[y][x] == 1-pl {
					break
				}
				np1d := from2dTo1d([2]int{y, x})
				if np1d == p1 || np1d == p2 {
					break
				}
				if results[y][x] == -1 {
					if bfscount == 0 {
						end3++
					}
					queue.Push(np1d)
					results[y][x] = pl
				}
			}
		}
		bfscount++
	}
	for i := range results {
		for j := range results[i] {
			if results[i][j] == -1 {
				results[i][j] = state.Colors[i][j]
			}
		}
	}
	return results
}

func from2dTo1d(p2d [2]int) int {
	return p2d[0]*BOARD_SIZE + p2d[1]
}
func from1dTo2d(p1d int) [2]int {
	return [2]int{p1d / BOARD_SIZE, p1d % BOARD_SIZE}
}

// 最大Loopがわかっている場合のqueue BFSで使うことが多い
type BFSQueue struct {
	queue     []int // 対象をなんらかのindexで持っておく想定 距離の場合はposなど
	popIndex  int
	pushIndex int
}

func NewBFSQueue(maxLoop int) *BFSQueue {
	q := BFSQueue{}
	q.queue = make([]int, maxLoop)
	return &q
}

func (q *BFSQueue) Init() {
	q.popIndex = 0
	q.pushIndex = 0
}

func (q *BFSQueue) Pop() int {
	v := q.queue[q.popIndex]
	q.popIndex++
	return v
}

func (q *BFSQueue) Push(v int) {
	q.queue[q.pushIndex] = v
	q.pushIndex++
}

func (q *BFSQueue) Len() int {
	return q.pushIndex - q.popIndex
}
