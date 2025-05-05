package statiolake

import (
	"math"
	"strconv"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
	"github.com/montplusa/skate-area-battle-game/pkg/game/debug"
)

const INF = 1e8

var DIRS = [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

// StatiolakeAI はスケートエリア陣取りゲームのAI実装です
type StatiolakeAI struct{}

func (ai *StatiolakeAI) Name() string {
	return "statiolake"
}

// New はStatiolakeAIのインスタンスを返します
func New() *StatiolakeAI {
	return &StatiolakeAI{}
}

// SelectBoard は提示された複数の初期盤面から1つを選択します
func (ai *StatiolakeAI) SelectBoard(states []*game.GameState) int {
	bestIdx := 0
	bestValue := -math.MaxFloat64

	for i, state := range states {
		// 先手と後手の両方で評価し、より高い方を採用
		firstEval := ai.evaluateInitialPosition(state, 0)
		secondEval := ai.evaluateInitialPosition(state, 1)
		value := math.Max(firstEval, secondEval)

		if value > bestValue {
			bestValue = value
			bestIdx = i
		}
	}

	return bestIdx
}

// SelectTurn は先手(0)か後手(1)かを選択します
func (ai *StatiolakeAI) SelectTurn(states []*game.GameState) int {
	return 1
	// state := states[0]
	//
	// // 先手と後手どちらが有利かを評価
	// firstEval := ai.evaluateInitialPosition(state, 0)
	// secondEval := ai.evaluateInitialPosition(state, 1)
	//
	// if firstEval >= secondEval {
	// 	return 0 // 先手
	// }
	// return 1 // 後手
}

// Evaluate は現在の盤面の評価値を返します
func (ai *StatiolakeAI) Evaluate(state *game.GameState, player int) float64 {
	// プレイヤーが動けない場合は非常に不利
	myMoves := state.LegalMoves(player)
	if len(myMoves) == 0 {
		return -100000.0
	}

	eval := 0.0

	controlledArea := getControlledArea(state, player)
	myArea := controlledArea.me
	opArea := controlledArea.op
	debug.Log("myArea.TotalValue: %d, opArea.TotalValue: %d", myArea.TotalValue, opArea.TotalValue)

	eval += float64(myArea.TotalValue - opArea.TotalValue)

	// タイブレーク要素・相手とのマンハッタン距離^2
	// eval += (20.0*20.0 - float64((state.Player0.X-state.Player1.X)*(state.Player0.X-state.Player1.X))) * 0.001
	// eval += (20.0*20.0 - float64((state.Player0.Y-state.Player1.Y)*(state.Player0.Y-state.Player1.Y))) * 0.001

	return eval
}

// 初期位置の評価
func (ai *StatiolakeAI) evaluateInitialPosition(state *game.GameState, player int) float64 {
	boardSize := len(state.Board)

	// プレイヤーの位置
	pos := getPlayerPosition(state, player)

	// 1. 高価値領域へのアクセス評価
	accessValue := evaluateSurroundingValue(state, pos, 5)

	// 2. 中央からの距離評価（中央に近いほど良い）
	centerX, centerY := boardSize/2, boardSize/2
	distToCenter := calculateDistance(pos, game.Position{X: centerX, Y: centerY})
	centerValue := (float64(boardSize) - distToCenter) * 5.0

	// 3. 相手との距離評価（適度な距離が理想）
	opPos := getPlayerPosition(state, 1-player)
	myDist := calculateDistance(pos, opPos)

	// 適度な距離（盤面サイズの1/3程度）が理想
	optimalDist := float64(boardSize) * 0.3
	distanceValue := 0.0

	if myDist < optimalDist {
		distanceValue = (myDist / optimalDist) * 100.0 // 近すぎると低評価
	} else {
		distanceValue = 100.0 // 適度な距離以上は高評価
	}

	return 0.6*accessValue + 0.2*centerValue + 0.2*distanceValue
}

// 周辺の価値評価
func evaluateSurroundingValue(state *game.GameState, pos game.Position, radius int) float64 {
	boardSize := len(state.Board)
	totalValue := 0.0
	count := 0

	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			// 現在位置は除外
			if dx == 0 && dy == 0 {
				continue
			}

			x, y := pos.X+dx, pos.Y+dy

			// 盤面内のみ評価
			if x >= 0 && x < boardSize && y >= 0 && y < boardSize {
				// 距離による重み付け（近いほど重要）
				dist := math.Sqrt(float64(dx*dx + dy*dy))
				weight := math.Max(0.0, 1.0-dist/float64(radius+1))

				// マスの価値
				value := float64(state.Board[y][x])
				totalValue += value * weight
				count++
			}
		}
	}

	if count > 0 {
		return totalValue
	}
	return 0.0
}

type ControlledArea struct {
	me AreaAnalysis
	op AreaAnalysis
}

// 領域分析結果を格納する構造体
type AreaAnalysis struct {
	Cells          []game.Position // 領域内のセル
	TotalValue     int             // 領域の総価値
	UnclaimedValue int             // 未獲得の価値
	Size           int             // 領域の大きさ
}

func computeDist(state *game.GameState, player int, start game.Position, blocked [][]bool) [][]int {
	dist := make([][]int, len(state.Board))
	for i := range dist {
		dist[i] = make([]int, len(state.Board))
		for j := range dist[i] {
			dist[i][j] = INF
		}
	}

	ActionBfs(state, blocked, player, start, func(pos game.Position, d int) {
		dist[pos.Y][pos.X] = d
	})

	return dist
}

func computeBlockers(freeDistA, freeDistB [][]int, isBlocked func(a, b int) bool) [][]bool {
	blockers := make([][]bool, len(freeDistA))
	for i := range blockers {
		blockers[i] = make([]bool, len(freeDistA[i]))
	}

	for y := 0; y < len(freeDistA); y++ {
		for x := 0; x < len(freeDistA[y]); x++ {
			if isBlocked(freeDistA[y][x], freeDistB[y][x]) {
				blockers[y][x] = true
			}
		}
	}

	return blockers
}

func computeArea(state *game.GameState, player int, dist [][]int, accessible [][]int) AreaAnalysis {
	area := AreaAnalysis{
		Cells:          []game.Position{},
		TotalValue:     0,
		UnclaimedValue: 0,
	}

	for y := 0; y < len(state.Board); y++ {
		for x := 0; x < len(state.Board[y]); x++ {
			if dist[y][x] == INF || (accessible != nil && accessible[y][x] == INF) {
				continue
			}

			cellValue := state.Board[y][x]
			area.TotalValue += cellValue

			if state.Colors[y][x] != player {
				area.UnclaimedValue += cellValue
			}

			area.Cells = append(area.Cells, game.Position{X: x, Y: y})
		}
	}

	return area
}

func dumpDistField(field [][]int) {
	lines := ""
	for y := 0; y < len(field); y++ {
		line := ""
		for x := 0; x < len(field[y]); x++ {
			if field[y][x] == INF {
				line += "X"
			} else {
				line += strconv.Itoa(field[y][x])
			}
		}
		lines += line + "\n"
	}
	debug.Log(lines)
}

func dumpBoolField(field [][]bool) {
	lines := ""
	for y := 0; y < len(field); y++ {
		line := ""
		for x := 0; x < len(field[y]); x++ {
			if field[y][x] {
				line += "*"
			} else {
				line += "."
			}
		}
		lines += line + "\n"
	}
	debug.Log(lines)
}

func getControlledArea(state *game.GameState, player int) ControlledArea {
	myPos := getPlayerPosition(state, player)
	opPos := getPlayerPosition(state, 1-player)

	myFreeDist := computeDist(state, player, myPos, nil)
	opFreeDist := computeDist(state, 1-player, opPos, nil)
	// debug.Log("myFreeDist")
	// dumpDistField(myFreeDist)
	// debug.Log("opFreeDist")
	// dumpDistField(opFreeDist)

	myBlocking := computeBlockers(myFreeDist, opFreeDist, func(a, b int) bool { return a < b })
	opBlocking := computeBlockers(opFreeDist, myFreeDist, func(a, b int) bool { return a <= b })
	// debug.Log("myBlocking")
	// dumpBoolField(myBlocking)
	// debug.Log("opBlocking")
	// dumpBoolField(opBlocking)

	myDist := computeDist(state, player, myPos, opBlocking)
	opDist := computeDist(state, 1-player, opPos, myBlocking)
	// debug.Log("myDist")
	// dumpDistField(myDist)
	// debug.Log("opDist")
	// dumpDistField(opDist)

	// 最初の一歩を確定させて評価する
	// (そうしないとエリアを分けた瞬間を適切に評価できない)
	var myBestArea AreaAnalysis
	for _, dir := range DIRS {
		nx, ny := myPos.X+dir[0], myPos.Y+dir[1]
		start := game.Position{X: nx, Y: ny}

		accessible := computeDist(state, player, start, opBlocking)
		// debug.Log("accessible (%d)", i)
		// dumpDistField(accessible)

		area := computeArea(state, player, myDist, accessible)
		if area.TotalValue >= myBestArea.TotalValue {
			myBestArea = area
		}
	}

	opArea := computeArea(state, 1-player, opDist, nil)

	return ControlledArea{
		me: myBestArea,
		op: opArea,
	}
}

func IsInside(state *game.GameState, pos game.Position) bool {
	boardSize := len(state.Board)
	return pos.X >= 0 && pos.X < boardSize && pos.Y >= 0 && pos.Y < boardSize
}

func IsMovable(state *game.GameState, player int, pos game.Position) bool {
	if !IsInside(state, pos) {
		return false
	}

	if state.Rocks[pos.Y][pos.X] {
		return false
	}

	if player == 1 && pos.X == state.Player0.X && pos.Y == state.Player0.Y {
		return false
	}
	if player == 0 && pos.X == state.Player1.X && pos.Y == state.Player1.Y {
		return false
	}

	return true
}

func ActionBfs(state *game.GameState, blocked [][]bool, player int, start game.Position, visit func(game.Position, int)) {
	boardSize := len(state.Board)

	type BfsNode struct {
		Pos  game.Position
		Dist int
	}

	queue := []BfsNode{{start, 0}}
	visited := make([][]bool, boardSize)
	for i := range visited {
		visited[i] = make([]bool, boardSize)
	}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if !IsMovable(state, player, current.Pos) || (blocked != nil && blocked[current.Pos.Y][current.Pos.X]) {
			continue
		}

		visit(current.Pos, current.Dist)

		pos := current.Pos
		for _, dir := range DIRS {
			for dist := 1; dist < boardSize; dist++ {
				x := pos.X + dir[0]*dist
				y := pos.Y + dir[1]*dist

				// 盤面外または岩石にぶつかる場合は終了
				if !IsMovable(state, player, game.Position{X: x, Y: y}) {
					break
				}
				if state.Rocks[y][x] {
					break
				}
				if blocked != nil && blocked[y][x] {
					break
				}

				// プレイヤー位置でも終了
				if x == state.Player0.X && y == state.Player0.Y {
					break
				}
				if x == state.Player1.X && y == state.Player1.Y {
					break
				}

				if !visited[y][x] {
					queue = append(queue, BfsNode{game.Position{X: x, Y: y}, current.Dist + 1})
					visited[y][x] = true
				}
			}
		}
	}
}

// プレイヤーの位置を取得
func getPlayerPosition(state *game.GameState, player int) game.Position {
	if player == 0 {
		return state.Player0
	}
	return state.Player1
}

// 2点間の距離計算
func calculateDistance(pos1, pos2 game.Position) float64 {
	dx := pos1.X - pos2.X
	dy := pos1.Y - pos2.Y
	return math.Sqrt(float64(dx*dx + dy*dy))
}
