package statiolake

import (
	"math"

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
	playerMoves := state.LegalMoves(player)
	if len(playerMoves) == 0 {
		return -100000.0
	}

	eval := 0.0

	playerArea := getControlledArea(state, player)
	opponentArea := getControlledArea(state, 1-player)
	debug.Log("playerArea.UnclaimedValue: %d, opponentArea.UnclaimedValue: %d", playerArea.UnclaimedValue, opponentArea.UnclaimedValue)

	eval += float64(playerArea.UnclaimedValue - opponentArea.UnclaimedValue)

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
	opponentPos := getPlayerPosition(state, 1-player)
	playerDist := calculateDistance(pos, opponentPos)

	// 適度な距離（盤面サイズの1/3程度）が理想
	optimalDist := float64(boardSize) * 0.3
	distanceValue := 0.0

	if playerDist < optimalDist {
		distanceValue = (playerDist / optimalDist) * 100.0 // 近すぎると低評価
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

func getBlockingCells(state *game.GameState, player int) [][]bool {
	pos := getPlayerPosition(state, player)

	blocking := make([][]bool, len(state.Board))
	for i := range blocking {
		blocking[i] = make([]bool, len(state.Board))
	}

	// 全方位ブロックできる
	for _, dir := range DIRS {
		for dist := 1; dist < len(state.Board); dist++ {
			x := pos.X + dir[0]*dist
			y := pos.Y + dir[1]*dist

			// 盤面外または岩石にぶつかる場合は終了
			if !IsMovable(state, player, game.Position{X: x, Y: y}) {
				break
			}

			// プレイヤー位置でも終了
			if x == state.Player0.X && y == state.Player0.Y {
				break
			}
			if x == state.Player1.X && y == state.Player1.Y {
				break
			}

			blocking[y][x] = true
		}
	}

	return blocking
}

func getControlledArea(state *game.GameState, player int) AreaAnalysis {
	opPos := getPlayerPosition(state, 1-player)
	myPos := getPlayerPosition(state, player)

	computeDist := func(player int, start game.Position, addition int, blocked [][]bool) [][]int {
		dist := make([][]int, len(state.Board))
		for i := range dist {
			dist[i] = make([]int, len(state.Board))
			for j := range dist[i] {
				dist[i][j] = INF
			}
		}

		ActionBfs(state, blocked, player, start, func(pos game.Position, d int) {
			dist[pos.Y][pos.X] = d + addition
		})

		return dist
	}

	opDist := computeDist(1-player, opPos, 0, nil)
	debug.Log("opDist: %v", opDist)

	opBlocking := getBlockingCells(state, 1-player)

	var bestArea AreaAnalysis

	// 最初の一歩を確定させて評価する
	// (そうしないとエリアを分けた瞬間を適切に評価できない)
	for i, dir := range DIRS {
		nx, ny := myPos.X+dir[0], myPos.Y+dir[1]
		start := game.Position{X: nx, Y: ny}

		myDist := computeDist(player, start, 1, opBlocking)
		// スタート位置をずらした分の補正
		// 自分が歩いた方向からは 1 引く (まとめて移動できるので)
		for d := 2; d < len(myDist); d++ {
			x := myPos.X + dir[0]*d
			y := myPos.Y + dir[1]*d
			if !IsMovable(state, player, game.Position{X: x, Y: y}) {
				break
			}

			myDist[y][x]--
		}

		debug.Log("myDist (%d): %v", i, myDist)

		totalValue := 0
		unclaimedValue := 0
		for y := 0; y < len(state.Board); y++ {
			for x := 0; x < len(state.Board[y]); x++ {
				if myDist[y][x] == INF || myDist[y][x] > opDist[y][x] {
					continue
				}

				// 価値の計算
				cellValue := state.Board[y][x]
				totalValue += cellValue

				// 未獲得のセルであれば、獲得可能価値に加算
				if state.Colors[y][x] != player {
					unclaimedValue += cellValue
				}
			}
		}

		if unclaimedValue > bestArea.UnclaimedValue {
			bestArea = AreaAnalysis{
				TotalValue:     totalValue,
				UnclaimedValue: unclaimedValue,
			}
		}
	}

	return bestArea
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

		if !IsMovable(state, player, current.Pos) || (blocked != nil && !blocked[current.Pos.Y][current.Pos.X]) {
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

// 領域分析結果を格納する構造体
type AreaAnalysis struct {
	Cells          []game.Position // 領域内のセル
	TotalValue     int             // 領域の総価値
	UnclaimedValue int             // 未獲得の価値
	Size           int             // 領域の大きさ
}
