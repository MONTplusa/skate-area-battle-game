package statiolake

import (
	"math"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
	"github.com/montplusa/skate-area-battle-game/pkg/game/debug"
)

const INF = 1e8

// StatiolakeAI はスケートエリア陣取りゲームのAI実装です
type StatiolakeAI struct{}

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
	state := states[0]

	// 先手と後手どちらが有利かを評価
	firstEval := ai.evaluateInitialPosition(state, 0)
	secondEval := ai.evaluateInitialPosition(state, 1)

	if firstEval >= secondEval {
		return 0 // 先手
	}
	return 1 // 後手
}

// Evaluate は現在の盤面の評価値を返します
func (ai *StatiolakeAI) Evaluate(state *game.GameState, player int) float64 {
	// プレイヤーが動けない場合は非常に不利
	playerMoves := state.LegalMoves(player)
	if len(playerMoves) == 0 {
		return -100000.0
	}

	eval := 0.0

	// 1. アクセス可能領域を分析
	playerArea := getControlledArea(state, player)
	opponentArea := getControlledArea(state, 1-player)
	debug.Log("playerArea.UnclaimedValue: %d, opponentArea.UnclaimedValue: %d", playerArea.UnclaimedValue, opponentArea.UnclaimedValue)

	eval += float64(playerArea.UnclaimedValue - opponentArea.UnclaimedValue)

	// タイブレーク要素・相手とのマンハッタン距離^2
	// eval += (20.0*20.0 - float64((state.Player0.X-state.Player1.X)*(state.Player0.X-state.Player1.X))) * 0.001
	// eval += (20.0*20.0 - float64((state.Player0.Y-state.Player1.Y)*(state.Player0.Y-state.Player1.Y))) * 0.001

	// boardSize := len(state.Board)
	// totalCells := boardSize * boardSize
	//
	// // 2. 最良の次の手を評価
	// bestMoveValue := -math.MaxFloat64
	// bestMoveAreaSize := 0
	//
	// for _, move := range playerMoves {
	// 	// 移動後の領域サイズと潜在価値を評価
	// 	afterArea := getAccessibleArea(move.State, player)
	//
	// 	// 移動後の相手の行動制限を評価
	// 	opponentRestriction := 0.0
	// 	afterOpponentMoves := move.State.LegalMoves(1 - player)
	//
	// 	// 相手の行動を制限できる度合い
	// 	if len(opponentMoves) > 0 {
	// 		opponentRestriction = (1.0 - float64(len(afterOpponentMoves))/float64(len(opponentMoves))) * 100.0
	// 	}
	//
	// 	// 相手を行動不能にできれば大きなボーナス
	// 	if len(afterOpponentMoves) == 0 {
	// 		opponentRestriction += 200.0
	// 	}
	//
	// 	// 「移動後の領域サイズ」、「獲得点数」、「相手制限」のバランスで評価
	// 	moveScore := float64(afterArea.Size)*5.0 +
	// 		float64(calculateScoreGain(state, move, player)) +
	// 		opponentRestriction*2.0
	//
	// 	if moveScore > bestMoveValue {
	// 		bestMoveValue = moveScore
	// 		bestMoveAreaSize = afterArea.Size
	// 	}
	// }
	//
	// // 3. 現在と将来の領域サイズを比較
	// // 現在より小さくなる場合は問題があるため、サイズ比を重視
	// sizeFactor := 1.0
	// if bestMoveAreaSize < playerArea.Size {
	// 	// 領域が縮小する場合は、縮小度合いに応じて評価を下げる
	// 	sizeFactor = float64(bestMoveAreaSize) / float64(playerArea.Size)
	//
	// 	// 極端に小さな領域に入り込む場合は著しく評価を下げる
	// 	if bestMoveAreaSize < totalCells/10 {
	// 		sizeFactor *= 0.5
	// 	}
	// }
	//
	// // 4. 基本スコア = 領域内の総潜在価値 × サイズ係数
	// potentialValue := float64(playerArea.TotalValue)
	//
	// // 極めて小さい領域は重大なペナルティ
	// if playerArea.Size < totalCells/10 {
	// 	potentialValue *= 0.2 // 80%減少
	// } else if playerArea.Size < totalCells/5 {
	// 	potentialValue *= 0.7 // 30%減少
	// }
	//
	// // 5. タイブレーク要素：同程度に安全な状態間の優劣判断
	//
	// // 5.1 相手との領域サイズ比較 - 自分の方が広いほど有利
	// areaRatio := 0.0
	// if opponentArea.Size > 0 {
	// 	areaRatio = float64(playerArea.Size) / float64(opponentArea.Size)
	// 	// 自分の領域が相手より大きければボーナス
	// 	if areaRatio > 1.0 {
	// 		areaRatio = math.Min(2.0, areaRatio) // 最大2倍まで評価
	// 	}
	// } else {
	// 	areaRatio = 2.0 // 相手の領域がゼロなら最大評価
	// }
	//
	// // 5.2 移動の自由度比較 - 自分の選択肢が多いほど有利
	// mobilityRatio := 0.0
	// if len(opponentMoves) > 0 {
	// 	mobilityRatio = float64(len(playerMoves)) / float64(len(opponentMoves))
	// 	// 自分の移動選択肢が相手より多ければボーナス
	// 	if mobilityRatio > 1.0 {
	// 		mobilityRatio = math.Min(2.0, mobilityRatio) // 最大2倍まで評価
	// 	}
	// } else {
	// 	mobilityRatio = 2.0 // 相手の移動選択肢がゼロなら最大評価
	// }
	//
	// // 5.3 領域内の平均価値比較 - 自分の領域の方が価値が高いほど有利
	// valueRatio := 0.0
	// if playerArea.Size > 0 && opponentArea.Size > 0 {
	// 	playerAvgValue := float64(playerArea.TotalValue) / float64(playerArea.Size)
	// 	opponentAvgValue := float64(opponentArea.TotalValue) / float64(opponentArea.Size)
	//
	// 	if opponentAvgValue > 0 {
	// 		valueRatio = playerAvgValue / opponentAvgValue
	// 		// 自分の平均価値が相手より高ければボーナス
	// 		if valueRatio > 1.0 {
	// 			valueRatio = math.Min(2.0, valueRatio) // 最大2倍まで評価
	// 		}
	// 	} else {
	// 		valueRatio = 2.0 // 相手の平均価値がゼロなら最大評価
	// 	}
	// }
	//
	// // 移動方向の多様性評価 - 異なる方向に移動できるほど有利
	// directionDiversity := calculateDirectionDiversity(playerMoves)
	//
	// // タイブレーク効果計算 (最大で基本スコアの40%程度が適切)
	// tiebreakBonus := (areaRatio + mobilityRatio + valueRatio + directionDiversity*0.5 - 4.0) * 50.0
	//
	// // 6. 最終評価値を返す
	// return potentialValue*sizeFactor + tiebreakBonus

	return eval
}

// 移動方向の多様性を評価（0.0～1.0）
func calculateDirectionDiversity(moves []game.Move) float64 {
	if len(moves) == 0 {
		return 0.0
	}

	// 各方向への移動フラグ
	hasUp := false
	hasDown := false
	hasLeft := false
	hasRight := false

	for _, move := range moves {
		dx := move.ToX - move.FromX
		dy := move.ToY - move.FromY

		if dx > 0 {
			hasRight = true
		} else if dx < 0 {
			hasLeft = true
		}

		if dy > 0 {
			hasDown = true
		} else if dy < 0 {
			hasUp = true
		}
	}

	// 方向の数をカウント
	dirCount := 0
	if hasUp {
		dirCount++
	}
	if hasDown {
		dirCount++
	}
	if hasLeft {
		dirCount++
	}
	if hasRight {
		dirCount++
	}

	// 方向の多様性（0.0～1.0）
	return float64(dirCount) / 4.0
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

// 到達可能領域の取得
func getAccessibleArea(state *game.GameState, player int) AreaAnalysis {
	// プレイヤーの位置
	pos := getPlayerPosition(state, player)

	// 方向ベクトル
	dirs := [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

	var bestArea AreaAnalysis

	// 最初の一歩を確定させて評価する
	// (そうしないとエリアを分けた瞬間を適切に評価できない)
	for _, dir := range dirs {
		nx, ny := pos.X+dir[0], pos.Y+dir[1]
		start := game.Position{X: nx, Y: ny}

		if !IsMovable(state, player, start) {
			continue
		}

		var accessibleCells []game.Position
		totalValue := 0
		unclaimedValue := 0
		ActionBfs(state, player, start, func(pos game.Position, dist int) bool {
			// アクセス可能なセルとして記録
			accessibleCells = append(accessibleCells, pos)

			// 価値の計算
			cellValue := state.Board[pos.Y][pos.X]
			totalValue += cellValue

			// 未獲得のセルであれば、獲得可能価値に加算
			if state.Colors[pos.Y][pos.X] != player {
				unclaimedValue += cellValue
			}

			return true
		})

		area := AreaAnalysis{
			Cells:          accessibleCells,
			TotalValue:     totalValue,
			UnclaimedValue: unclaimedValue,
			Size:           len(accessibleCells),
		}

		if area.UnclaimedValue > bestArea.UnclaimedValue {
			bestArea = area
		}
	}

	return bestArea
}

func getControlledArea(state *game.GameState, player int) AreaAnalysis {
	// 方向ベクトル
	dirs := [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

	opDist := make([][]int, len(state.Board))
	for i := range opDist {
		opDist[i] = make([]int, len(state.Board))
		for j := range opDist[i] {
			opDist[i][j] = INF
		}
	}

	opPos := getPlayerPosition(state, 1-player)

	ActionBfs(state, 1-player, opPos, func(pos game.Position, dist int) bool {
		opDist[pos.Y][pos.X] = dist
		return true
	})

	// プレイヤーの位置
	myPos := getPlayerPosition(state, player)

	var bestArea AreaAnalysis

	// 最初の一歩を確定させて評価する
	// (そうしないとエリアを分けた瞬間を適切に評価できない)
	for _, dir := range dirs {
		nx, ny := myPos.X+dir[0], myPos.Y+dir[1]
		start := game.Position{X: nx, Y: ny}

		myDist := make([][]int, len(state.Board))
		for i := range myDist {
			myDist[i] = make([]int, len(state.Board))
			for j := range myDist[i] {
				myDist[i][j] = INF
			}
		}

		ActionBfs(state, player, start, func(pos game.Position, dist int) bool {
			myDist[pos.Y][pos.X] = dist
			return true
		})

		// debug.Log("myDist: %v", myDist)
		// debug.Log("opDist: %v", opDist)

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

		area := AreaAnalysis{
			TotalValue:     totalValue,
			UnclaimedValue: unclaimedValue,
		}

		if area.UnclaimedValue > bestArea.UnclaimedValue {
			bestArea = area
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

func ActionBfs(state *game.GameState, player int, start game.Position, visit func(game.Position, int) bool) {
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

	dirs := [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if !IsMovable(state, player, current.Pos) || visited[current.Pos.Y][current.Pos.X] {
			continue
		}

		visited[current.Pos.Y][current.Pos.X] = true
		if !visit(current.Pos, current.Dist) {
			return
		}

		pos := current.Pos
		for _, dir := range dirs {
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

				// プレイヤー位置でも終了
				if x == state.Player0.X && y == state.Player0.Y {
					break
				}
				if x == state.Player1.X && y == state.Player1.Y {
					break
				}

				if !visited[y][x] {
					queue = append(queue, BfsNode{game.Position{X: x, Y: y}, current.Dist + 1})
				}
			}
		}
	}
}

// 移動による得点獲得量の計算
func calculateScoreGain(state *game.GameState, move game.Move, player int) int {
	beforeScore := 0
	afterScore := 0

	// 移動前のスコア計算
	for y := 0; y < len(state.Board); y++ {
		for x := 0; x < len(state.Board[y]); x++ {
			if state.Colors[y][x] == player {
				beforeScore += state.Board[y][x]
			}
		}
	}

	// 移動後のスコア計算
	for y := 0; y < len(state.Board); y++ {
		for x := 0; x < len(state.Board[y]); x++ {
			if move.State.Colors[y][x] == player {
				afterScore += state.Board[y][x]
			}
		}
	}

	return afterScore - beforeScore
}

// プレイヤーの位置を取得
func getPlayerPosition(state *game.GameState, player int) game.Position {
	if player == 0 {
		return state.Player0
	}
	return state.Player1
}

// ゲーム終了時の評価
func evaluateFinalScore(state *game.GameState, player int) float64 {
	opponent := 1 - player

	// 最終スコア計算
	playerScore := 0
	opponentScore := 0

	for y := 0; y < len(state.Board); y++ {
		for x := 0; x < len(state.Board[y]); x++ {
			if state.Colors[y][x] == player {
				playerScore += state.Board[y][x]
			} else if state.Colors[y][x] == opponent {
				opponentScore += state.Board[y][x]
			}
		}
	}

	// スコア差
	scoreDiff := playerScore - opponentScore

	// 勝ち：大きな正の値、負け：大きな負の値
	if scoreDiff > 0 {
		return 10000.0 + float64(scoreDiff)
	} else if scoreDiff < 0 {
		return -10000.0 + float64(scoreDiff)
	}

	return 0.0 // 引き分け
}

// ゲーム終了判定
func isGameOver(state *game.GameState) bool {
	return len(state.LegalMoves(0)) == 0 && len(state.LegalMoves(1)) == 0
}

// 2点間の距離計算
func calculateDistance(pos1, pos2 game.Position) float64 {
	dx := pos1.X - pos2.X
	dy := pos1.Y - pos2.Y
	return math.Sqrt(float64(dx*dx + dy*dy))
}

// マンハッタン距離
func calculateManhattanDistance(pos1, pos2 game.Position) int {
	dx := pos1.X - pos2.X
	dy := pos1.Y - pos2.Y
	return int(math.Abs(float64(dx)) + math.Abs(float64(dy)))
}

// 領域分析結果を格納する構造体
type AreaAnalysis struct {
	Cells          []game.Position // 領域内のセル
	TotalValue     int             // 領域の総価値
	UnclaimedValue int             // 未獲得の価値
	Size           int             // 領域の大きさ
}
