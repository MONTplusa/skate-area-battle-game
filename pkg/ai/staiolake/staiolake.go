package staiolake

import (
	"math"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

// StaiolakeAI はスケートエリア陣取りゲームのAI実装です
type StaiolakeAI struct{}

// New はStaiolakeAIのインスタンスを返します
func New() *StaiolakeAI {
	return &StaiolakeAI{}
}

// SelectBoard は提示された複数の初期盤面から1つを選択します
func (ai *StaiolakeAI) SelectBoard(states []*game.GameState) int {
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
func (ai *StaiolakeAI) SelectTurn(states []*game.GameState) int {
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
func (ai *StaiolakeAI) Evaluate(state *game.GameState, player int) float64 {
	// ゲーム終了時は実際のスコア差で評価
	if isGameOver(state) {
		return evaluateFinalScore(state, player)
	}

	// プレイヤーが動けない場合は非常に不利
	playerMoves := state.LegalMoves(player)
	if len(playerMoves) == 0 {
		return -10000.0
	}

	// 1. アクセス可能領域を分析
	area := getAccessibleArea(state, player)
	boardSize := len(state.Board)
	totalCells := boardSize * boardSize

	// 2. 最良の次の手を評価
	bestMoveValue := -math.MaxFloat64
	bestMoveAreaSize := 0

	for _, move := range playerMoves {
		// 移動後の領域サイズと潜在価値を評価
		afterArea := getAccessibleArea(move.State, player)

		// 「移動後の領域サイズ」と「獲得点数」のバランスで評価
		moveScore := float64(afterArea.Size)*5.0 + float64(calculateScoreGain(state, move, player))

		if moveScore > bestMoveValue {
			bestMoveValue = moveScore
			bestMoveAreaSize = afterArea.Size
		}
	}

	// 3. 現在と将来の領域サイズを比較
	// 現在より小さくなる場合は問題があるため、サイズ比を重視
	sizeFactor := 1.0
	if bestMoveAreaSize < area.Size {
		// 領域が縮小する場合は、縮小度合いに応じて評価を下げる
		sizeFactor = float64(bestMoveAreaSize) / float64(area.Size)

		// 極端に小さな領域に入り込む場合は著しく評価を下げる
		if bestMoveAreaSize < totalCells/10 {
			sizeFactor *= 0.5
		}
	}

	// 4. 基本スコア = 領域内の総潜在価値 × サイズ係数
	potentialValue := float64(area.TotalValue)

	// 極めて小さい領域は重大なペナルティ
	if area.Size < totalCells/10 {
		potentialValue *= 0.2 // 80%減少
	} else if area.Size < totalCells/5 {
		potentialValue *= 0.7 // 30%減少
	}

	return potentialValue * sizeFactor
}

// 初期位置の評価
func (ai *StaiolakeAI) evaluateInitialPosition(state *game.GameState, player int) float64 {
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
	boardSize := len(state.Board)

	// プレイヤーの位置
	pos := getPlayerPosition(state, player)

	// 訪問マップ
	visited := make([][]bool, boardSize)
	for i := range visited {
		visited[i] = make([]bool, boardSize)
	}

	// BFSで到達可能領域を探索
	queue := []game.Position{pos}
	visited[pos.Y][pos.X] = true

	var accessibleCells []game.Position
	totalValue := 0
	unclaimedValue := 0

	// 方向ベクトル
	dirs := [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		// アクセス可能なセルとして記録
		accessibleCells = append(accessibleCells, current)

		// 価値の計算
		cellValue := state.Board[current.Y][current.X]
		totalValue += cellValue

		// 未獲得のセルであれば、獲得可能価値に加算
		if state.Colors[current.Y][current.X] == -1 {
			unclaimedValue += cellValue
		}

		// 隣接セルの探索
		for _, dir := range dirs {
			nx, ny := current.X+dir[0], current.Y+dir[1]

			// 盤面内、未訪問、岩石なし、相手の位置でない
			if nx >= 0 && nx < boardSize && ny >= 0 && ny < boardSize &&
				!visited[ny][nx] && !state.Rocks[ny][nx] {

				// 相手の位置チェック
				isOpponentPos := false
				if player == 0 && nx == state.Player1.X && ny == state.Player1.Y {
					isOpponentPos = true
				} else if player == 1 && nx == state.Player0.X && ny == state.Player0.Y {
					isOpponentPos = true
				}

				if !isOpponentPos {
					visited[ny][nx] = true
					queue = append(queue, game.Position{X: nx, Y: ny})
				}
			}
		}
	}

	return AreaAnalysis{
		Cells:          accessibleCells,
		TotalValue:     totalValue,
		UnclaimedValue: unclaimedValue,
		Size:           len(accessibleCells),
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

// 領域分析結果を格納する構造体
type AreaAnalysis struct {
	Cells          []game.Position // 領域内のセル
	TotalValue     int             // 領域の総価値
	UnclaimedValue int             // 未獲得の価値
	Size           int             // 領域の大きさ
}
