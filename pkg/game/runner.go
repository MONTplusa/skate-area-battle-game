package game

import (
	"math"
	"math/rand"
	"time"
)

const (
	NUM_SAMPLE = 10
	N          = 20
)

// BattleResult は対戦結果の記録
type BattleResult struct {
	InitialState *GameState // 初期状態
	Moves        []Move     // 手の履歴
}

// GameRunner は対戦を管理
type GameRunner struct {
	agents [2]AI
}

// NewGameRunner は AI エージェントをセットして返す
func NewGameRunner(a0, a1 AI) *GameRunner {
	return &GameRunner{agents: [2]AI{a0, a1}}
}

// Run は対戦を実行して BattleResult を返す
func (gr *GameRunner) Run() BattleResult {
	rand.Seed(time.Now().UnixNano())

	// 1) 初期盤面サンプル生成 & 盤面選択
	states := GenerateInitialStates(NUM_SAMPLE, N)
	chooser := rand.Intn(2)
	other := 1 - chooser
	idx := gr.agents[other].SelectBoard(states)
	state := states[idx].Clone()

	// 2) 先後選択
	first := gr.agents[chooser].SelectTurn(states)
	state.Turn = first

	// 3) 結果オブジェクトの初期化
	result := BattleResult{
		InitialState: state.Clone(),
		Moves:        make([]Move, 0),
	}

	// 4) ゲームループ
	skips := 0
	for skips < 2 {
		player := state.Turn
		moves := state.LegalMoves(player)
		if len(moves) == 0 {
			skips++
			state.Turn = 1 - player
			continue
		}
		skips = 0
		bestEval := math.Inf(-1)
		var bestM Move
		var bestSt *GameState
		for _, m := range moves {
			c := state.Clone()
			c.ApplyMove(m)
			ev := gr.agents[player].Evaluate(c, player)
			if ev > bestEval {
				bestEval = ev
				bestM = m
				bestSt = c
			}
		}

		// 有効な手が見つかった場合のみ進める
		if bestSt != nil {
			state = bestSt
			result.Moves = append(result.Moves, bestM)
		} else {
			// 有効な手がない場合はスキップ
			state.Turn = 1 - player
			skips++
		}
	}

	return result
}
