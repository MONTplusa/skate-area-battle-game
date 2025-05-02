package game

import (
	"math"
	"math/rand"
	"time"

	"github.com/montplusa/skate-area-battle-game/pkg/game/debug"
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
		debug.Log("Turn:", player, "Position:", state.Player0, state.Player1)

		moves := state.LegalMoves(player)
		debug.Log("Legal moves count:", len(moves))

		if len(moves) == 0 {
			debug.Log("No legal moves, skipping player", player)
			skips++
			state.Turn = 1 - player
			continue
		}

		skips = 0
		bestEval := math.Inf(-1)
		var bestM Move

		for i, m := range moves {
			ev := gr.agents[player].Evaluate(m.State, player)
			debug.Log("Move", i, ":", m.FromX, m.FromY, "->", m.ToX, m.ToY, "eval:", ev)
			if ev > bestEval {
				bestEval = ev
				bestM = m
			}
		}

		// 有効な手が見つかった場合のみ進める
		if bestEval != math.Inf(-1) {
			debug.Log("Selected move:", bestM.FromX, bestM.FromY, "->", bestM.ToX, bestM.ToY)
			state = bestM.State
			result.Moves = append(result.Moves, bestM)
		} else {
			// 有効な手がない場合はスキップ
			debug.Log("No valid moves found for player", player)
			state.Turn = 1 - player
			skips++
		}
	}

	return result
}
