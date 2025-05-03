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
	InitialState *GameState `json:"initialState"` // 初期状態
	Moves        []Move     `json:"moves"`        // 手の履歴
	FinalState   *GameState `json:"finalState"`   // 最終状態
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
	if first == 1 {
		state.Player0, state.Player1 = state.Player1, state.Player0
	}

	// 3) AIの名前を設定
	state.Player0Name = gr.agents[0].Name()
	state.Player1Name = gr.agents[1].Name()

	// 4) 結果オブジェクトの初期化
	result := BattleResult{
		InitialState: state.Clone(),
		Moves:        make([]Move, 0),
	}

	// 4) ゲームループ
	skips := 0
	for skips < 2 {
		player := state.Turn
		debug.Log("Turn: %d, Position: %v and %v", player, state.Player0, state.Player1)

		moves := state.LegalMoves(player)
		debug.Log("Legal moves count: %d", len(moves))

		if len(moves) == 0 {
			debug.Log("No legal moves, skipping player %d", player)
			skips++
			state.Turn = 1 - player
			continue
		}

		skips = 0
		bestEval := math.Inf(-1)
		var bestM Move

		for i, m := range moves {
			ev := gr.agents[player].Evaluate(m.State, player)
			debug.Log("Move %d: %v -> %v (eval: %f)", i, m.From(), m.To(), ev)

			if ev > bestEval {
				bestEval = ev
				bestM = m
			}
		}

		// 有効な手が見つかった場合のみ進める
		if bestEval != math.Inf(-1) {
			debug.Log("Selected move: %v -> %v", bestM.From(), bestM.To())
			state = bestM.State
			result.Moves = append(result.Moves, bestM)
		} else {
			// 有効な手がない場合はスキップ
			debug.Log("No valid moves found for player %d", player)
			state.Turn = 1 - player
			skips++
		}
	}

	// 最終状態を保存
	result.FinalState = state.Clone()
	return result
}
