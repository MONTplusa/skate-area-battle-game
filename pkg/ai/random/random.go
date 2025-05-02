package random

import (
	"math/rand"
	"skate-area-battle-game/pkg/game"
)

// RandomAI はランダムに行動を選ぶ実装
type RandomAI struct{}

// New は RandomAI を生成する
func New() game.AI { return &RandomAI{} }

func (r *RandomAI) SelectBoard(states []*game.GameState) int {
	return rand.Intn(len(states))
}
func (r *RandomAI) SelectTurn(states []*game.GameState) int {
	return rand.Intn(2)
}
func (r *RandomAI) Evaluate(state *game.GameState, player int) float64 {
	// 評価: 自分の塗りマス数 - 相手の塗りマス数
	var me, opp int
	for y := 0; y < state.N; y++ {
		for x := 0; x < state.N; x++ {
			if state.Owner[y][x] == player {
				me += state.Value[y][x]
			} else if state.Owner[y][x] == 1-player {
				opp += state.Value[y][x]
			}
		}
	}
	return float64(me - opp)
}
