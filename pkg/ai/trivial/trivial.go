package trivial

import (
	"math/rand"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

type TrivialAI struct{}

func (ai *TrivialAI) Name() string {
	return "trivial"
}

func New() *TrivialAI {
	return &TrivialAI{}
}

// SelectBoard は提示された複数の初期盤面から1つを選択します
func (ai *TrivialAI) SelectBoard(states []*game.GameState) int {
	return rand.Intn(len(states))
}

// SelectTurn は先手(0)か後手(1)かを選択します
func (ai *TrivialAI) SelectTurn(states []*game.GameState) int {
	return rand.Intn(2)
}

// Evaluate は現在の盤面の評価値を返します
// 単純に自分の領域の数値合計 - 相手の領域の数値合計 を返します
func (ai *TrivialAI) Evaluate(state *game.GameState, player int) float64 {
	var score float64
	opponent := 1 - player

	for y := 0; y < len(state.Board); y++ {
		for x := 0; x < len(state.Board[y]); x++ {
			value := float64(state.Board[y][x])
			switch state.Colors[y][x] {
			case player:
				score += value
			case opponent:
				score -= value
			}
		}
	}

	return score
}
