package random

import (
	"math/rand"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

type RandomAI struct{}

func New() *RandomAI {
	return &RandomAI{}
}

// SelectBoard は提示された複数の初期盤面から1つを選択します
func (ai *RandomAI) SelectBoard(states []*game.GameState) int {
	return rand.Intn(len(states))
}

// SelectTurn は先手(0)か後手(1)かを選択します
func (ai *RandomAI) SelectTurn(states []*game.GameState) int {
	return rand.Intn(2)
}

// Evaluate は現在の盤面の評価値を返します
// 単純に自分の領域の数値合計 - 相手の領域の数値合計 を返します
func (ai *RandomAI) Evaluate(state *game.GameState, player int) float64 {
	var score float64
	score = 0

	return score
}
