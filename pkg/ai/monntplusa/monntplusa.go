package monntplusa

import (
	"math/rand"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

// NeuralNet はモデル推論用インターフェースです。
type NeuralNet interface {
	// Predict は (policy, value) を返します。
	Predict(state *game.GameState) (policy []float64, value float64, err error)
}

// MoNNtplusaAI は MLP ネットワークを用いた評価関数を提供します。
type MoNNtplusaAI struct {
	nn NeuralNet
}

func (ai *MoNNtplusaAI) Name() string {
	return "moNNtplusa"
}

// New は MLPNeuralNet を初期化し、MontplusaAI を返します。
func New() *MoNNtplusaAI {
	return &MoNNtplusaAI{nn: NewCNNNeuralNet()}
}

// SelectBoard は複数の初期盤面から 1 つを選択します（常に 0）。
func (ai *MoNNtplusaAI) SelectBoard(states []*game.GameState) int {
	return 0
}

// SelectTurn は先手(0)か後手(1)かをランダムに選択します。
func (ai *MoNNtplusaAI) SelectTurn(states []*game.GameState) int {
	return rand.Intn(2)
}

// Evaluate は指定プレイヤー視点で盤面を評価し、value を返します。
func (ai *MoNNtplusaAI) Evaluate(st *game.GameState, player int) float64 {
	_, v, err := ai.nn.Predict(st)
	if err != nil {
		return 0
	}
	return v
}
