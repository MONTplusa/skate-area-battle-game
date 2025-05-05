package sneuaiolake

import (
	"fmt"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
	"github.com/patrikeh/go-deep"
)

func New(config NetworkConfig) (*Sneuaiolake, error) {
	// Create neural network
	var layers []int
	layers = append(layers, config.HiddenLayers...)
	layers = append(layers, 1) // Output: single evaluation score

	network := deep.NewNeural(&deep.Config{
		Inputs:     getFeatureSize(),
		Layout:     layers,
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeRegression, // 回帰モードに変更（より高速）
		Weight:     deep.NewUniform(0.2, 0.0),
		Bias:       true,
	})

	// Apply loaded weights if any
	if config.Weights != nil {
		network.ApplyWeights(config.Weights)
	}

	return &Sneuaiolake{
		network: network,
		config:  config,
	}, nil
}

func (s *Sneuaiolake) Name() string {
	return fmt.Sprintf("sneuaiolake (%s)", s.config.Name)
}

// SelectBoard implements the AI interface
func (s *Sneuaiolake) SelectBoard(states []*game.GameState) int {
	// Choose the board that gives the most balanced starting position
	// or highest expected value
	bestScore := -1e9
	bestIndex := 0

	for i, state := range states {
		// Evaluate both as player 0 and as player 1
		score0 := s.Evaluate(state, 0)
		score1 := s.Evaluate(state, 1)

		// Choose the most balanced board (minimum absolute difference)
		diff := score0 - score1
		if diff < 0 {
			diff = -diff
		}

		// Prefer balanced positions (smaller diff is better)
		balanceScore := 1000 - diff
		if balanceScore > bestScore {
			bestScore = balanceScore
			bestIndex = i
		}
	}

	return bestIndex
}

// SelectTurn implements the AI interface
func (s *Sneuaiolake) SelectTurn(states []*game.GameState) int {
	// Choose the turn with higher expected value
	state := states[0] // We only need one state to evaluate

	score0 := s.Evaluate(state, 0)
	score1 := s.Evaluate(state, 1)

	if score0 > score1 {
		return 0 // Select first player
	}
	return 1 // Select second player
}

// Evaluate implements the AI interface
func (s *Sneuaiolake) Evaluate(state *game.GameState, player int) float64 {
	// Convert state to features
	features := stateToFeatures(state, player)

	// Get network prediction
	prediction := s.network.Predict(features)

	// Return the normalized evaluation score
	return prediction[0]
}
