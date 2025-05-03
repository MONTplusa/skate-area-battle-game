package sneuaiolake

import (
	"encoding/json"
	"fmt"
	"os"
	"text/template"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
	"github.com/patrikeh/go-deep"
)

// NetworkConfig defines the neural network architecture
type NetworkConfig struct {
	Name         string
	InputSize    int
	HiddenLayers []int
	LearningRate float64
	Weights      [][][]float64
}

func DefaultNetworkConfig() NetworkConfig {
	return NetworkConfig{
		Name:         "default",
		InputSize:    20*20*3 + 4,   // Board, colors, rocks + player positions
		HiddenLayers: []int{64, 32}, // より小さいネットワーク
		LearningRate: 0.01,
		Weights:      nil,
	}
}

// Sneuaiolake implements the game.AI interface with neural network evaluation
type Sneuaiolake struct {
	network     *deep.Neural
	config      NetworkConfig
	temperature float64 // For exploration during training
}

// New creates a new Sneuaiolake AI with optional pre-trained weights
func New(config NetworkConfig) (*Sneuaiolake, error) {
	// Create neural network
	layers := []int{config.InputSize}
	layers = append(layers, config.HiddenLayers...)
	layers = append(layers, 1) // Output: single evaluation score

	network := deep.NewNeural(&deep.Config{
		Inputs:     config.InputSize,
		Layout:     layers[1:],
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeRegression, // 回帰モードに変更（より高速）
		Weight:     deep.NewNormal(0.0, 0.1),
		Bias:       true,
	})

	// Apply loaded weights if any
	if config.Weights != nil {
		network.ApplyWeights(config.Weights)
	}

	return &Sneuaiolake{
		network:     network,
		config:      config,
		temperature: 1.0,
	}, nil
}

func (s *Sneuaiolake) Name() string {
	return fmt.Sprintf("sneuaiolake (%s)", s.config.Name)
}

// SetTemperature sets the exploration temperature for training
func (s *Sneuaiolake) SetTemperature(temp float64) {
	s.temperature = temp
}

// SetLearningRate sets the learning rate for the neural network
func (s *Sneuaiolake) SetLearningRate(lr float64) {
	s.config.LearningRate = lr
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

// Feature normalization: 特徴量を[-1, 1]の範囲に正規化
func normalizeFeature(value, min, max float64) float64 {
	if max == min {
		return 0.0
	}
	normalized := 2.0*(value-min)/(max-min) - 1.0
	if normalized < -1.0 {
		return -1.0
	}
	if normalized > 1.0 {
		return 1.0
	}
	return normalized
}

// Helper functions for state conversion, weight loading etc.
func stateToFeatures(state *game.GameState, player int) []float64 {
	boardSize := len(state.Board)
	features := make([]float64, boardSize*boardSize*3+4)

	// Normalize all features to range [-1, 1]
	idx := 0

	// Board values (normalized to [-1, 1])
	for y := 0; y < boardSize; y++ {
		for x := 0; x < boardSize; x++ {
			features[idx] = normalizeFeature(float64(state.Board[y][x]), 0, 100)
			idx++
		}
	}

	// Colors: remap to be from perspective of current player
	// -1 (none) -> 0, player -> 1, opponent -> -1
	for y := 0; y < boardSize; y++ {
		for x := 0; x < boardSize; x++ {
			color := state.Colors[y][x]
			if color == -1 {
				features[idx] = 0
			} else if color == player {
				features[idx] = 1
			} else {
				features[idx] = -1
			}
			idx++
		}
	}

	// Rocks (1 for rock, 0 for no rock)
	for y := 0; y < boardSize; y++ {
		for x := 0; x < boardSize; x++ {
			if state.Rocks[y][x] {
				features[idx] = 1
			} else {
				features[idx] = 0
			}
			idx++
		}
	}

	// Player positions (normalized by board size to [-1, 1])
	if player == 0 {
		features[idx] = normalizeFeature(float64(state.Player0.X), 0, float64(boardSize-1))
		idx++
		features[idx] = normalizeFeature(float64(state.Player0.Y), 0, float64(boardSize-1))
		idx++
		features[idx] = normalizeFeature(float64(state.Player1.X), 0, float64(boardSize-1))
		idx++
		features[idx] = normalizeFeature(float64(state.Player1.Y), 0, float64(boardSize-1))
	} else {
		features[idx] = normalizeFeature(float64(state.Player1.X), 0, float64(boardSize-1))
		idx++
		features[idx] = normalizeFeature(float64(state.Player1.Y), 0, float64(boardSize-1))
		idx++
		features[idx] = normalizeFeature(float64(state.Player0.X), 0, float64(boardSize-1))
		idx++
		features[idx] = normalizeFeature(float64(state.Player0.Y), 0, float64(boardSize-1))
	}

	return features
}

// Fallback heuristic evaluation
func heuristicEvaluation(state *game.GameState, player int) float64 {
	boardSize := len(state.Board)
	playerScore := 0.0
	opponentScore := 0.0

	for y := 0; y < boardSize; y++ {
		for x := 0; x < boardSize; x++ {
			if state.Colors[y][x] == player {
				playerScore += float64(state.Board[y][x])
			} else if state.Colors[y][x] == 1-player {
				opponentScore += float64(state.Board[y][x])
			}
		}
	}

	// Add mobility score - count legal moves
	playerMoves := len(state.LegalMoves(player))
	opponentMoves := len(state.LegalMoves(1 - player))

	// Normalize to [-1, 1] range for consistency with neural network output
	totalScore := playerScore + opponentScore
	if totalScore == 0 {
		totalScore = 1.0 // Avoid division by zero
	}
	normalizedScore := (playerScore - opponentScore) / totalScore

	totalMoves := playerMoves + opponentMoves
	if totalMoves == 0 {
		totalMoves = 1.0 // Avoid division by zero
	}
	mobilityFactor := float64(playerMoves-opponentMoves) / float64(totalMoves)

	// Combine score and mobility
	return 0.7*normalizedScore + 0.3*mobilityFactor
}

func loadConfig(config *NetworkConfig, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, config)
}

func saveConfig(config NetworkConfig) error {
	path := fmt.Sprintf("pkg/ai/sneuaiolake/networks/%s.go", config.Name)
	// Goソースコードのテンプレートを定義
	tmpl := template.Must(template.New("configFile").Parse(`package networks
import "github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake"
func LoadConfig_{{.Name}}() sneuaiolake.NetworkConfig {
	return sneuaiolake.NetworkConfig{
		Name: "{{.Name}}",
		InputSize:    {{.InputSize}},
		HiddenLayers: []int{ {{range $i, $v := .HiddenLayers}}{{if $i}}, {{end}}{{$v}}{{end}} },
		LearningRate: {{.LearningRate}},
		Weights: [][][]float64{
			{{range $layer := .Weights}}
			{
				{{range $neuron := $layer}}
				{
					{{range $i, $weight := $neuron}}
						{{printf "%.6f" $weight}},
					{{end}}
				},
				{{end}}
			},
			{{end}}
		},
	}
}
`))

	// ファイルを作成
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("ファイル作成エラー: %w", err)
	}
	defer file.Close()

	// テンプレートを実行してファイルに書き込む
	err = tmpl.Execute(file, config)
	if err != nil {
		return fmt.Errorf("テンプレート実行エラー: %w", err)
	}

	return nil
}
