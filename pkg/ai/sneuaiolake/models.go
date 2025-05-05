package sneuaiolake

import (
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
	Weights      [][][]float64
}

func DefaultNetworkConfig() NetworkConfig {
	return NetworkConfig{
		Name:         "default",
		HiddenLayers: []int{64, 32}, // より小さいネットワーク
		Weights:      nil,
	}
}

// Sneuaiolake implements the game.AI interface with neural network evaluation
type Sneuaiolake struct {
	network *deep.Neural
	config  NetworkConfig
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
			switch color {
			case -1:
				features[idx] = 0
			case player:
				features[idx] = 1
			default:
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

func getFeatureSize() int {
	return 20*20*3 + 4 // Board, colors, rocks + player positions
}

func (config *NetworkConfig) export() error {
	path := fmt.Sprintf("pkg/ai/sneuaiolake/networks/%s.go", config.Name)
	// Goソースコードのテンプレートを定義
	tmpl := template.Must(template.New("configFile").Parse(`package networks
import "github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake"
func LoadConfig_{{.Name}}() sneuaiolake.NetworkConfig {
	return sneuaiolake.NetworkConfig{
		Name: "{{.Name}}",
		InputSize:    {{.InputSize}},
		HiddenLayers: []int{ {{range $i, $v := .HiddenLayers}}{{if $i}}, {{end}}{{$v}}{{end}} },
		Weights: [][][]float64{
			{{range $layer := .Weights}}{
				{{range $neuron := $layer}}{
					{{range $i, $weight := $neuron}}{{printf "%.6f" $weight}},{{end}}
				},{{end}}
			},{{end}}
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

	fmt.Printf("Config saved to %s\n", path)

	return nil
}
