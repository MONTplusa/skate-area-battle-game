package monntplusa

import (
	"math"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

const (
	boardN    = 20
	inCh      = 5
	outCh     = 8 // conv_pre out channels
	resBlocks = 3
)

// CNNNeuralNet performs forward inference using embedded weights (cnn_weights.go).
// Only value head is computed because Evaluate() ignores policy.
// All math is done in float64 with simple loops – slow but portable.
type CNNNeuralNet struct{}

func NewCNNNeuralNet() *CNNNeuralNet { return &CNNNeuralNet{} }

// Predict returns nil policy and value scalar.
func (c *CNNNeuralNet) Predict(st *game.GameState) ([]float64, float64, error) {
	// -------------- build input 5×N×N --------------
	in := make([][][]float64, inCh)
	for ch := 0; ch < inCh; ch++ {
		in[ch] = make([][]float64, boardN)
		for y := 0; y < boardN; y++ {
			in[ch][y] = make([]float64, boardN)
		}
	}
	for y := 0; y < boardN; y++ {
		for x := 0; x < boardN; x++ {
			in[0][y][x] = float64(st.Board[y][x])
			in[1][y][x] = float64(st.Colors[y][x]) // -1/0/1
			if st.Rocks[y][x] {
				in[2][y][x] = 1
			}
		}
	}
	in[3][st.Player0.Y][st.Player0.X] = 1
	in[4][st.Player1.Y][st.Player1.X] = 1

	// -------------- Conv Pre --------------
	feat := convBlock(in, ConvPreW, ConvPreB) // shape 64×N×N

	// -------------- Residual Blocks --------------
	for i := 0; i < resBlocks; i++ {
		feat = residualBlock(i, feat)
	}

	// -------------- Value Head --------------
	// 1×1 conv to 1 channel
	valMap := conv1x1(feat, ValW, ValB) // 1×N×N (just take channel 0)

	// flatten (N*N)
	flat := make([]float64, boardN*boardN)
	idx := 0
	for y := 0; y < boardN; y++ {
		for x := 0; x < boardN; x++ {
			flat[idx] = valMap[0][y][x]
			idx++
		}
	}
	// FC1 (hidden 64) + ReLU
	hidden := make([]float64, len(ValFC1B))
	for i := 0; i < len(hidden); i++ {
		sum := ValFC1B[i]
		for j, w := range ValFC1W[i] {
			sum += w * flat[j]
		}
		if sum < 0 {
			sum = 0
		}
		hidden[i] = sum
	}
	// FC2 -> scalar then tanh
	val := ValFC2B
	for i, w := range ValFC2W {
		val += w * hidden[i]
	}
	val = math.Tanh(val)

	return nil, val, nil
}

// ---------------- helper functions ----------------

// convBlock: 3×3 conv + ReLU (BN folded: y=gamma*x+beta)
func convBlock(input [][][]float64, weights [][][][]float64, bias []float64) [][][]float64 {
	outCh := len(weights)
	out := make([][][]float64, outCh)
	for oc := 0; oc < outCh; oc++ {
		out[oc] = make([][]float64, boardN)
		for y := 0; y < boardN; y++ {
			row := make([]float64, boardN)
			for x := 0; x < boardN; x++ {
				sum := bias[oc]
				for ic := 0; ic < len(weights[oc]); ic++ {
					for ky := -1; ky <= 1; ky++ {
						yy := y + ky
						if yy < 0 || yy >= boardN {
							continue
						}
						for kx := -1; kx <= 1; kx++ {
							xx := x + kx
							if xx < 0 || xx >= boardN {
								continue
							}
							w := weights[oc][ic][ky+1][kx+1]
							sum += w * input[ic][yy][xx]
						}
					}
				}
				if sum < 0 {
					sum = 0
				}
				row[x] = sum
			}
			out[oc][y] = row
		}
	}
	return out
}

// residualBlock id i, uses exported weights
func residualBlock(i int, x [][][]float64) [][][]float64 {
	// First conv
	w1 := get4D("Res", i, "_W1")
	b1 := get1D("Res", i, "_B1")
	out1 := convBlock(x, w1, b1)
	// Second conv (no ReLU here)
	w2 := get4D("Res", i, "_W2")
	b2 := get1D("Res", i, "_B2")
	out2 := convBlock(out1, w2, b2)
	// element‑wise add & ReLU
	for oc := 0; oc < len(out2); oc++ {
		for y := 0; y < boardN; y++ {
			for x0 := 0; x0 < boardN; x0++ {
				s := out2[oc][y][x0] + x[oc][y][x0]
				if s < 0 {
					s = 0
				}
				out2[oc][y][x0] = s
			}
		}
	}
	return out2
}

// conv1x1 for value/policy heads
func conv1x1(input [][][]float64, w [][][][]float64, b []float64) [][][]float64 {
	outCh := len(w)
	out := make([][][]float64, outCh)
	for oc := 0; oc < outCh; oc++ {
		out[oc] = make([][]float64, boardN)
		for y := 0; y < boardN; y++ {
			row := make([]float64, boardN)
			for x := 0; x < boardN; x++ {
				sum := b[oc]
				for ic := 0; ic < len(w[oc]); ic++ {
					sum += w[oc][ic][0][0] * input[ic][y][x]
				}
				row[x] = math.Max(0, sum)
			}
			out[oc][y] = row
		}
	}
	return out
}

// ---------------- weight helpers ----------------
func get4D(prefix string, idx int, suffix string) [][][][]float64 {
	switch prefix {
	case "Res":
		switch suffix {
		case "_W1":
			return ResWeights1[idx]
		case "_W2":
			return ResWeights2[idx]
		}
	}
	return nil
}
func get1D(prefix string, idx int, suffix string) []float64 {
	switch prefix {
	case "Res":
		switch suffix {
		case "_B1":
			return ResBias1[idx]
		case "_B2":
			return ResBias2[idx]
		}
	}
	return nil
}

// Note: ResWeights* & ResBias* slices should be declared in cnn_weights.go manually (or by export script)
