package sneuaiolake

import (
	"embed"
	"fmt"
	"log"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

//go:embed model.onnx
var modelFS embed.FS

const (
	BOARD_SIZE   = 20
	NUM_CHANNELS = 6
)

// NeuralNetAI はニューラルネットワークを使用したAI
type NeuralNetAI struct {
	model   *onnx.Model
	backend *gorgonnx.Graph
}

// New は新しいNeuralNetAIインスタンスを作成します
func New() *NeuralNetAI {
	// モデルファイルの読み込み
	modelData, err := modelFS.ReadFile("model.onnx")
	if err != nil {
		log.Fatalf("モデルファイルの読み込みに失敗: %v", err)
	}

	// ONNX モデルの作成
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)

	// モデルのデシリアライズ
	err = model.UnmarshalBinary(modelData)
	if err != nil {
		log.Fatalf("モデルのデシリアライズに失敗: %v", err)
	}

	return &NeuralNetAI{
		model:   model,
		backend: backend,
	}
}

// Name はAIの名前を返します
func (ai *NeuralNetAI) Name() string {
	return "sneuaiolake"
}

// SelectBoard は提示された複数の初期盤面から1つを選択します
func (ai *NeuralNetAI) SelectBoard(states []*game.GameState) int {
	// 各盤面を評価し、最も評価値が高いものを選択
	bestScore := -1.0
	bestIndex := 0

	for i, state := range states {
		// プレイヤー0の視点で評価
		score := ai.Evaluate(state, 0)
		if score > bestScore {
			bestScore = score
			bestIndex = i
		}
	}

	return bestIndex
}

// SelectTurn は先手(0)か後手(1)かを選択します
func (ai *NeuralNetAI) SelectTurn(states []*game.GameState) int {
	// 先手と後手それぞれの場合の評価値の平均を計算
	var firstSum, secondSum float64
	for _, state := range states {
		// 先手の場合の評価値
		firstSum += ai.Evaluate(state, 0)

		// 後手の場合の評価値
		secondSum += ai.Evaluate(state, 1)
	}

	firstAvg := firstSum / float64(len(states))
	secondAvg := secondSum / float64(len(states))

	// 評価値が高い方を選択
	if firstAvg >= secondAvg {
		return 0 // 先手
	}
	return 1 // 後手
}

// Evaluate は現在の盤面の評価値を返します
func (ai *NeuralNetAI) Evaluate(state *game.GameState, player int) float64 {
	// 入力テンソルの作成
	inputTensor, err := ai.createInputTensor(state, player)
	if err != nil {
		log.Printf("入力テンソルの作成に失敗: %v", err)
		return 0.0
	}

	// 推論の実行
	err = ai.model.SetInput(0, inputTensor)
	if err != nil {
		log.Printf("入力の設定に失敗: %v", err)
		return 0.0
	}

	err = ai.backend.Run()
	if err != nil {
		log.Printf("推論の実行に失敗: %v", err)
		return 0.0
	}

	// 出力の取得
	outputs, err := ai.model.GetOutputTensors()
	if err != nil {
		log.Printf("出力の取得に失敗: %v", err)
		return 0.0
	}

	if len(outputs) == 0 {
		log.Printf("出力が空です")
		return 0.0
	}

	// 出力テンソルから評価値を取得
	output := outputs[0]
	value, err := getScalarFromTensor(output)
	if err != nil {
		log.Printf("評価値の取得に失敗: %v", err)
		return 0.0
	}

	return value
}

// createInputTensor はGameStateから入力テンソルを作成します
func (ai *NeuralNetAI) createInputTensor(state *game.GameState, player int) (tensor.Tensor, error) {
	// 入力データの初期化 (バッチサイズ1, 20x20の盤面, 6チャンネル)
	inputData := make([]float32, 1*BOARD_SIZE*BOARD_SIZE*NUM_CHANNELS)

	// 次のプレイヤーが1となるように入力を作成する
	// つまり、player == 1の場合はプレイヤー0とプレイヤー1を逆に読み替える必要がある
	player0 := player
	player1 := 1 - player

	// ボードの最大値を取得して正規化
	boardMax := 0
	for _, row := range state.Board {
		for _, cell := range row {
			if cell > boardMax {
				boardMax = cell
			}
		}
	}

	// 入力データの作成
	for y := 0; y < BOARD_SIZE; y++ {
		for x := 0; x < BOARD_SIZE; x++ {
			// インデックスの計算 (NCHW形式)
			// チャンネル0: ボードの数値を0-1に正規化
			idx := (0 * BOARD_SIZE * BOARD_SIZE) + (y * BOARD_SIZE) + x
			if boardMax > 0 {
				inputData[idx] = float32(state.Board[y][x]) / float32(boardMax)
			}

			// チャンネル1: プレイヤー0の色
			// チャンネル2: プレイヤー1の色
			idx1 := (1 * BOARD_SIZE * BOARD_SIZE) + (y * BOARD_SIZE) + x
			idx2 := (2 * BOARD_SIZE * BOARD_SIZE) + (y * BOARD_SIZE) + x
			if state.Colors[y][x] == player0 {
				inputData[idx1] = 1.0
			} else if state.Colors[y][x] == player1 {
				inputData[idx2] = 1.0
			}

			// チャンネル3: 岩の位置
			idx3 := (3 * BOARD_SIZE * BOARD_SIZE) + (y * BOARD_SIZE) + x
			if state.Rocks[y][x] {
				inputData[idx3] = 1.0
			}
		}
	}

	// チャンネル4: プレイヤー0の位置
	player0X := state.Player0.X
	player0Y := state.Player0.Y
	if player != 0 {
		player0X = state.Player1.X
		player0Y = state.Player1.Y
	}
	idx4 := (4 * BOARD_SIZE * BOARD_SIZE) + (player0Y * BOARD_SIZE) + player0X
	inputData[idx4] = 1.0

	// チャンネル5: プレイヤー1の位置
	player1X := state.Player1.X
	player1Y := state.Player1.Y
	if player != 0 {
		player1X = state.Player0.X
		player1Y = state.Player0.Y
	}
	idx5 := (5 * BOARD_SIZE * BOARD_SIZE) + (player1Y * BOARD_SIZE) + player1X
	inputData[idx5] = 1.0

	// テンソルの作成 (NCHW形式)
	inputTensor := tensor.New(
		tensor.WithShape(1, BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS),
		tensor.WithBacking(inputData),
	)

	return inputTensor, nil
}

// getScalarFromTensor はテンソルからスカラー値を取得します
func getScalarFromTensor(t tensor.Tensor) (float64, error) {
	// テンソルの形状を取得
	shape := t.Shape()
	if len(shape) == 0 {
		// スカラーの場合
		value := t.ScalarValue()
		switch v := value.(type) {
		case float32:
			return float64(v), nil
		case float64:
			return v, nil
		default:
			return 0, fmt.Errorf("未対応の型: %T", v)
		}
	} else if len(shape) == 1 && shape[0] == 1 {
		// 1次元で要素数1の場合
		value, err := t.At(0)
		if err != nil {
			return 0, err
		}
		switch v := value.(type) {
		case float32:
			return float64(v), nil
		case float64:
			return v, nil
		default:
			return 0, fmt.Errorf("未対応の型: %T", v)
		}
	} else if len(shape) == 2 && shape[0] == 1 && shape[1] == 1 {
		// 2次元で要素数1の場合
		value, err := t.At(0, 0)
		if err != nil {
			return 0, err
		}
		switch v := value.(type) {
		case float32:
			return float64(v), nil
		case float64:
			return v, nil
		default:
			return 0, fmt.Errorf("未対応の型: %T", v)
		}
	}

	return 0, fmt.Errorf("スカラー値を取得できません: %v", shape)
}
