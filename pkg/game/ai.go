package game

// AI インターフェースは、ゲームの意思決定に必要なメソッドを定義します
type AI interface {
	// SelectBoard は提示された複数の初期盤面から1つを選択します
	// states: 選択可能な盤面の配列
	// 戻り値: 選択した盤面のインデックス
	SelectBoard(states []*GameState) int

	// SelectTurn は先手(0)か後手(1)かを選択します
	// states: 現在の盤面の配列
	// 戻り値: 選択したターン（0:先手、1:後手）
	SelectTurn(states []*GameState) int

	// Evaluate は現在の盤面の評価値を返します
	// state: 評価する盤面
	// player: 評価するプレイヤー（0または1）
	// 戻り値: 盤面の評価値（大きいほど有利）
	Evaluate(state *GameState, player int) float64
}
