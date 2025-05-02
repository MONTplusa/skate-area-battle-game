package game

// AI はゲーム用エージェントのインターフェース
type AI interface {
	// 複数の初期盤面から使用するものを選ぶ
	SelectBoard(states []*GameState) int
	// 先後を選択する
	SelectTurn(states []*GameState) int
	// プレイヤーが一手を適用した後の盤面を評価
	Evaluate(state *GameState, player int) float64
}
