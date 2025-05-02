//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/montplusa/skate-area-battle-game/pkg/ai/staiolake"
	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

func runBattle(this js.Value, args []js.Value) interface{} {
	// 1) AI の初期化
	ai1 := staiolake.New()
	ai2 := staiolake.New()

	// 2) GameRunner の実行
	gr := game.NewGameRunner(ai1, ai2)
	result := gr.Run() // BattleResult 型を返す

	// 3) JSON 文字列にシリアライズ
	b, _ := json.Marshal(result)
	return string(b)
}

func main() {
	js.Global().Set("runBattle", js.FuncOf(runBattle))
	select {} // ブロック
}
