// cmd/gamerunner/main.go
//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/montplusa/skate-area-battle-game/pkg/ai/montplusa"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/montplusai"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/random"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/staiolake"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/statiolake"
	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

// AI ファクトリーの登録
var aiFactory = []func() game.AI{
	func() game.AI { return statiolake.New() },
	func() game.AI { return montplusa.New() },
	func() game.AI { return staiolake.New() },
	func() game.AI { return montplusai.New() },
	func() game.AI { return random.New() },
	func() game.AI { return sneuaiolake.New() },
}
var aiList = map[string]int{}

// getAIList returns the available AI names as JSON
func getAIList(this js.Value, args []js.Value) interface{} {
	names := make([]string, 0, len(aiFactory))
	for k := range aiFactory {
		name := aiFactory[k]().Name()
		aiList[name] = k
		names = append(names, name)
	}
	b, _ := json.Marshal(names)
	return string(b)
}

// runBattle executes a battle between two selected AIs
func runBattle(this js.Value, args []js.Value) interface{} {
	// デフォルトAI名
	name1, name2 := "statiolake", "montplusa"
	if len(args) >= 2 {
		name1 = args[0].String()
		name2 = args[1].String()
	}

	// AI インスタンス生成
	f1 := aiFactory[aiList[name1]]
	f2 := aiFactory[aiList[name2]]
	ai1 := f1()
	ai2 := f2()

	// GameRunner の実行
	gr := game.NewGameRunner(ai1, ai2)
	result := gr.Run()

	// 結果を JSON 化
	b, _ := json.Marshal(result)
	return string(b)
}

func main() {
	js.Global().Set("runBattle", js.FuncOf(runBattle))
	js.Global().Set("getAIList", js.FuncOf(getAIList))
	select {}
}
