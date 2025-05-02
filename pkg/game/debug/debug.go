//go:build !js || !wasm
// +build !js !wasm

package debug

func Log(args ...interface{}) {
	// 非WASM環境では何もしない
}
