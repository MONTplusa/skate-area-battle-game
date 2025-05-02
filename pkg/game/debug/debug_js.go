//go:build js && wasm
// +build js,wasm

package debug

import (
	"fmt"
	"syscall/js"
)

func Log(args ...interface{}) {
	js.Global().Get("console").Call("log", fmt.Sprint(args...))
}
