//go:build js && wasm
// +build js,wasm

package debug

import (
	"fmt"
	"syscall/js"
)

func Log(format string, args ...any) {
	js.Global().Get("console").Call("log", fmt.Sprintf(format, args...))
}
