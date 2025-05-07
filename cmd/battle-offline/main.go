package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"sync"

	"github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake"
	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

// 指定されたディレクトリ内の同じプレフィックスを持つファイルの最大連番を取得する
func findMaxSequenceNumber(dir, prefix string) (int, error) {
	// ディレクトリが存在しない場合は0を返す
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return 0, nil
	}

	files, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}

	// プレフィックス_NNNNN.json の形式にマッチする正規表現
	pattern := regexp.MustCompile(fmt.Sprintf(`^%s_(\d{5})\.json$`, regexp.QuoteMeta(prefix)))
	maxSeq := 0

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		matches := pattern.FindStringSubmatch(file.Name())
		if len(matches) == 2 {
			seq, err := strconv.Atoi(matches[1])
			if err != nil {
				continue
			}
			if seq > maxSeq {
				maxSeq = seq
			}
		}
	}

	return maxSeq, nil
}

func main() {
	// コマンドライン引数の解析
	outputDir := flag.String("output", "output", "出力ディレクトリ名")
	outputPrefix := flag.String("output-prefix", "", "出力ファイル名のプレフィックス")
	games := flag.Int("games", 1, "実行する試合数")
	flag.Parse()

	// 出力プレフィックスが指定されていない場合はエラー
	if *outputPrefix == "" {
		fmt.Println("エラー: --output-prefix は必須です")
		flag.Usage()
		os.Exit(1)
	}

	// 出力ディレクトリの作成
	err := os.MkdirAll(*outputDir, 0755)
	if err != nil {
		fmt.Printf("エラー: 出力ディレクトリの作成に失敗しました: %v\n", err)
		os.Exit(1)
	}

	// 既存ファイルの最大連番を取得
	maxSeq, err := findMaxSequenceNumber(*outputDir, *outputPrefix)
	if err != nil {
		fmt.Printf("警告: 既存ファイルの確認中にエラーが発生しました: %v\n", err)
	}
	startSeq := maxSeq + 1
	fmt.Printf("連番 %05d から開始します\n", startSeq)

	// 並列実行のためのWaitGroup
	var wg sync.WaitGroup
	// 結果の書き込みを同期するためのミューテックス
	var mu sync.Mutex

	fmt.Printf("自己対戦を %d 回実行します\n", *games)

	// 指定された回数のバトルを実行
	for i := 0; i < *games; i++ {
		wg.Add(1)
		go func(gameIndex int) {
			defer wg.Done()

			// 自己対戦を実行
			ai1 := sneuaiolake.New()
			ai2 := sneuaiolake.New()
			gr := game.NewGameRunner(ai1, ai2)
			result := gr.Run()

			// 結果をJSONに変換（インデントなし）
			jsonData, err := json.Marshal(result)
			if err != nil {
				mu.Lock()
				fmt.Printf("エラー: JSONの変換に失敗しました: %v\n", err)
				mu.Unlock()
				return
			}

			// ファイル名の生成（5桁のゼロ詰め連番）
			seqNum := startSeq + gameIndex
			filename := filepath.Join(*outputDir, fmt.Sprintf("%s_%05d.json", *outputPrefix, seqNum))

			// ファイルへの書き込み
			mu.Lock()
			err = os.WriteFile(filename, jsonData, 0644)
			if err != nil {
				fmt.Printf("エラー: ファイルの書き込みに失敗しました: %v\n", err)
			} else {
				fmt.Printf("対戦結果を %s に保存しました\n", filename)
			}
			mu.Unlock()
		}(i)
	}

	// すべてのゴルーチンの終了を待つ
	wg.Wait()
	fmt.Println("すべての対戦が完了しました")
}
