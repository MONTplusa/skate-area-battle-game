package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"sync"

	"github.com/montplusa/skate-area-battle-game/pkg/ai/random"
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

// 対戦タスクの構造体
type battleTask struct {
	gameIndex int
	seqNum    int
}

// 対戦結果の構造体
type battleResult struct {
	gameIndex int
	winner    int
	result    game.BattleResult
}

// ワーカー関数
func worker(id int, tasks <-chan battleTask, results chan<- battleResult, outputDir, outputPrefix string, noOutput bool, wg *sync.WaitGroup) {
	defer wg.Done()

	for task := range tasks {
		ai1 := random.New()
		ai2 := random.New()

		gr := game.NewGameRunner(ai1, ai2)
		result := gr.Run()

		winner := getWinner(&result)

		if !noOutput {
			// 結果をJSONに変換（インデントなし）
			jsonData, err := json.Marshal(result)
			if err != nil {
				fmt.Printf("エラー: JSONの変換に失敗しました: %v\n", err)
				continue
			}

			// ファイル名の生成（5桁のゼロ詰め連番）
			filename := filepath.Join(outputDir, fmt.Sprintf("%s_%05d.json", outputPrefix, task.seqNum))

			// ファイルへの書き込み
			err = os.WriteFile(filename, jsonData, 0644)
			if err != nil {
				fmt.Printf("エラー: ファイルの書き込みに失敗しました: %v\n", err)
			}
		}

		results <- battleResult{
			gameIndex: task.gameIndex,
			winner:    winner,
			result:    result,
		}

		fmt.Printf("対戦 %d が完了しました（ワーカー %d）\n", task.gameIndex, id)
	}
}

func main() {
	// コマンドライン引数の解析
	outputDir := flag.String("output", "output", "出力ディレクトリ名")
	outputPrefix := flag.String("output-prefix", "", "出力ファイル名のプレフィックス")
	noOutput := flag.Bool("no-output", false, "出力しない")
	games := flag.Int("games", 1, "実行する試合数")
	numWorkers := flag.Int("workers", runtime.NumCPU(), "ワーカー数")
	flag.Parse()

	// 出力プレフィックスが指定されていない場合はエラー
	if !*noOutput && *outputPrefix == "" {
		fmt.Println("エラー: --output-prefix は必須です")
		flag.Usage()
		os.Exit(1)
	}

	if !*noOutput {
		// 出力ディレクトリの作成
		err := os.MkdirAll(*outputDir, 0755)
		if err != nil {
			fmt.Printf("エラー: 出力ディレクトリの作成に失敗しました: %v\n", err)
			os.Exit(1)
		}
	}

	// 既存ファイルの最大連番を取得
	maxSeq, err := findMaxSequenceNumber(*outputDir, *outputPrefix)
	if err != nil {
		fmt.Printf("警告: 既存ファイルの確認中にエラーが発生しました: %v\n", err)
	}
	startSeq := maxSeq + 1
	fmt.Printf("連番 %05d から開始します\n", startSeq)

	fmt.Printf("自己対戦を %d 回実行します（ワーカー数: %d）\n", *games, *numWorkers)

	// チャネルの作成
	tasks := make(chan battleTask, *games)
	results := make(chan battleResult, *games)

	// ワーカープールの作成
	var wg sync.WaitGroup
	for i := 0; i < *numWorkers; i++ {
		wg.Add(1)
		go worker(i, tasks, results, *outputDir, *outputPrefix, *noOutput, &wg)
	}

	// タスクの送信
	go func() {
		for i := 0; i < *games; i++ {
			tasks <- battleTask{
				gameIndex: i,
				seqNum:    startSeq + i,
			}
		}
		close(tasks)
	}()

	// 結果の収集
	wins := []int{0, 0}
	for i := 0; i < *games; i++ {
		result := <-results
		if result.winner != -1 {
			wins[result.winner]++
		}
	}

	// すべてのワーカーの終了を待つ
	wg.Wait()

	fmt.Println("すべての対戦が完了しました")
	fmt.Printf("勝利数: P0: %d, P1: %d\n", wins[0], wins[1])
}

func getWinner(result *game.BattleResult) int {
	// スコアを計算
	scores := []int{0, 0}
	board := result.FinalState.Board

	for i := range board {
		for j := range board[i] {
			color := result.FinalState.Colors[i][j]
			if color == -1 {
				continue
			}
			scores[color] += board[i][j]
		}
	}

	if scores[0] > scores[1] {
		return 0
	}
	if scores[0] < scores[1] {
		return 1
	}
	return -1 // 引き分け
}
