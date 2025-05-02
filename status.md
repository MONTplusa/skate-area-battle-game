現状のスキャフォールドでは、以下の機能を実装済み／未実装になっています。

---

## 実装済み

1. **ディレクトリ構成**  
   - `cmd/skate-battle/`：WASM エントリーポイント 
   - `pkg/game/`：盤面・ランナー・AI インターフェース
   - `pkg/ai/random/`：サンプル RandomAI 実装
   - `docs/`：GitHub Pages 用の HTML と WASM

2. **`cmd/skate-battle/main.go`**  
   - `// +build js,wasm` タグによるビルド制約  
   - `runBattle` を `syscall/js` 経由でエクスポート  
   - JSON シリアライズしてブラウザへ返却

3. **`pkg/game/ai.go`**  
   - `AI` インターフェース定義（`SelectBoard`／`SelectTurn`／`Evaluate`）

4. **`pkg/ai/random/random.go`**  
   - `RandomAI` が `AI` を実装  
   - ランダムに盤面選択／先後選択  
   - 簡易評価（塗り点差）

5. **`pkg/game/state.go`**  
   - `GameState` 構造体定義  
   - `Clone()`, `GenerateInitialStates()` 実装  
   - 合法手列挙 `LegalMoves()`  
   - 一手適用 `ApplyMove()`

6. **`pkg/game/runner.go`**  
   - `GameRunner` による対戦ループ  
   - 初期盤面サンプル生成 → 盤面選択 → 先後選択  
   - 手番交互ループで最良手を `AI.Evaluate()` で選択  
   - `BattleResult.Moves` に一連の手を記録

7. **`docs/index.html`**  
   - WASM ランタイム読み込み＆初期化  
   - 「対戦」ボタンで `runBattle()` 呼び出し  
   - 結果(JSON) の受け取り

---

## 未実装

1. **Canvas 描画ロジック**  
   - `res.Moves` を受け取って盤面を絵として逐次アニメーション表示する部分  
2. **CI／自動ビルド**  
   - `.github/workflows/build.yml` での WASM ビルド & `wasm_exec.js` コピー  
3. **追加 AI 実装**  
   - Minimax/MCTS など、より高度な AI の雛形  
4. **`go.mod`・`go.sum` の整備**  
   - モジュールパス設定・依存管理  
5. **細かいユーティリティ**  
   - 終了判定（両者スキップ時以外の条件）  
   - パフォーマンス最適化やログ出力  

---

このあたりを順に埋めていけば、最終的なクライアント完結型のビジュアライザ付き対戦ゲームが完成します。次に着手したい部分をお知らせください！
