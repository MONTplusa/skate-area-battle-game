class GameVisualizer {
    constructor(canvasId, gridSize = 20) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.gridSize = gridSize;
        this.cellSize = Math.min(
            (this.canvas.width - 40) / gridSize,
            (this.canvas.height - 40) / gridSize
        );
        this.offsetX = (this.canvas.width - this.cellSize * gridSize) / 2;
        this.offsetY = (this.canvas.height - this.cellSize * gridSize) / 2;

        // プレイヤーの色
        this.colors = {
            player0: '#ff4444',
            player1: '#4444ff',
            rock: '#666666',
            empty: '#ffffff',
            grid: '#cccccc',
            text: '#000000'
        };

        // テキスト表示用の設定
        this.statusY = 20;  // 上部のステータス表示位置
        this.turnY = 45;    // ターン数表示位置
        this.historyY = 70; // 履歴表示開始位置
        this.textX = 10;    // 左揃えテキストのX座標
        this.textRightX = this.canvas.width - 150; // 右揃えテキストのX座標
    }

    // グリッドの描画
    drawGrid() {
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 1;

        // 縦線
        for (let x = 0; x <= this.gridSize; x++) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.offsetX + x * this.cellSize, this.offsetY);
            this.ctx.lineTo(this.offsetX + x * this.cellSize, this.offsetY + this.gridSize * this.cellSize);
            this.ctx.stroke();
        }

        // 横線
        for (let y = 0; y <= this.gridSize; y++) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.offsetX, this.offsetY + y * this.cellSize);
            this.ctx.lineTo(this.offsetX + this.gridSize * this.cellSize, this.offsetY + y * this.cellSize);
            this.ctx.stroke();
        }
    }

    // マスの描画
    drawCell(x, y, value, owner, isRock) {
        const cellX = this.offsetX + x * this.cellSize;
        const cellY = this.offsetY + y * this.cellSize;

        // マスの塗りつぶし
        this.ctx.fillStyle = isRock ? this.colors.rock :
            owner === 0 ? this.colors.player0 :
            owner === 1 ? this.colors.player1 :
            this.colors.empty;
        this.ctx.fillRect(cellX, cellY, this.cellSize, this.cellSize);

        // 数値の描画
        if (!isRock) {
            this.ctx.fillStyle = this.colors.text;
            this.ctx.font = `${this.cellSize * 0.4}px Arial`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(
                value.toString(),
                cellX + this.cellSize / 2,
                cellY + this.cellSize / 2
            );
        }
    }

    // プレイヤーの描画
    drawPlayer(x, y, playerId) {
        const centerX = this.offsetX + (x + 0.5) * this.cellSize;
        const centerY = this.offsetY + (y + 0.5) * this.cellSize;
        const radius = this.cellSize * 0.3;

        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = playerId === 0 ? this.colors.player0 : this.colors.player1;
        this.ctx.fill();
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
    }

    // スコアの計算
    calculateScore(gameState) {
        if (!gameState || !gameState.board || !gameState.colors) {
            return [0, 0];
        }

        let scores = [0, 0];
        for (let y = 0; y < gameState.board.length; y++) {
            for (let x = 0; x < gameState.board[y].length; x++) {
                const value = gameState.board[y][x];
                const owner = gameState.colors[y][x];
                if (owner >= 0) {
                    scores[owner] += value;
                }
            }
        }
        return scores;
    }

    // ステータス情報の描画
    drawStatus(gameState, scores) {
        this.ctx.font = '16px Arial';
        this.ctx.fillStyle = this.colors.text;

        // スコア表示
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`赤: ${scores[0]}点`, this.textX, this.statusY);
        this.ctx.fillText(`青: ${scores[1]}点`, this.textRightX, this.statusY);

        // ターン表示
        if (gameState && typeof gameState.turn !== 'undefined') {
            const turnText = `ターン: ${gameState.turn === 0 ? '赤' : '青'}`;
            this.ctx.fillText(turnText, this.textX, this.turnY);
        }
    }

    // エラーメッセージの表示
    showError(message) {
        this.ctx.fillStyle = '#ff0000';
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(message, this.canvas.width / 2, this.turnY);
    }

    // ゲーム状態の描画
    drawGameState(gameState) {
        try {
            if (!gameState) {
                throw new Error("Invalid game state");
            }

            // キャンバスのクリア
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            // グリッドの描画
            this.drawGrid();

            // スコアとステータスの描画
            const scores = this.calculateScore(gameState);
            this.drawStatus(gameState, scores);

            // マスの描画
            if (gameState.board && gameState.colors && gameState.rocks) {
                for (let y = 0; y < this.gridSize; y++) {
                    for (let x = 0; x < this.gridSize; x++) {
                        this.drawCell(
                            x, y,
                            gameState.board[y][x],
                            gameState.colors[y][x],
                            gameState.rocks[y][x]
                        );
                    }
                }
            }

            // プレイヤーの描画
            if (gameState.player0 && gameState.player1) {
                this.drawPlayer(gameState.player0.x, gameState.player0.y, 0);
                this.drawPlayer(gameState.player1.x, gameState.player1.y, 1);
            }
        } catch (error) {
            console.error("Error drawing game state:", error);
            this.showError("描画エラー: " + error.message);
        }
    }
}
