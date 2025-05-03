class GameVisualizer {
  constructor(canvasId, gridSize = 20) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.gridSize = gridSize;
    const margin = 40;
    this.cellSize = Math.min(
      (this.canvas.width - margin * 2) / gridSize,
      (this.canvas.height - margin * 2) / gridSize
    );
    this.offsetX = (this.canvas.width - this.cellSize * gridSize) / 2;
    this.offsetY = (this.canvas.height - this.cellSize * gridSize) / 2;

    // プレイヤーの色
    this.colors = {
      player0: "#ffaaaa", // 赤プレイヤー
      player1: "#aaaaff", // 青プレイヤー
      rock0: "#552222", // 赤プレイヤーの岩（赤みがかったグレー）
      rock1: "#222255", // 青プレイヤーの岩（青みがかったグレー）
      empty: "#ffffff", // 未着色マス
      grid: "#cccccc", // グリッド線
      text: "#ffffff", // 数字（白）
      textRock: "#ffffff", // 岩の上の数字（白）
      textDark: "#333333", // 未着色マスの数字（濃いグレー）
    };
  }

  // グリッドの描画
  drawGrid() {
    this.ctx.strokeStyle = this.colors.grid;
    this.ctx.lineWidth = 1;

    // 縦線
    for (let x = 0; x <= this.gridSize; x++) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.offsetX + x * this.cellSize, this.offsetY);
      this.ctx.lineTo(
        this.offsetX + x * this.cellSize,
        this.offsetY + this.gridSize * this.cellSize
      );
      this.ctx.stroke();
    }

    // 横線
    for (let y = 0; y <= this.gridSize; y++) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.offsetX, this.offsetY + y * this.cellSize);
      this.ctx.lineTo(
        this.offsetX + this.gridSize * this.cellSize,
        this.offsetY + y * this.cellSize
      );
      this.ctx.stroke();
    }
  }

  // マスの描画
  drawCell(x, y, value, owner, isRock) {
    const cellX = this.offsetX + x * this.cellSize;
    const cellY = this.offsetY + y * this.cellSize;

    // マスの塗りつぶし
    if (isRock) {
      // 岩の色を所有者によって変える
      this.ctx.fillStyle = owner === 0 ? this.colors.rock0 : this.colors.rock1;
    } else {
      this.ctx.fillStyle =
        owner === 0
          ? this.colors.player0
          : owner === 1
          ? this.colors.player1
          : this.colors.empty;
    }
    this.ctx.fillRect(cellX, cellY, this.cellSize, this.cellSize);

    // 数値の描画（岩の上にも表示）
    this.ctx.fillStyle = isRock
      ? this.colors.textRock
      : owner >= 0
      ? this.colors.text
      : this.colors.textDark;

    // テキストの縁取り（視認性向上）
    if (isRock || owner >= 0) {
      this.ctx.strokeStyle = "#000000";
      this.ctx.lineWidth = this.cellSize * 0.05;
      this.ctx.strokeText(
        value.toString(),
        cellX + this.cellSize / 2,
        cellY + this.cellSize / 2
      );
    }
    this.ctx.font = `bold ${this.cellSize * 0.4}px Arial`;
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";
    this.ctx.fillText(
      value.toString(),
      cellX + this.cellSize / 2,
      cellY + this.cellSize / 2
    );
  }

  // プレイヤーの描画
  drawPlayer(x, y, playerId) {
    const centerX = this.offsetX + (x + 0.5) * this.cellSize;
    const centerY = this.offsetY + (y + 0.5) * this.cellSize;
    const radius = this.cellSize * 0.3;

    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    this.ctx.fillStyle =
      playerId === 0 ? this.colors.player0 : this.colors.player1;
    this.ctx.fill();
    this.ctx.strokeStyle = "#000000";
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

  // 外部のUIを更新
  updateUI(gameState, scores) {
    // 名前とスコア表示の更新
    document.getElementById("name-p0").textContent = `赤: ${gameState.player0Name || "-"}`;
    document.getElementById("name-p1").textContent = `青: ${gameState.player1Name || "-"}`;
    document.getElementById("score-p0").textContent = `${scores[0]}点`;
    document.getElementById("score-p1").textContent = `${scores[1]}点`;

    // ターン表示の更新
    if (gameState && typeof gameState.turn !== "undefined") {
      document.getElementById("current-turn").textContent = `ターン: ${
        gameState.turn === 0 ? "赤" : "青"
      }`;
    }
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

      // UIの更新
      const scores = this.calculateScore(gameState);
      this.updateUI(gameState, scores);

      // マスの描画
      if (gameState.board && gameState.colors && gameState.rocks) {
        for (let y = 0; y < this.gridSize; y++) {
          for (let x = 0; x < this.gridSize; x++) {
            this.drawCell(
              x,
              y,
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
