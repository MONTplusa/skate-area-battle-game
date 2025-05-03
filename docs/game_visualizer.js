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
    this.colors = {
      player0: "#ffaaaa",
      player1: "#aaaaff",
      rock0: "#552222",
      rock1: "#222255",
      empty: "#ffffff",
      grid: "#cccccc",
      text: "#ffffff",
      textRock: "#ffffff",
      textDark: "#333333",
    };
  }

  // グリッドをクリアして描画
  drawGrid() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = this.colors.grid;
    this.ctx.lineWidth = 1;
    for (let x = 0; x <= this.gridSize; x++) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.offsetX + x * this.cellSize, this.offsetY);
      this.ctx.lineTo(
        this.offsetX + x * this.cellSize,
        this.offsetY + this.gridSize * this.cellSize
      );
      this.ctx.stroke();
    }
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

  drawCell(x, y, value, owner, isRock) {
    const cellX = this.offsetX + x * this.cellSize;
    const cellY = this.offsetY + y * this.cellSize;
    this.ctx.fillStyle = isRock
      ? owner === 0
        ? this.colors.rock0
        : this.colors.rock1
      : owner === 0
      ? this.colors.player0
      : owner === 1
      ? this.colors.player1
      : this.colors.empty;
    this.ctx.fillRect(cellX, cellY, this.cellSize, this.cellSize);

    this.ctx.font = `bold ${this.cellSize * 0.4}px Arial`;
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";

    if (isRock || owner >= 0) {
      this.ctx.strokeStyle = "#000000";
      this.ctx.lineWidth = this.cellSize * 0.05;
      this.ctx.strokeText(
        value.toString(),
        cellX + this.cellSize / 2,
        cellY + this.cellSize / 2
      );
    }
    this.ctx.fillStyle = isRock
      ? this.colors.textRock
      : owner >= 0
      ? this.colors.text
      : this.colors.textDark;
    this.ctx.fillText(
      value.toString(),
      cellX + this.cellSize / 2,
      cellY + this.cellSize / 2
    );
  }

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

  calculateScore(gameState) {
    if (!gameState || !gameState.board || !gameState.colors) return [0, 0];
    let scores = [0, 0];
    gameState.board.forEach((row, y) => {
      row.forEach((value, x) => {
        const owner = gameState.colors[y][x];
        if (owner >= 0) scores[owner] += value;
      });
    });
    return scores;
  }

  drawGameState(gameState) {
    this.drawGrid();
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
    if (gameState.player0 && gameState.player1) {
      this.drawPlayer(gameState.player0.x, gameState.player0.y, 0);
      this.drawPlayer(gameState.player1.x, gameState.player1.y, 1);
    }
  }
}
