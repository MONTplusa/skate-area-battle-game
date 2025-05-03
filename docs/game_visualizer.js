class GameVisualizer {
  constructor(canvasId, gridSize = 20) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.gridSize = gridSize;
    const m = 40;
    this.cellSize = Math.min(
      (this.canvas.width - 2 * m) / gridSize,
      (this.canvas.height - 2 * m) / gridSize
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
    this.ripples = [];
    this.particles = [];
    this.highlights = [];
    this.collisionEffects = []; // 衝突エフェクト用配列
  }

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

  drawCell(x, y, v, o, r) {
    const cx = this.offsetX + x * this.cellSize,
      cy = this.offsetY + y * this.cellSize;
    this.ctx.fillStyle = r
      ? o === 0
        ? this.colors.rock0
        : this.colors.rock1
      : o === 0
      ? this.colors.player0
      : o === 1
      ? this.colors.player1
      : this.colors.empty;
    this.ctx.fillRect(cx, cy, this.cellSize, this.cellSize);
    this.ctx.font = `bold ${this.cellSize * 0.4}px Arial`;
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";
    if (r || o >= 0) {
      this.ctx.strokeStyle = "#000";
      this.ctx.lineWidth = this.cellSize * 0.05;
      this.ctx.strokeText(
        v.toString(),
        cx + this.cellSize / 2,
        cy + this.cellSize / 2
      );
    }
    this.ctx.fillStyle = r
      ? this.colors.textRock
      : o >= 0
      ? this.colors.text
      : this.colors.textDark;
    this.ctx.fillText(
      v.toString(),
      cx + this.cellSize / 2,
      cy + this.cellSize / 2
    );
  }

  drawPlayer(x, y, p) {
    const cx = this.offsetX + (x + 0.5) * this.cellSize,
      cy = this.offsetY + (y + 0.5) * this.cellSize,
      r = this.cellSize * 0.3;
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    this.ctx.fillStyle = p === 0 ? this.colors.player0 : this.colors.player1;
    this.ctx.fill();
    this.ctx.strokeStyle = "#000";
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
  }

  calculateScore(gs) {
    if (!gs || !gs.board || !gs.colors) return [0, 0];
    let s = [0, 0];
    gs.board.forEach((row, y) =>
      row.forEach((v, x) => {
        if (gs.colors[y][x] >= 0) s[gs.colors[y][x]] += v;
      })
    );
    return s;
  }

  // エフェクト描画: Ripple (波紋)
  drawRipples() {
    this.ripples = this.ripples.filter((r) => r.alpha > 0);
    this.ripples.forEach((r) => {
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.arc(r.x, r.y, r.radius, 0, 2 * Math.PI);
      // 色を白から黄色に変更
      this.ctx.strokeStyle = `rgba(255,255,0,${r.alpha})`;
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
      this.ctx.restore();
      r.radius += 1.5;
      r.alpha -= 0.02;
    });
  }

  drawParticles() {
    this.particles = this.particles.filter((p) => p.alpha > 0);
    this.particles.forEach((p) => {
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.radius, 0, 2 * Math.PI);
      this.ctx.fillStyle = `rgba(${parseInt(
        p.color.slice(1, 3),
        16
      )},${parseInt(p.color.slice(3, 5), 16)},${parseInt(
        p.color.slice(5, 7),
        16
      )},${p.alpha})`;
      this.ctx.fill();
      p.x += p.vx;
      p.y += p.vy;
      p.alpha -= 0.02;
      p.radius = Math.max(0, p.radius - 0.02);
    });
  }

  drawHighlights() {
    this.highlights = this.highlights.filter((h) => h.alpha > 0);
    this.highlights.forEach((h) => {
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.arc(h.x, h.y, h.radius, 0, 2 * Math.PI);
      this.ctx.strokeStyle = `rgba(255,255,0,${h.alpha})`;
      this.ctx.lineWidth = 4;
      this.ctx.stroke();
      this.ctx.restore();
      h.radius += 2;
      h.alpha -= 0.03;
    });
  }

  drawCollisionEffects() {
    this.collisionEffects = this.collisionEffects.filter((ef) => {
      const { x, y, radius, alpha } = ef;
      this.ctx.save();
      this.ctx.globalAlpha = alpha;
      this.ctx.lineWidth = 4;
      this.ctx.strokeStyle = "rgba(255,200,0,1)";
      this.ctx.beginPath();
      this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
      this.ctx.stroke();
      this.ctx.restore();
      ef.radius += this.cellSize * 0.1;
      ef.alpha -= 0.02;
      return ef.alpha > 0;
    });
  }

  drawGameState(gs) {
    this.drawGrid();
    if (gs.board && gs.colors && gs.rocks)
      for (let y = 0; y < this.gridSize; y++)
        for (let x = 0; x < this.gridSize; x++)
          this.drawCell(x, y, gs.board[y][x], gs.colors[y][x], gs.rocks[y][x]);
    if (gs.player0 && gs.player1) {
      this.drawPlayer(gs.player0.x, gs.player0.y, 0);
      this.drawPlayer(gs.player1.x, gs.player1.y, 1);
    }
    this.drawRipples();
    this.drawParticles();
    this.drawHighlights();
    this.drawCollisionEffects();
  }
}

window.GameVisualizer = GameVisualizer;
