const els = {
  runSelect: document.getElementById("runSelect"),
  seriesP0Select: document.getElementById("seriesP0Select"),
  seriesP1Select: document.getElementById("seriesP1Select"),
  reloadBtn: document.getElementById("reloadBtn"),
  statusBadge: document.getElementById("statusBadge"),
  manifestPath: document.getElementById("manifestPath"),
  metricParticipants: document.getElementById("metricParticipants"),
  metricSeries: document.getElementById("metricSeries"),
  metricGames: document.getElementById("metricGames"),
  metricTop: document.getElementById("metricTop"),
  leaderboard: document.getElementById("leaderboard"),
  pairMatrix: document.getElementById("pairMatrix"),
  pairMatrixNote: document.getElementById("pairMatrixNote"),
  gameButtons: document.getElementById("gameButtons"),
  boardCanvas: document.getElementById("boardCanvas"),
  prevBtn: document.getElementById("prevBtn"),
  playBtn: document.getElementById("playBtn"),
  nextBtn: document.getElementById("nextBtn"),
  seekBar: document.getElementById("seekBar"),
  speedSelect: document.getElementById("speedSelect"),
  moveInfo: document.getElementById("moveInfo"),
  scoreInfo: document.getElementById("scoreInfo"),
  playersInfo: document.getElementById("playersInfo"),
};

const state = {
  manifest: null,
  runs: [],
  currentRun: null,
  summary: null,
  series: [],
  seriesIndex: new Map(),
  currentSeriesIndex: -1,
  gameFiles: [],
  currentGameIndex: -1,
  gameData: null,
  frameIndex: 0,
  isPlaying: false,
  playTimer: null,
};

const BOARD_SIZE = 20;
const cellPadding = 1.0;
const MATRIX_MAX_PARTICIPANTS = 8;
const BASELINES = new Set(["random", "trivial", "statiolake", "montplusa"]);

function resizeCanvasForHiDpi() {
  const canvas = els.boardCanvas;
  const rect = canvas.getBoundingClientRect();
  const cssSize = Math.max(1, Math.round(rect.width));
  const dpr = window.devicePixelRatio || 1;
  const pixelSize = Math.max(1, Math.round(cssSize * dpr));
  if (canvas.width !== pixelSize || canvas.height !== pixelSize) {
    canvas.width = pixelSize;
    canvas.height = pixelSize;
  }
}

function setStatus(text) {
  els.statusBadge.textContent = text;
}

function makeUrl(path) {
  const u = new URL(path, window.location.href);
  u.searchParams.set("_ts", String(Date.now()));
  return u.toString();
}

async function fetchJson(path) {
  const res = await fetch(makeUrl(path), { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return res.json();
}

function formatRating(v) {
  return Number(v).toFixed(1);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function pairKey(a, b) {
  return `${a}\u0000${b}`;
}

function simplifyPlayerLabel(raw) {
  if (!raw) {
    return "-";
  }
  const src = String(raw).trim();
  const lower = src.toLowerCase();
  if (BASELINES.has(lower)) {
    return lower;
  }

  const m = src.match(/\(([^)]+)\)\s*$/);
  if (m && m[1]) {
    const inside = m[1].trim();
    const insideLower = inside.toLowerCase();
    if (BASELINES.has(insideLower)) {
      return insideLower;
    }
    const base = inside.split("/").pop() || inside;
    const v = base.match(/^(v\d+)(?:_ac)?\.(?:onnx|keras)$/i);
    if (v && v[1]) {
      return v[1].toLowerCase();
    }
    return base;
  }

  const removed = src.replace(/^sneuaiolake\s*/i, "").trim();
  return removed || src;
}

function scoreFromState(s) {
  const board = s.board;
  const colors = s.colors;
  let p0 = 0;
  let p1 = 0;
  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      const c = colors[y][x];
      if (c === 0) {
        p0 += board[y][x];
      } else if (c === 1) {
        p1 += board[y][x];
      }
    }
  }
  return { p0, p1 };
}

function buildParticipants(summary) {
  return summary.participants
    .map((name) => ({
      name,
      rating: Number(summary.ratings[name] ?? 1500),
      total: summary.totals[name] ?? { W: 0, L: 0, D: 0, G: 0 },
    }))
    .sort((a, b) => b.rating - a.rating);
}

function renderMetrics(participants, summary) {
  const top = participants[0];
  els.metricParticipants.textContent = String(participants.length);
  els.metricSeries.textContent = String(summary.series_results.length);
  els.metricGames.textContent = String(
    summary.series_results.reduce((acc, s) => acc + Number(s.games || 0), 0),
  );
  els.metricTop.textContent = top ? `${top.name} (${formatRating(top.rating)})` : "-";
}

function renderLeaderboard(participants) {
  if (!participants.length) {
    els.leaderboard.innerHTML = "<p class='mono'>No participants</p>";
    return;
  }
  const maxRating = participants[0].rating;
  const minRating = participants[participants.length - 1].rating;
  const ratingRange = Math.max(1e-6, maxRating - minRating);
  const rows = participants
    .map((p, idx) => {
      const w = p.total.W ?? 0;
      const l = p.total.L ?? 0;
      const d = p.total.D ?? 0;
      const ratio = clamp((p.rating - minRating) / ratingRange, 0, 1);
      const barPct = 14 + ratio * 86;
      return `
        <div class="lb-row">
          <div>#${idx + 1}</div>
          <div class="lb-name">
            <span class="lb-main">${p.name}</span>
            <span class="lb-sub">W-L-D ${w}-${l}-${d}</span>
            <div class="lb-bar"><span style="width:${barPct}%;"></span></div>
          </div>
          <div class="lb-rating">${formatRating(p.rating)}</div>
        </div>
      `;
    })
    .join("");
  els.leaderboard.innerHTML = rows;
}

function findPair(summary, a, b) {
  for (const p of summary.pair_results) {
    if ((p.a === a && p.b === b) || (p.a === b && p.b === a)) {
      return p;
    }
  }
  return null;
}

function matrixCellText(pair, rowName, colName) {
  if (!pair) {
    return "-";
  }
  if (pair.a === rowName && pair.b === colName) {
    return `${pair.a_wins}-${pair.b_wins}-${pair.draws}`;
  }
  return `${pair.b_wins}-${pair.a_wins}-${pair.draws}`;
}

function renderPairMatrix(summary, participants) {
  if (!participants.length) {
    els.pairMatrixNote.textContent = "";
    els.pairMatrix.innerHTML = "<p class='mono'>No pair data</p>";
    return;
  }
  const shown = participants.slice(0, MATRIX_MAX_PARTICIPANTS);
  const names = shown.map((p) => p.name);
  const hiddenCount = participants.length - names.length;
  if (hiddenCount > 0) {
    els.pairMatrixNote.textContent = `ELO上位 ${names.length} 名のみ表示（残り ${hiddenCount} 名は省略）`;
  } else {
    els.pairMatrixNote.textContent = `全 ${names.length} 名を表示`;
  }

  let html = "<table class='matrix'><thead><tr><th></th>";
  for (const name of names) {
    html += `<th>${name}</th>`;
  }
  html += "</tr></thead><tbody>";
  for (const rowName of names) {
    html += `<tr><th>${rowName}</th>`;
    for (const colName of names) {
      if (rowName === colName) {
        html += "<td class='empty'>-</td>";
        continue;
      }
      const pair = findPair(summary, rowName, colName);
      const text = matrixCellText(pair, rowName, colName);
      html += `<td data-row="${rowName}" data-col="${colName}" title="Row-Column-Draw">${text}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  els.pairMatrix.innerHTML = html;

  els.pairMatrix.querySelectorAll("td[data-row]").forEach((td) => {
    td.addEventListener("click", () => {
      const a = td.dataset.row;
      const b = td.dataset.col;
      els.seriesP0Select.value = a;
      els.seriesP1Select.value = b;
      onSeriesPickerChanged();
    });
  });
}

function renderSeriesOptions(summary) {
  state.series = summary.series_results || [];
  state.seriesIndex = new Map();
  state.series.forEach((s, i) => {
    state.seriesIndex.set(pairKey(s.p0, s.p1), i);
  });

  if (!state.series.length) {
    els.seriesP0Select.innerHTML = "";
    els.seriesP1Select.innerHTML = "";
    els.gameButtons.innerHTML = "<p class='mono'>No series in this run</p>";
    return;
  }

  const byElo = buildParticipants(summary).map((p) => p.name);
  const names = Array.from(new Set([...byElo, ...summary.participants]));
  const options = names.map((name) => `<option value="${name}">${name}</option>`).join("");
  els.seriesP0Select.innerHTML = options;
  els.seriesP1Select.innerHTML = options;

  const first = state.series[0];
  els.seriesP0Select.value = first.p0;
  els.seriesP1Select.value = first.p1;
  onSeriesPickerChanged();
}

function clearGame() {
  stopPlayback();
  state.gameData = null;
  state.currentGameIndex = -1;
  state.frameIndex = 0;
  els.seekBar.max = "0";
  els.seekBar.value = "0";
  els.moveInfo.textContent = "No game selected";
  els.scoreInfo.textContent = "-";
  els.playersInfo.innerHTML = "P0: - vs P1: -";
  drawBoard(null, null);
}

function renderGameButtons(files) {
  if (!files.length) {
    els.gameButtons.innerHTML = "<p class='mono'>No game files</p>";
    return;
  }
  const html = files
    .map((name, idx) => `<button class="game-btn" data-idx="${idx}" title="${name}">Game ${idx + 1}</button>`)
    .join("");
  els.gameButtons.innerHTML = html;
  els.gameButtons.querySelectorAll("button.game-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const idx = Number(btn.dataset.idx);
      await loadGameByIndex(idx);
    });
  });
}

function setActiveGameButton(gameIndex) {
  els.gameButtons.querySelectorAll("button.game-btn").forEach((btn) => {
    const idx = Number(btn.dataset.idx);
    if (idx === gameIndex) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
}

function getStatesFromGame(gameData) {
  const states = [gameData.initialState];
  for (const m of gameData.moves || []) {
    states.push(m.state);
  }
  return states;
}

function getMoveAtFrame(gameData, frameIndex) {
  if (frameIndex <= 0) {
    return null;
  }
  const i = frameIndex - 1;
  if (!gameData.moves || i >= gameData.moves.length) {
    return null;
  }
  return gameData.moves[i];
}

function setFrame(frameIndex) {
  if (!state.gameData) {
    return;
  }
  const states = getStatesFromGame(state.gameData);
  const maxFrame = states.length - 1;
  state.frameIndex = clamp(frameIndex, 0, maxFrame);
  els.seekBar.max = String(maxFrame);
  els.seekBar.value = String(state.frameIndex);

  const currentState = states[state.frameIndex];
  const currentMove = getMoveAtFrame(state.gameData, state.frameIndex);
  drawBoard(currentState, currentMove);
  updateMoveMeta(currentState, currentMove, maxFrame);
}

function updateMoveMeta(stateObj, move, maxFrame) {
  const { p0, p1 } = scoreFromState(stateObj);
  const p0Name = state.gameData?.initialState?.player0Name || "P0";
  const p1Name = state.gameData?.initialState?.player1Name || "P1";
  const p0Short = simplifyPlayerLabel(p0Name);
  const p1Short = simplifyPlayerLabel(p1Name);
  els.playersInfo.innerHTML = `P0 <span class="p0">${p0Short}</span> vs P1 <span class="p1">${p1Short}</span>`;
  els.scoreInfo.textContent = `score | P0(${p0Short}): ${p0}  P1(${p1Short}): ${p1}  diff: ${p0 - p1}`;

  if (!move) {
    els.moveInfo.textContent = `move 0/${maxFrame} | opening position`;
    return;
  }
  els.moveInfo.textContent = `move ${state.frameIndex}/${maxFrame} | P${move.player} (${move.fromX},${move.fromY}) -> (${move.toX},${move.toY})`;
}

function drawBoard(stateObj, move) {
  resizeCanvasForHiDpi();
  const canvas = els.boardCanvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "#0c2a36";
  ctx.fillRect(0, 0, w, h);

  if (!stateObj) {
    ctx.fillStyle = "rgba(235,245,250,0.65)";
    ctx.font = "20px sans-serif";
    ctx.fillText("No Game", w * 0.43, h * 0.52);
    return;
  }

  const board = stateObj.board;
  const colors = stateObj.colors;
  const rocks = stateObj.rocks;
  const cell = w / BOARD_SIZE;
  const valueFontPx = Math.max(8, Math.floor(cell * 0.34));

  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      const cx = x * cell;
      const cy = y * cell;
      ctx.fillStyle = "rgba(14,43,54,0.85)";
      ctx.fillRect(cx + cellPadding, cy + cellPadding, cell - 2 * cellPadding, cell - 2 * cellPadding);

      const owner = colors[y][x];
      if (owner === 0) {
        ctx.fillStyle = "rgba(34,211,166,0.38)";
        ctx.fillRect(cx + cellPadding, cy + cellPadding, cell - 2 * cellPadding, cell - 2 * cellPadding);
      } else if (owner === 1) {
        ctx.fillStyle = "rgba(249,115,22,0.34)";
        ctx.fillRect(cx + cellPadding, cy + cellPadding, cell - 2 * cellPadding, cell - 2 * cellPadding);
      }

      if (rocks && rocks[y][x]) {
        ctx.fillStyle = "rgba(11,17,23,0.84)";
        ctx.beginPath();
        ctx.arc(cx + cell * 0.5, cy + cell * 0.5, cell * 0.27, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = "rgba(232,244,250,0.8)";
      ctx.font = `${valueFontPx}px "SF Mono", "Menlo", monospace`;
      const v = String(board[y][x]);
      ctx.fillText(v, cx + cell * 0.16, cy + cell * 0.62);
    }
  }

  ctx.strokeStyle = "rgba(190,230,245,0.2)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= BOARD_SIZE; i += 1) {
    const p = i * cell;
    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(w, p);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, h);
    ctx.stroke();
  }

  if (move) {
    const fromX = (move.fromX + 0.5) * cell;
    const fromY = (move.fromY + 0.5) * cell;
    const toX = (move.toX + 0.5) * cell;
    const toY = (move.toY + 0.5) * cell;
    ctx.strokeStyle = move.player === 0 ? "rgba(34,211,166,0.92)" : "rgba(249,115,22,0.92)";
    ctx.lineWidth = Math.max(2, cell * 0.08);
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
  }

  const p0 = stateObj.player0;
  const p1 = stateObj.player1;
  if (p0) {
    ctx.fillStyle = "#4ff6cf";
    ctx.beginPath();
    ctx.arc((p0.x + 0.5) * cell, (p0.y + 0.5) * cell, cell * 0.24, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(7,25,34,0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  if (p1) {
    ctx.fillStyle = "#ffd08a";
    ctx.beginPath();
    ctx.arc((p1.x + 0.5) * cell, (p1.y + 0.5) * cell, cell * 0.24, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(7,25,34,0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function stopPlayback() {
  state.isPlaying = false;
  if (state.playTimer) {
    clearTimeout(state.playTimer);
    state.playTimer = null;
  }
  els.playBtn.textContent = "Play";
}

function playbackStep() {
  if (!state.isPlaying || !state.gameData) {
    return;
  }
  const maxFrame = getStatesFromGame(state.gameData).length - 1;
  if (state.frameIndex >= maxFrame) {
    stopPlayback();
    return;
  }
  setFrame(state.frameIndex + 1);
  const speed = Number(els.speedSelect.value || 1);
  const delay = Math.max(45, 260 / speed);
  state.playTimer = setTimeout(playbackStep, delay);
}

function togglePlayback() {
  if (!state.gameData) {
    return;
  }
  if (state.isPlaying) {
    stopPlayback();
    return;
  }
  state.isPlaying = true;
  els.playBtn.textContent = "Pause";
  playbackStep();
}

async function loadGameByIndex(gameIndex) {
  if (gameIndex < 0 || gameIndex >= state.gameFiles.length) {
    clearGame();
    return;
  }
  stopPlayback();
  state.currentGameIndex = gameIndex;
  setActiveGameButton(gameIndex);
  const gameFile = state.gameFiles[gameIndex];
  setStatus("Loading game");
  try {
    state.gameData = await fetchJson(`./data/${gameFile}`);
    setFrame(0);
    setStatus("Ready");
  } catch (err) {
    console.error(err);
    setStatus("Load failed");
    clearGame();
  }
}

async function onSeriesChanged() {
  if (state.currentSeriesIndex < 0 || state.currentSeriesIndex >= state.series.length) {
    clearGame();
    return;
  }
  const series = state.series[state.currentSeriesIndex];
  state.gameFiles = Array.isArray(series.files) ? series.files : [];
  renderGameButtons(state.gameFiles);
  if (state.gameFiles.length > 0) {
    await loadGameByIndex(0);
  } else {
    clearGame();
  }
}

async function onSeriesPickerChanged() {
  const p0 = els.seriesP0Select.value;
  const p1 = els.seriesP1Select.value;
  if (!p0 || !p1) {
    clearGame();
    return;
  }
  if (p0 === p1) {
    setStatus("Pick different players");
    clearGame();
    return;
  }

  let idx = state.seriesIndex.get(pairKey(p0, p1));
  if (idx === undefined) {
    idx = state.seriesIndex.get(pairKey(p1, p0));
  }
  if (idx === undefined) {
    setStatus("Series not found");
    clearGame();
    return;
  }

  const picked = state.series[idx];
  if (picked.p0 !== p0 || picked.p1 !== p1) {
    els.seriesP0Select.value = picked.p0;
    els.seriesP1Select.value = picked.p1;
  }
  state.currentSeriesIndex = idx;
  await onSeriesChanged();
}

async function loadRun(runId) {
  const run = state.runs.find((r) => r.run_id === runId);
  if (!run) {
    return;
  }
  state.currentRun = run;
  setStatus("Loading run");
  try {
    const summary = await fetchJson(`./data/${run.summary_file}`);
    state.summary = summary;
    const participants = buildParticipants(summary);
    renderMetrics(participants, summary);
    renderLeaderboard(participants);
    renderPairMatrix(summary, participants);
    renderSeriesOptions(summary);
    setStatus("Ready");
  } catch (err) {
    console.error(err);
    setStatus("Run load failed");
  }
}

async function loadManifest(preferredRunId = null) {
  setStatus("Loading manifest");
  els.manifestPath.textContent = "data/manifest.json";
  try {
    const manifest = await fetchJson("./data/manifest.json");
    state.manifest = manifest;
    state.runs = manifest.runs || [];
    if (!state.runs.length) {
      els.runSelect.innerHTML = "";
      els.seriesP0Select.innerHTML = "";
      els.seriesP1Select.innerHTML = "";
      els.leaderboard.innerHTML = "<p class='mono'>No run found. Run elo_league.py first.</p>";
      els.pairMatrix.innerHTML = "";
      clearGame();
      setStatus("No data");
      return;
    }

    els.runSelect.innerHTML = state.runs
      .map((run) => {
        const title = `${run.run_id} | p=${run.participants} s=${run.series} g=${run.games}`;
        return `<option value="${run.run_id}">${title}</option>`;
      })
      .join("");

    const runId = preferredRunId && state.runs.some((r) => r.run_id === preferredRunId)
      ? preferredRunId
      : state.runs[0].run_id;
    els.runSelect.value = runId;
    await loadRun(runId);
  } catch (err) {
    console.error(err);
    setStatus("Manifest missing");
    els.leaderboard.innerHTML = "<p class='mono'>data/manifest.json が見つかりません。</p>";
    els.pairMatrix.innerHTML = "<p class='mono'>先に elo_league.py を実行してください。</p>";
    clearGame();
  }
}

function wireEvents() {
  els.reloadBtn.addEventListener("click", async () => {
    await loadManifest(els.runSelect.value || null);
  });
  els.runSelect.addEventListener("change", async () => {
    await loadRun(els.runSelect.value);
  });
  els.seriesP0Select.addEventListener("change", async () => {
    await onSeriesPickerChanged();
  });
  els.seriesP1Select.addEventListener("change", async () => {
    await onSeriesPickerChanged();
  });
  els.prevBtn.addEventListener("click", () => {
    stopPlayback();
    setFrame(state.frameIndex - 1);
  });
  els.nextBtn.addEventListener("click", () => {
    stopPlayback();
    setFrame(state.frameIndex + 1);
  });
  els.playBtn.addEventListener("click", () => {
    togglePlayback();
  });
  els.seekBar.addEventListener("input", () => {
    stopPlayback();
    setFrame(Number(els.seekBar.value));
  });
  window.addEventListener("keydown", (ev) => {
    if (ev.code === "Space") {
      if (!state.gameData) {
        return;
      }
      ev.preventDefault();
      togglePlayback();
    } else if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      if (!state.gameData) {
        return;
      }
      stopPlayback();
      setFrame(state.frameIndex - 1);
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      if (!state.gameData) {
        return;
      }
      stopPlayback();
      setFrame(state.frameIndex + 1);
    }
  });
  window.addEventListener("resize", () => {
    if (state.gameData) {
      setFrame(state.frameIndex);
    } else {
      drawBoard(null, null);
    }
  });
}

async function boot() {
  wireEvents();
  drawBoard(null, null);
  await loadManifest(null);
}

boot();
