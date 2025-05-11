use anyhow::{Context, Result};
use clap::Parser;
use itertools::{Itertools, izip};
use ndarray::{Array3, Array4, s};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, cmp::Ordering, sync::OnceLock};
use std::{fs, sync::Arc};

// 定数
const BOARD_SIZE: usize = 20;
const NUM_CHANNELS: usize = 6;
const NUM_SAMPLE: usize = 10;

// 上下左右の移動方向
const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

impl Position {
    pub fn new(x: usize, y: usize) -> Self {
        Position { x, y }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Move {
    pub player: usize,
    #[serde(rename = "fromX")]
    pub from_x: usize,
    #[serde(rename = "fromY")]
    pub from_y: usize,
    #[serde(rename = "toX")]
    pub to_x: usize,
    #[serde(rename = "toY")]
    pub to_y: usize,
    pub state: GameState,
}

impl Move {
    pub fn new(
        player: usize,
        from_x: usize,
        from_y: usize,
        to_x: usize,
        to_y: usize,
        state: GameState,
    ) -> Self {
        Move {
            player,
            from_x,
            from_y,
            to_x,
            to_y,
            state,
        }
    }

    pub fn from_pos(&self) -> Position {
        Position::new(self.from_x, self.from_y)
    }

    pub fn to_pos(&self) -> Position {
        Position::new(self.to_x, self.to_y)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Vec<Vec<i32>>,
    pub colors: Vec<Vec<i32>>,
    pub rocks: Vec<Vec<bool>>,
    pub player0: Position,
    pub player1: Position,
    pub turn: usize,
    #[serde(rename = "player0Name")]
    pub player0_name: String,
    #[serde(rename = "player1Name")]
    pub player1_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleResult {
    #[serde(rename = "initialState")]
    pub initial_state: GameState,
    pub moves: Vec<Move>,
    #[serde(rename = "finalState")]
    pub final_state: GameState,
}

impl GameState {
    pub fn new(board_size: usize) -> Self {
        GameState {
            board: vec![vec![0; board_size]; board_size],
            colors: vec![vec![-1; board_size]; board_size],
            rocks: vec![vec![false; board_size]; board_size],
            player0: Position::new(0, 0),
            player1: Position::new(0, 0),
            turn: 0,
            player0_name: "Player0".to_string(),
            player1_name: "Player1".to_string(),
        }
    }

    pub fn legal_moves<R: Rng>(&self, rng: &mut R, player: usize) -> Vec<Move> {
        let mut moves = Vec::new();
        let pos = if player == 0 {
            self.player0
        } else {
            self.player1
        };
        let board_size = self.board.len();

        for (dir_x, dir_y) in DIRS {
            let mut max_dist = 0;
            for dist in 1..board_size {
                let x = pos.x as i32 + dir_x * dist as i32;
                let y = pos.y as i32 + dir_y * dist as i32;

                if x < 0 || x >= board_size as i32 || y < 0 || y >= board_size as i32 {
                    break;
                }

                let ux = x as usize;
                let uy = y as usize;

                if self.rocks[uy][ux] {
                    break;
                }

                // 相手プレイヤーの位置チェック
                let op_pos = self.position_of(1 - player);
                if ux == op_pos.x && uy == op_pos.y {
                    break;
                }

                max_dist = dist;
            }

            // 各距離について合法手を生成
            for dist in 1..=max_dist {
                let new_x = pos.x as i32 + dir_x * dist as i32;
                let new_y = pos.y as i32 + dir_y * dist as i32;

                let mut new_state = self.clone();
                if player == 0 {
                    new_state.player0.x = new_x as usize;
                    new_state.player0.y = new_y as usize;
                } else {
                    new_state.player1.x = new_x as usize;
                    new_state.player1.y = new_y as usize;
                }

                // 移動元を岩石に変更
                new_state.rocks[pos.y][pos.x] = true;

                // 移動経路を塗る
                for d in 0..=dist {
                    let path_x = pos.x as i32 + dir_x * d as i32;
                    let path_y = pos.y as i32 + dir_y * d as i32;
                    new_state.colors[path_y as usize][path_x as usize] = player as i32;
                }

                new_state.turn = 1 - player;
                moves.push(Move::new(
                    player,
                    pos.x,
                    pos.y,
                    new_x as usize,
                    new_y as usize,
                    new_state,
                ));
            }
        }

        // 合法手をシャッフル
        moves.shuffle(rng);
        moves
    }

    fn position_of(&self, player: usize) -> Position {
        if player == 0 {
            self.player0
        } else {
            self.player1
        }
    }
}

pub fn generate_initial_states<R: Rng>(
    rng: &mut R,
    num_samples: usize,
    board_size: usize,
) -> Vec<GameState> {
    let mut states = Vec::new();

    for _ in 0..num_samples {
        let mut state = GameState::new(board_size);

        // 盤面の初期化
        for y in 0..board_size {
            for x in 0..board_size {
                state.board[y][x] = rng.random_range(0..=100);
                state.colors[y][x] = -1;
            }
        }

        // プレイヤーの初期位置をランダムに設定
        loop {
            state.player0.x = rng.random_range(0..board_size);
            state.player0.y = rng.random_range(0..board_size);
            state.player1.x = board_size - 1 - state.player0.x;
            state.player1.y = board_size - 1 - state.player0.y;

            // プレイヤーが十分離れているか確認
            let dx = state.player1.x as i32 - state.player0.x as i32;
            let dy = state.player1.y as i32 - state.player0.y as i32;
            if (dx * dx + dy * dy) as usize >= board_size {
                break;
            }
        }

        state.turn = 0;
        states.push(state);
    }

    states
}

pub fn create_input_data(state: &GameState) -> Array3<f32> {
    let mut input_data = Array3::<f32>::zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS));

    let next_player = state.turn;
    let player0 = 1 - next_player;
    let player1 = next_player;

    // ボードの最大値を取得して正規化
    let mut board_max = 0;
    for row in &state.board {
        for &val in row {
            if val > board_max {
                board_max = val;
            }
        }
    }

    // チャンネル0: ボードの数値を0-1に正規化
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            if board_max > 0 {
                input_data[[y, x, 0]] = state.board[y][x] as f32 / board_max as f32;
            }
        }
    }

    // チャンネル1,2: プレイヤーの色
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let color = state.colors[y][x];
            if color == player0 as i32 {
                input_data[[y, x, 1]] = 1.0;
            } else if color == player1 as i32 {
                input_data[[y, x, 2]] = 1.0;
            }
        }
    }

    // チャンネル3: 岩の位置
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            if state.rocks[y][x] {
                input_data[[y, x, 3]] = 1.0;
            }
        }
    }

    // チャンネル4,5: プレイヤーの位置
    let player0_pos = if player0 == 0 {
        state.player0
    } else {
        state.player1
    };
    let player1_pos = if player1 == 1 {
        state.player1
    } else {
        state.player0
    };

    input_data[[player0_pos.y, player0_pos.x, 4]] = 1.0;
    input_data[[player1_pos.y, player1_pos.x, 5]] = 1.0;

    input_data
}

pub trait AI: std::fmt::Debug {
    fn name(&self) -> &str;
    fn select_board(&mut self, states: &[GameState]) -> Result<usize>;
    fn select_turn(&mut self, states: &[GameState]) -> Result<usize>;
    fn evaluate(&mut self, state: &GameState, player: usize) -> Result<f32>;
    fn batch_evaluate(&mut self, states: &[&GameState], player: usize) -> Result<Vec<f32>> {
        states
            .iter()
            .map(|state| self.evaluate(state, player))
            .collect()
    }
}

#[derive(Debug)]
pub struct SneuaiolakeAI {
    session: ort::Session,
    name: String,
    rng: XorShiftRng,
}

impl SneuaiolakeAI {
    pub fn new(model_path: &str) -> Result<Self> {
        // ONNX Runtimeの初期化
        let environment = Arc::new(Environment::builder().with_name("GameAI").build()?);

        // 自動的に最適化されたプロバイダーを選択
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(model_path)
            .context(format!("Failed to load ONNX model: {}", model_path))?;

        let name = format!("sneuaiolake ({})", model_path);
        let rng = XorShiftRng::from_os_rng();

        Ok(SneuaiolakeAI { session, name, rng })
    }

    fn prepare_input(&self, states: &[&GameState]) -> Result<Array4<f32>> {
        let mut batch_data =
            Array4::<f32>::zeros((states.len(), BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS));

        for (i, state) in states.iter().enumerate() {
            let input_data = create_input_data(state);
            input_data.assign_to(batch_data.slice_mut(s![i, .., .., ..]));
            // for c in 0..NUM_CHANNELS {
            //     for y in 0..BOARD_SIZE {
            //         for x in 0..BOARD_SIZE {
            //             // // チャンネル次元を前に持ってくる (HWC -> CHW)
            //             // batch_data[[i, c, y, x]] = input_data[[y, x, c]];
            //             batch_data[[i, y, x, c]] = input_data[[y, x, c]];
            //         }
            //     }
            // }
        }

        Ok(batch_data)
    }
}

impl AI for SneuaiolakeAI {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_board(&mut self, states: &[GameState]) -> Result<usize> {
        Ok(self.rng.random_range(0..states.len()))
    }

    fn select_turn(&mut self, _states: &[GameState]) -> Result<usize> {
        Ok(self.rng.random_range(0..2))
    }

    fn evaluate(&mut self, state: &GameState, _player: usize) -> Result<f32> {
        let states = vec![state];
        let evals = self.batch_evaluate(&states, _player)?;

        Ok(*evals.first().expect("should return one value"))
    }

    fn batch_evaluate(&mut self, states: &[&GameState], _player: usize) -> Result<Vec<f32>> {
        let input = self.prepare_input(states)?;
        let input = input.into_dyn().into();
        let input_value = Value::from_array(self.session.allocator(), &input)?;
        let outputs = self
            .session
            .run(vec![input_value])
            .context("Failed to run inference")?;
        let output = outputs[0].try_extract::<f32>()?;
        let evals = output.view().iter().copied().collect_vec();

        Ok(evals)
    }
}

#[derive(Debug)]
pub struct RandomAI {
    rng: XorShiftRng,
}

impl RandomAI {
    pub fn new() -> Self {
        RandomAI {
            rng: XorShiftRng::from_os_rng(),
        }
    }
}

impl Default for RandomAI {
    fn default() -> Self {
        Self::new()
    }
}

impl AI for RandomAI {
    fn name(&self) -> &str {
        "random"
    }

    fn select_board(&mut self, states: &[GameState]) -> Result<usize> {
        Ok(self.rng.random_range(0..states.len()))
    }

    fn select_turn(&mut self, _states: &[GameState]) -> Result<usize> {
        Ok(self.rng.random_range(0..2))
    }

    fn evaluate(&mut self, _state: &GameState, _player: usize) -> Result<f32> {
        Ok(self.rng.random_range(-1.0..1.0))
    }
}

#[derive(Debug)]
pub struct TrivialAI {
    rng: XorShiftRng,
}

impl TrivialAI {
    pub fn new() -> Self {
        TrivialAI {
            rng: XorShiftRng::from_os_rng(),
        }
    }
}

impl Default for TrivialAI {
    fn default() -> Self {
        Self::new()
    }
}

impl AI for TrivialAI {
    fn name(&self) -> &str {
        "trivial"
    }

    fn select_board(&mut self, states: &[GameState]) -> Result<usize> {
        Ok(self.rng.random_range(0..states.len()))
    }

    fn select_turn(&mut self, _states: &[GameState]) -> Result<usize> {
        Ok(self.rng.random_range(0..2))
    }

    fn evaluate(&mut self, state: &GameState, player: usize) -> Result<f32> {
        let mut score = 0;
        let opponent = 1 - player;

        for y in 0..state.board.len() {
            for x in 0..state.board[y].len() {
                let value = state.board[y][x];
                let color = state.colors[y][x];
                if color == player as i32 {
                    score += value;
                } else if color == opponent as i32 {
                    score -= value;
                }
            }
        }

        Ok(score as f32)
    }
}

pub struct GameRunner<'a> {
    rng: XorShiftRng,
    agents: [&'a mut dyn AI; 2],
    show_log: bool,
}

impl<'a> GameRunner<'a> {
    pub fn new(agent0: &'a mut dyn AI, agent1: &'a mut dyn AI, show_log: bool) -> Self {
        let rng = XorShiftRng::from_os_rng();
        GameRunner {
            rng,
            agents: [agent0, agent1],
            show_log,
        }
    }

    pub fn run(&mut self) -> Result<BattleResult> {
        // 1) 初期盤面サンプル生成 & 盤面選択
        let states = generate_initial_states(&mut self.rng, NUM_SAMPLE, BOARD_SIZE);
        let chooser = self.rng.random_range(0..2);
        let other = 1 - chooser;
        let idx = self.agents[other].select_board(&states)?;
        let mut state = states[idx].clone();

        // 2) 先後選択
        let first = self.agents[chooser].select_turn(&states)?;
        state.turn = (first + chooser) % 2;
        if first != chooser {
            std::mem::swap(&mut state.player0, &mut state.player1);
        }

        // 3) AIの名前を設定
        state.player0_name = self.agents[0].name().to_string();
        state.player1_name = self.agents[1].name().to_string();

        let initial_state = state.clone();

        // 4) 結果オブジェクトの初期化
        let mut moves = vec![];

        // 5) ゲームループ
        let mut number = 0;
        let mut skips = 0;
        while skips < 2 {
            let player = state.turn;
            let mut legal_moves = state.legal_moves(&mut self.rng, player);
            legal_moves.shuffle(&mut self.rng);

            if legal_moves.is_empty() {
                if self.show_log {
                    println!("プレイヤー{}の合法手がありません。スキップします。", player);
                }
                skips += 1;
                state.turn = 1 - player;
                continue;
            }

            skips = 0;
            let mut best_eval = f32::NEG_INFINITY;
            let mut best_move: Option<Move> = None;

            let move_states = legal_moves.iter().map(|m| &m.state).collect_vec();
            let evals = self.agents[player].batch_evaluate(&move_states, player)?;
            for (mov, eval) in izip!(legal_moves, evals) {
                if self.show_log {
                    println!(
                        "| プレイヤー{}の手: ({}, {}) -> ({}, {}), 評価値: {}",
                        player, mov.from_y, mov.from_x, mov.to_y, mov.to_x, eval
                    );
                }
                if eval > best_eval {
                    best_eval = eval;
                    best_move = Some(mov);
                }
            }

            // 有効な手が見つかった場合のみ進める
            if let Some(mov) = best_move {
                if self.show_log {
                    println!(
                        "-> #{:<3} プレイヤー{}の最良手 ({}): ({}, {}) -> ({}, {})",
                        number, player, best_eval, mov.from_y, mov.from_x, mov.to_y, mov.to_x
                    );
                }
                state = mov.state.clone();
                moves.push(mov);
                number += 1;
            } else {
                if self.show_log {
                    println!(
                        "プレイヤー{}の有効な手が見つかりません。スキップします。",
                        player
                    );
                }
                state.turn = 1 - player;
                skips += 1;
            }
        }

        let final_state = state;

        Ok(BattleResult {
            initial_state,
            moves,
            final_state,
        })
    }
}

fn find_next_sequence_number(directory: &str, prefix: &str) -> Result<u32> {
    let pattern = format!(r"^{}_(\d{{5}})\.json$", regex::escape(prefix));
    let re = Regex::new(&pattern)?;

    Ok(fs::read_dir(directory)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            if let Some(caps) = re.captures(&filename_str) {
                caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok())
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0))
}

fn create_ai(model_path: Option<&str>) -> Result<Box<dyn AI + Send>> {
    if let Some(path) = model_path {
        Ok(Box::new(SneuaiolakeAI::new(path)?))
    } else {
        Ok(Box::new(RandomAI::new()))
    }
}

#[derive(Parser)]
#[command(name = "game")]
#[command(about = "自己対戦によるトレーニングデータ生成")]
struct Options {
    #[arg(long, help = "P0として使用するONNXモデルのパス（.onnx形式）")]
    p0_model: Option<String>,

    #[arg(long, help = "P1として使用するONNXモデルのパス（.onnx形式）")]
    p1_model: Option<String>,

    #[arg(long, default_value = "10", help = "自己対戦の回数")]
    games: usize,

    #[arg(long, help = "対戦結果の保存先ディレクトリ")]
    output_dir: String,

    #[arg(long, help = "出力ファイル名のプレフィックス")]
    prefix: String,

    #[arg(long, help = "並列処理のスレッド数")]
    jobs: Option<usize>,

    #[arg(long, help = "ログを表示する")]
    show_log: bool,
}

fn play(opts: &Options) -> Result<()> {
    // 出力ディレクトリの作成
    fs::create_dir_all(&opts.output_dir)?;

    // 既存ファイルの最大連番を取得
    let start_seq = find_next_sequence_number(&opts.output_dir, &opts.prefix)?;
    println!("連番 {:05} から開始します", start_seq);

    if let Some(jobs) = opts.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()?;
    }

    // AI は毎回モデルをロードすると重いので、スレッドローカルストレージにキャッシュする
    thread_local! {
        static P0_AI: OnceLock<RefCell<Box<dyn AI>>> = OnceLock::new();
        static P1_AI: OnceLock<RefCell<Box<dyn AI>>> = OnceLock::new();
    }

    // 対戦の実行（並列化）
    let battle_results = (0..opts.games)
        .into_par_iter()
        .map(|i| {
            P0_AI.with(|p0_ai| {
                P1_AI.with(|p1_ai| {
                    let p0_ai = p0_ai.get_or_init(|| {
                        let ai =
                            create_ai(opts.p0_model.as_deref()).expect("failed to create AI 0");
                        RefCell::new(ai)
                    });
                    let p1_ai = p1_ai.get_or_init(|| {
                        let ai =
                            create_ai(opts.p1_model.as_deref()).expect("failed to create AI 1");
                        RefCell::new(ai)
                    });

                    let mut p0_ai = p0_ai.borrow_mut();
                    let mut p1_ai = p1_ai.borrow_mut();

                    // 自己対戦
                    let mut runner = GameRunner::new(&mut **p0_ai, &mut **p1_ai, opts.show_log);
                    let result = runner.run()?;

                    // 勝者を判定
                    let mut scores = [0, 0];
                    let state = &result.final_state;
                    let board = &state.board;
                    let colors = &state.colors;

                    for (y, row) in colors.iter().enumerate() {
                        for (x, color) in row.iter().enumerate() {
                            if *color != -1 {
                                let value = board[y][x];
                                scores[*color as usize] += value;
                            }
                        }
                    }

                    // スコアを出力
                    println!(
                        "対戦 #{}: スコア - {}: {}, {}: {}",
                        i, state.player0_name, scores[0], state.player1_name, scores[1]
                    );

                    // 勝者を決定
                    let winner = match scores[0].cmp(&scores[1]) {
                        Ordering::Greater => Some(0),
                        Ordering::Less => Some(1),
                        Ordering::Equal => None,
                    };

                    // 結果の保存
                    let seq_num = start_seq + i as u32;
                    let filename =
                        format!("{}/{}_{:05}.json", opts.output_dir, opts.prefix, seq_num);
                    fs::write(filename, serde_json::to_string(&result)?)?;

                    Ok(winner)
                })
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // 結果の集計
    let mut wins = [0, 0];
    for winner in battle_results.into_iter().flatten() {
        wins[winner] += 1;
    }

    let draw = opts.games as i32 - wins[0] - wins[1];
    println!(
        "勝利数: P0: {}, P1: {}, 引き分け: {}",
        wins[0], wins[1], draw
    );

    Ok(())
}

fn main() -> Result<()> {
    let opts = Options::parse();

    play(&opts)
}
