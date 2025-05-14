use std::fs;

use anyhow::Result;
use itertools::{Itertools, izip};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_xorshift::XorShiftRng;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::ai::AI;

// 定数
pub const BOARD_SIZE: usize = 20;
const NUM_SAMPLE: usize = 10;

// 上下左右の移動方向
pub const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

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
        .map(|x| x + 1)
        .unwrap_or(0))
}
