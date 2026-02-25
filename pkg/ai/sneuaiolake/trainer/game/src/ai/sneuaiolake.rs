use anyhow::{Context, Result, bail};
use itertools::Itertools;
use ndarray::{Array3, Array4, s};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::sync::Arc;

use super::AI;
use crate::game::{BOARD_SIZE, GameState, Move};

const NUM_CHANNELS: usize = 8;
const NUM_ACTIONS: usize = 4 * (BOARD_SIZE - 1);
const SCORE_NORM: f32 = 10000.0;

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

    fn prepare_input(&self, states: &[&GameState], player: usize) -> Result<Array4<f32>> {
        let mut batch_data =
            Array4::<f32>::zeros((states.len(), BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS));

        for (i, state) in states.iter().enumerate() {
            let input_data = create_input_data(state, player);
            input_data.assign_to(batch_data.slice_mut(s![i, .., .., ..]));
        }

        Ok(batch_data)
    }

    fn run_model(
        &mut self,
        states: &[&GameState],
        player: usize,
    ) -> Result<(Option<Vec<Vec<f32>>>, Vec<f32>)> {
        let input = self.prepare_input(states, player)?;
        let input = input.into_dyn().into();
        let input_value = Value::from_array(self.session.allocator(), &input)?;
        let outputs = self
            .session
            .run(vec![input_value])
            .context("Failed to run inference")?;

        let batch_size = states.len();
        let mut policy_logits: Option<Vec<Vec<f32>>> = None;
        let mut values: Option<Vec<f32>> = None;

        for output in outputs {
            let tensor = output.try_extract::<f32>()?;
            let view = tensor.view();
            let shape = view.shape();

            match shape {
                [n, m] if *n == batch_size && *m == NUM_ACTIONS => {
                    let flat = view.iter().copied().collect_vec();
                    let logits = flat
                        .chunks(NUM_ACTIONS)
                        .map(|chunk| chunk.to_vec())
                        .collect_vec();
                    policy_logits = Some(logits);
                }
                [n, m] if *n == batch_size && *m == 1 => {
                    let flat = view.iter().copied().collect_vec();
                    values = Some(flat);
                }
                [n] if *n == batch_size => {
                    let flat = view.iter().copied().collect_vec();
                    values = Some(flat);
                }
                _ => {
                    bail!("unexpected output shape: {:?}", shape);
                }
            }
        }

        let values = values.unwrap_or_else(|| vec![0.0; batch_size]);
        Ok((policy_logits, values))
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

    fn select_move(
        &mut self,
        state: &GameState,
        legal_moves: &[Move],
        player: usize,
    ) -> Result<Option<usize>> {
        let states = vec![state];
        let (policy_logits, _values) = self.run_model(&states, player)?;
        let Some(policy_logits) = policy_logits else {
            return Ok(None);
        };
        let logits = &policy_logits[0];

        let mut legal_action_ids = Vec::with_capacity(legal_moves.len());
        for mov in legal_moves {
            let action_id = action_id_from_move(mov)?;
            legal_action_ids.push(action_id);
        }
        if legal_action_ids.is_empty() {
            return Ok(None);
        }

        let mut max_logit = f32::NEG_INFINITY;
        for &action_id in &legal_action_ids {
            max_logit = max_logit.max(logits[action_id]);
        }

        let mut weights = Vec::with_capacity(legal_action_ids.len());
        let mut total = 0.0f32;
        for &action_id in &legal_action_ids {
            let scaled = (logits[action_id] - max_logit).exp();
            let w = if scaled.is_finite() { scaled } else { 0.0 };
            weights.push(w);
            total += w;
        }

        if total <= 0.0 {
            let (best_idx, _) = legal_action_ids
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| logits[**a].total_cmp(&logits[**b]))
                .unwrap_or((0, &legal_action_ids[0]));
            return Ok(Some(best_idx));
        }

        let mut threshold = self.rng.random_range(0.0..total);
        for (i, &w) in weights.iter().enumerate() {
            threshold -= w;
            if threshold <= 0.0 {
                return Ok(Some(i));
            }
        }
        Ok(Some(weights.len().saturating_sub(1)))
    }

    fn evaluate(&mut self, state: &GameState, player: usize) -> Result<f32> {
        let states = vec![state];
        let (_policy_logits, evals) = self.run_model(&states, player)?;
        Ok(*evals.first().expect("should return one value"))
    }

    fn batch_evaluate(&mut self, states: &[&GameState], player: usize) -> Result<Vec<f32>> {
        let (_policy_logits, evals) = self.run_model(states, player)?;
        Ok(evals)
    }
}

pub fn create_input_data(state: &GameState, player: usize) -> Array3<f32> {
    let mut input_data = Array3::<f32>::zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS));

    let player0 = player;
    let player1 = 1 - player;

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
    let player0_pos = if player0 == 0 { state.player0 } else { state.player1 };
    let player1_pos = if player1 == 1 { state.player1 } else { state.player0 };

    input_data[[player0_pos.y, player0_pos.x, 4]] = 1.0;
    input_data[[player1_pos.y, player1_pos.x, 5]] = 1.0;

    let mut p0_score = 0i32;
    let mut p1_score = 0i32;
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let color = state.colors[y][x];
            if color == player0 as i32 {
                p0_score += state.board[y][x];
            } else if color == player1 as i32 {
                p1_score += state.board[y][x];
            }
        }
    }

    let p0_score_norm = p0_score as f32 / SCORE_NORM;
    let p1_score_norm = p1_score as f32 / SCORE_NORM;
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            input_data[[y, x, 6]] = p0_score_norm;
            input_data[[y, x, 7]] = p1_score_norm;
        }
    }

    input_data
}

fn action_id_from_move(mov: &Move) -> Result<usize> {
    let dx = mov.to_x as i32 - mov.from_x as i32;
    let dy = mov.to_y as i32 - mov.from_y as i32;

    if dx != 0 && dy != 0 {
        bail!("invalid diagonal move");
    }

    let (direction, dist) = if dy > 0 {
        (0usize, dy as usize)
    } else if dx > 0 {
        (1usize, dx as usize)
    } else if dy < 0 {
        (2usize, (-dy) as usize)
    } else if dx < 0 {
        (3usize, (-dx) as usize)
    } else {
        bail!("invalid zero-length move");
    };

    if dist == 0 || dist >= BOARD_SIZE {
        bail!("invalid move distance: {}", dist);
    }

    Ok(direction * (BOARD_SIZE - 1) + (dist - 1))
}
