use anyhow::{Context, Result};
use itertools::Itertools;
use ndarray::{Array3, Array4, s};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::sync::Arc;

use super::AI;
use crate::game::{BOARD_SIZE, GameState};

const NUM_CHANNELS: usize = 6;

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
