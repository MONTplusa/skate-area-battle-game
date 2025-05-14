use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use super::AI;
use crate::game::GameState;

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
