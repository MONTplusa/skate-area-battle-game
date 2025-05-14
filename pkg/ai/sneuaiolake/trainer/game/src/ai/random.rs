use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use super::AI;
use crate::game::GameState;

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
