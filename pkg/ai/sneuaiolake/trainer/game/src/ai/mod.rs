use anyhow::Result;

use crate::game::GameState;

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

pub mod montplusa;
pub mod random;
pub mod sneuaiolake;
pub mod statiolake;
pub mod trivial;
