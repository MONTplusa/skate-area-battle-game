use anyhow::Result;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::collections::VecDeque;

use super::AI;
use crate::game::{DIRS, GameState, Position};

const INF: i32 = 100_000_000;

#[derive(Debug)]
pub struct StatiolakeAI {
    rng: XorShiftRng,
}

impl StatiolakeAI {
    pub fn new() -> Self {
        StatiolakeAI {
            rng: XorShiftRng::from_os_rng(),
        }
    }

    fn evaluate_initial_position(&mut self, state: &GameState, player: usize) -> f32 {
        let board_size = state.board.len();

        // プレイヤーの位置
        let pos = get_player_position(state, player);

        // 1. 高価値領域へのアクセス評価
        let access_value = evaluate_surrounding_value(state, pos, 5);

        // 2. 中央からの距離評価（中央に近いほど良い）
        let center_x = board_size / 2;
        let center_y = board_size / 2;
        let dist_to_center = calculate_distance(pos, Position::new(center_x, center_y));
        let center_value = (board_size as f32 - dist_to_center) * 5.0;

        // 3. 相手との距離評価（適度な距離が理想）
        let op_pos = get_player_position(state, 1 - player);
        let my_dist = calculate_distance(pos, op_pos);

        // 適度な距離（盤面サイズの1/3程度）が理想
        let optimal_dist = board_size as f32 * 0.3;
        let distance_value = if my_dist < optimal_dist {
            (my_dist / optimal_dist) * 100.0 // 近すぎると低評価
        } else {
            100.0 // 適度な距離以上は高評価
        };

        0.6 * access_value + 0.2 * center_value + 0.2 * distance_value
    }
}

impl Default for StatiolakeAI {
    fn default() -> Self {
        Self::new()
    }
}

// 制御領域の解析結果を格納する構造体
struct ControlledArea {
    me: AreaAnalysis,
    op: AreaAnalysis,
}

// 領域分析結果を格納する構造体
struct AreaAnalysis {
    cells: Vec<Position>,
    total_value: i32,
    unclaimed_value: i32,
    size: usize,
}

fn get_controlled_area(state: &GameState, player: usize) -> ControlledArea {
    let my_pos = get_player_position(state, player);
    let op_pos = get_player_position(state, 1 - player);

    let my_free_dist = compute_dist(state, player, my_pos, None);
    let op_free_dist = compute_dist(state, 1 - player, op_pos, None);

    let my_blocking = compute_blockers(&my_free_dist, &op_free_dist, |a, b| a < b);
    let op_blocking = compute_blockers(&op_free_dist, &my_free_dist, |a, b| a <= b);

    let my_dist = compute_dist(state, player, my_pos, Some(&op_blocking));
    let op_dist = compute_dist(state, 1 - player, op_pos, Some(&my_blocking));

    // 最初の一歩を確定させて評価
    let mut my_best_area = AreaAnalysis {
        cells: Vec::new(),
        total_value: 0,
        unclaimed_value: 0,
        size: 0,
    };

    for dir in DIRS {
        let start = Position::new(
            (my_pos.x as i32 + dir.0) as usize,
            (my_pos.y as i32 + dir.1) as usize,
        );
        let accessible = compute_dist(state, player, start, Some(&op_blocking));
        let area = compute_area(state, player, &my_dist, Some(&accessible));
        if area.total_value >= my_best_area.total_value {
            my_best_area = area;
        }
    }

    let op_area = compute_area(state, 1 - player, &op_dist, None);

    ControlledArea {
        me: my_best_area,
        op: op_area,
    }
}

fn compute_dist(
    state: &GameState,
    player: usize,
    start: Position,
    blocked: Option<&Vec<Vec<bool>>>,
) -> Vec<Vec<i32>> {
    let board_size = state.board.len();
    let mut dist = vec![vec![INF; board_size]; board_size];

    action_bfs(state, blocked, player, start, |pos, d| {
        dist[pos.y][pos.x] = d;
    });

    dist
}

fn compute_blockers<F>(
    free_dist_a: &[Vec<i32>],
    free_dist_b: &[Vec<i32>],
    is_blocked: F,
) -> Vec<Vec<bool>>
where
    F: Fn(i32, i32) -> bool,
{
    let size = free_dist_a.len();
    let mut blockers = vec![vec![false; size]; size];

    for y in 0..size {
        for x in 0..size {
            if is_blocked(free_dist_a[y][x], free_dist_b[y][x]) {
                blockers[y][x] = true;
            }
        }
    }

    blockers
}

fn compute_area(
    state: &GameState,
    player: usize,
    dist: &[Vec<i32>],
    accessible: Option<&[Vec<i32>]>,
) -> AreaAnalysis {
    let mut area = AreaAnalysis {
        cells: Vec::new(),
        total_value: 0,
        unclaimed_value: 0,
        size: 0,
    };

    for y in 0..state.board.len() {
        for x in 0..state.board[y].len() {
            if dist[y][x] == INF || accessible.is_some_and(|acc| acc[y][x] == INF) {
                continue;
            }

            let cell_value = state.board[y][x];
            area.total_value += cell_value;

            if state.colors[y][x] != player as i32 {
                area.unclaimed_value += cell_value;
            }

            area.cells.push(Position::new(x, y));
            area.size += 1;
        }
    }

    area
}

fn action_bfs<F>(
    state: &GameState,
    blocked: Option<&Vec<Vec<bool>>>,
    player: usize,
    start: Position,
    mut visit: F,
) where
    F: FnMut(Position, i32),
{
    #[derive(Clone, Copy)]
    struct BfsNode {
        pos: Position,
        dist: i32,
    }

    let board_size = state.board.len();
    let mut queue = VecDeque::new();
    let mut visited = vec![vec![false; board_size]; board_size];

    queue.push_back(BfsNode {
        pos: start,
        dist: 0,
    });

    while let Some(current) = queue.pop_front() {
        if !is_movable(state, player, current.pos)
            || blocked.is_some_and(|b| b[current.pos.y][current.pos.x])
        {
            continue;
        }

        visit(current.pos, current.dist);

        for dir in DIRS {
            for dist in 1..board_size {
                let next_x = current.pos.x as i32 + dir.0 * dist as i32;
                let next_y = current.pos.y as i32 + dir.1 * dist as i32;

                if next_x < 0
                    || next_x >= board_size as i32
                    || next_y < 0
                    || next_y >= board_size as i32
                {
                    break;
                }

                let next_x = next_x as usize;
                let next_y = next_y as usize;
                let next_pos = Position::new(next_x, next_y);

                if !is_movable(state, player, next_pos)
                    || state.rocks[next_y][next_x]
                    || blocked.is_some_and(|b| b[next_y][next_x])
                {
                    break;
                }

                if !visited[next_y][next_x] {
                    queue.push_back(BfsNode {
                        pos: next_pos,
                        dist: current.dist + 1,
                    });
                    visited[next_y][next_x] = true;
                }
            }
        }
    }
}

fn is_movable(state: &GameState, player: usize, pos: Position) -> bool {
    let board_size = state.board.len();

    if pos.x >= board_size || pos.y >= board_size {
        return false;
    }

    if state.rocks[pos.y][pos.x] {
        return false;
    }

    if player == 1 && pos.x == state.player0.x && pos.y == state.player0.y {
        return false;
    }
    if player == 0 && pos.x == state.player1.x && pos.y == state.player1.y {
        return false;
    }

    true
}

impl AI for StatiolakeAI {
    fn name(&self) -> &str {
        "statiolake"
    }

    fn select_board(&mut self, states: &[GameState]) -> Result<usize> {
        let mut best_idx = 0;
        let mut best_value = f32::NEG_INFINITY;

        for (i, state) in states.iter().enumerate() {
            // 先手と後手の両方で評価し、より高い方を採用
            let first_eval = self.evaluate_initial_position(state, 0);
            let second_eval = self.evaluate_initial_position(state, 1);
            let value = first_eval.max(second_eval);

            if value > best_value {
                best_value = value;
                best_idx = i;
            }
        }

        Ok(best_idx)
    }

    fn select_turn(&mut self, _states: &[GameState]) -> Result<usize> {
        Ok(1) // 常に後手を選択
    }

    fn evaluate(&mut self, state: &GameState, player: usize) -> Result<f32> {
        // プレイヤーが動けない場合は非常に不利
        let my_moves = state.legal_moves(&mut self.rng, player);
        if my_moves.is_empty() {
            return Ok(-100000.0);
        }

        let mut eval = 0.0;

        // 領域の評価
        let controlled_area = get_controlled_area(state, player);
        let my_area = &controlled_area.me;
        let op_area = &controlled_area.op;

        // 領域の価値差を評価値とする
        eval += (my_area.total_value - op_area.total_value) as f32;

        Ok(eval)
    }
}

// プレイヤーの位置を取得
pub fn get_player_position(state: &GameState, player: usize) -> Position {
    if player == 0 {
        state.player0
    } else {
        state.player1
    }
}

// 距離計算
pub fn calculate_distance(pos1: Position, pos2: Position) -> f32 {
    let dx = pos1.x as i32 - pos2.x as i32;
    let dy = pos1.y as i32 - pos2.y as i32;
    ((dx * dx + dy * dy) as f32).sqrt()
}

// 周辺の価値評価
pub fn evaluate_surrounding_value(state: &GameState, pos: Position, radius: i32) -> f32 {
    let board_size = state.board.len();
    let mut total_value = 0.0;
    let mut count = 0;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            // 現在位置は除外
            if dx == 0 && dy == 0 {
                continue;
            }

            let x = pos.x as i32 + dx;
            let y = pos.y as i32 + dy;

            // 盤面内のみ評価
            if x >= 0 && x < board_size as i32 && y >= 0 && y < board_size as i32 {
                // 距離による重み付け（近いほど重要）
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let weight = (1.0 - dist / (radius as f32 + 1.0)).max(0.0);

                // マスの価値
                let value = state.board[y as usize][x as usize] as f32;
                total_value += value * weight;
                count += 1;
            }
        }
    }

    if count > 0 { total_value } else { 0.0 }
}
