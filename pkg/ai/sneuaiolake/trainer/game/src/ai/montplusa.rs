#![allow(clippy::needless_range_loop, clippy::collapsible_if, dead_code)]
use anyhow::Result;

use crate::game::{BOARD_SIZE, DIRS, GameState};

const INF: i32 = 1_000_000_000;

#[derive(Debug)]
pub struct MontplusaAI;

impl MontplusaAI {
    pub fn new() -> Self {
        MontplusaAI
    }
}

impl Default for MontplusaAI {
    fn default() -> Self {
        Self::new()
    }
}

// BFSキュー
struct BFSQueue {
    queue: Vec<i32>,
    pop_index: usize,
    push_index: usize,
}

impl BFSQueue {
    fn new(max_loop: usize) -> Self {
        BFSQueue {
            queue: vec![0; max_loop],
            pop_index: 0,
            push_index: 0,
        }
    }

    fn init(&mut self) {
        self.pop_index = 0;
        self.push_index = 0;
    }

    fn pop(&mut self) -> i32 {
        let v = self.queue[self.pop_index];
        self.pop_index += 1;
        v
    }

    fn push(&mut self, v: i32) {
        self.queue[self.push_index] = v;
        self.push_index += 1;
    }

    fn len(&self) -> usize {
        self.push_index - self.pop_index
    }
}

fn from_2d_to_1d(p2d: [usize; 2]) -> i32 {
    (p2d[0] * BOARD_SIZE + p2d[1]) as i32
}

fn from_1d_to_2d(p1d: i32) -> [usize; 2] {
    [(p1d as usize) / BOARD_SIZE, (p1d as usize) % BOARD_SIZE]
}

fn judge_territory(
    state: &GameState,
    poss: [[usize; 2]; 2],
    first_dir: usize,
    second_dir: usize,
    dir3: usize,
    turn: usize,
) -> Vec<Vec<i32>> {
    let mut results = vec![vec![-1; BOARD_SIZE]; BOARD_SIZE];
    let mut queue = BFSQueue::new(BOARD_SIZE * BOARD_SIZE);
    queue.init();

    let p1 = from_2d_to_1d(poss[turn]);
    let p2 = from_2d_to_1d(poss[1 - turn]);

    queue.push(p1);
    results[poss[turn][0]][poss[turn][1]] = turn as i32;
    queue.push(p2);
    results[poss[1 - turn][0]][poss[1 - turn][1]] = (1 - turn) as i32;

    let mut bfs_count = 0;
    let mut end3 = 2;

    while queue.len() > 0 {
        let p1d = queue.pop();
        let p2d = from_1d_to_2d(p1d);
        let pl = results[p2d[0]][p2d[1]];

        for d in 0..4 {
            if bfs_count == 0 {
                if d != first_dir {
                    continue;
                }
            } else if bfs_count == 1 {
                if d != second_dir {
                    continue;
                }
            } else if bfs_count < end3 {
                if d != dir3 {
                    continue;
                }
            }

            let mut y = p2d[0];
            let mut x = p2d[1];

            loop {
                y = (y as i32 + DIRS[d].1) as usize;
                x = (x as i32 + DIRS[d].0) as usize;

                if x >= BOARD_SIZE || y >= BOARD_SIZE {
                    break;
                }

                if state.rocks[y][x] {
                    break;
                }

                if results[y][x] == 1 - pl {
                    break;
                }

                let np1d = from_2d_to_1d([y, x]);
                if np1d == p1 || np1d == p2 {
                    break;
                }

                if results[y][x] == -1 {
                    if bfs_count == 0 {
                        end3 += 1;
                    }
                    queue.push(np1d);
                    results[y][x] = pl;
                }
            }
        }
        bfs_count += 1;
    }

    for i in 0..BOARD_SIZE {
        for j in 0..BOARD_SIZE {
            if results[i][j] == -1 {
                results[i][j] = state.colors[i][j];
            }
        }
    }

    results
}

impl super::AI for MontplusaAI {
    fn name(&self) -> &str {
        "montplusa"
    }

    fn select_board(&mut self, _states: &[GameState]) -> Result<usize> {
        Ok(0)
    }

    fn select_turn(&mut self, _states: &[GameState]) -> Result<usize> {
        Ok(0)
    }

    fn evaluate(&mut self, st: &GameState, player: usize) -> Result<f32> {
        let poss = [[st.player0.y, st.player0.x], [st.player1.y, st.player1.x]];
        let mut min_diff = 1_000_000_000.0;

        for first_dir in 0..4 {
            let mut max_diff = -1_000_000_000.0;
            for second_dir in 0..4 {
                let mut min_diff2 = 1_000_000_000.0;
                for dir3 in 0..4 {
                    let territory = judge_territory(st, poss, first_dir, second_dir, dir3, st.turn);
                    let mut scores = [0.0, 0.0];

                    for i in 0..BOARD_SIZE {
                        for j in 0..BOARD_SIZE {
                            if territory[i][j] != -1 {
                                scores[territory[i][j] as usize] += st.board[i][j] as f32;
                            }
                        }
                    }

                    for i in 0..BOARD_SIZE {
                        for j in 0..BOARD_SIZE {
                            if st.colors[i][j] != -1 {
                                scores[st.colors[i][j] as usize] += st.board[i][j] as f32 * 0.1;
                            }
                        }
                    }

                    let diff = scores[player] - scores[1 - player];
                    if diff < min_diff2 {
                        min_diff2 = diff;
                    }
                }
                if min_diff2 > max_diff {
                    max_diff = min_diff2;
                }
            }
            if max_diff < min_diff {
                min_diff = max_diff;
            }
        }

        let mut value = min_diff;
        let mut c = 0;

        let pos = if player == 0 { st.player0 } else { st.player1 };
        let op_pos = if player == 0 { st.player1 } else { st.player0 };

        for &(dx, dy) in DIRS.iter() {
            let x = pos.x as i32 + dx;
            let y = pos.y as i32 + dy;

            if x < 0 || x >= BOARD_SIZE as i32 || y < 0 || y >= BOARD_SIZE as i32 {
                continue;
            }

            let x = x as usize;
            let y = y as usize;

            if x == op_pos.x && y == op_pos.y {
                continue;
            }

            if st.rocks[y][x] {
                continue;
            }

            c += 1;
        }

        if c == 1 {
            value += 0.01;
        }

        Ok(value)
    }
}
