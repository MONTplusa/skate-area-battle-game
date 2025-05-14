use anyhow::{Context, Result};
use itertools::Itertools;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::collections::HashMap;
use std::fs;
use std::{
    cmp::Ordering as CmpOrdering,
    sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
};

use skate_area_battle_game::game::BattleResult;

fn calculate_elo_change(rating_a: f64, rating_b: f64, result: f64) -> f64 {
    let k = 32.0; // レーティング変動係数
    let expected = 1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0));
    k * (result - expected)
}

fn main() -> Result<()> {
    let mut ratings = HashMap::new();
    let default_rating = 1500.0;

    // コマンドライン引数からJSONファイルを読み込む
    // コマンドライン引数からディレクトリを取得
    let dir = std::env::args()
        .nth(1)
        .context("Please specify a directory")?;

    // ディレクトリ内の全JSONファイルを読み込む
    // 事前にJSONファイルの総数をカウント
    let total_files = fs::read_dir(&dir)?
        .filter(|entry| {
            entry
                .as_ref()
                .map(|e| e.path().extension().is_some_and(|ext| ext == "json"))
                .unwrap_or(false)
        })
        .count();
    println!("total: {}", total_files);

    // 処理済みファイル数を原子的にカウント
    let processed = AtomicUsize::new(0);
    // 結果を収集するためのチャネル

    let final_states = fs::read_dir(&dir)?
        .par_bridge()
        .map(|entry| {
            let entry = entry?;
            let path = entry.path();

            // JSONファイルのみを処理
            let Some(ext) = path.extension() else {
                return Ok(None);
            };

            if ext != "json" {
                return Ok(None);
            }

            let content = fs::read_to_string(&path)
                .context(format!("Failed to read file: {}", path.display()))?;
            let result: BattleResult = serde_json::from_str(&content).context(format!(
                "Failed to parse JSON from file: {}",
                path.display()
            ))?;

            let processed = processed.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            print!(
                "\rProcessing files... {}/{} ({:.1}%)\r",
                processed,
                total_files,
                (processed as f64 / total_files as f64) * 100.0
            );

            Ok(Some(result.final_state))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect_vec();
    println!(); // 改行を入れて進捗表示を確定

    // 対戦結果から勝敗を判定してELOレーティングを更新
    let mut counted_games = HashMap::new();
    for state in &final_states {
        let mut scores = [0, 0];

        // スコアの計算
        for (i, row) in state.colors.iter().enumerate() {
            for (j, &color) in row.iter().enumerate() {
                if color >= 0 {
                    scores[color as usize] += state.board[i][j];
                }
            }
        }

        let player0 = &state.player0_name;
        let player1 = &state.player1_name;

        // 各プレイヤーの現在のレーティングを取得（なければデフォルト値）
        let rating0 = *ratings.entry(player0.clone()).or_insert(default_rating);
        let rating1 = *ratings.entry(player1.clone()).or_insert(default_rating);

        // 勝敗に基づいてレーティング変動を計算
        let game_result = match scores[0].cmp(&scores[1]) {
            CmpOrdering::Greater => 1.0,
            CmpOrdering::Equal => 0.5,
            CmpOrdering::Less => 0.0,
        };

        let delta = calculate_elo_change(rating0, rating1, game_result);
        ratings.insert(player0.clone(), rating0 + delta);
        ratings.insert(player1.clone(), rating1 - delta);
        *counted_games.entry(player0.clone()).or_insert(0) += 1;
        *counted_games.entry(player1.clone()).or_insert(0) += 1;
    }

    // 結果を表示
    println!("\nELO Ratings:");
    println!("============");
    let mut sorted_ratings: Vec<_> = ratings.iter().collect();
    sorted_ratings.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (player, rating) in sorted_ratings {
        println!(
            "{}: {:.1} (from {} games)",
            player, rating, counted_games[player]
        );
    }

    Ok(())
}
