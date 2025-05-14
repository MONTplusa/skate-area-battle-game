use anyhow::{Result, bail};
use clap::Parser;
use rayon::prelude::*;
use regex::Regex;
use skate_area_battle_game::{
    ai::{
        AI, montplusa::MontplusaAI, random::RandomAI, sneuaiolake::SneuaiolakeAI,
        statiolake::StatiolakeAI, trivial::TrivialAI,
    },
    game::GameRunner,
};
use std::{
    cell::RefCell,
    cmp::Ordering as CmpOrdering,
    fs,
    sync::atomic::Ordering as AtomicOrdering,
    sync::{OnceLock, atomic::AtomicI32},
};

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
    result_dir: String,

    #[arg(long, help = "出力ファイル名のプレフィックス")]
    prefix: String,

    #[arg(long, help = "並列処理のスレッド数")]
    jobs: Option<usize>,

    #[arg(long, help = "ログを表示する")]
    show_log: bool,
}

fn main() -> Result<()> {
    let opts = Options::parse();
    play(&opts)
}

fn play(opts: &Options) -> Result<()> {
    // 出力ディレクトリの作成
    fs::create_dir_all(&opts.result_dir)?;

    // 既存ファイルの最大連番を取得
    let start_seq = find_next_sequence_number(&opts.result_dir, &opts.prefix)?;
    println!("連番 {:05} から開始します", start_seq);

    if let Some(jobs) = opts.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()?;
    }

    // AI は毎回モデルをロードすると重いので、スレッドローカルストレージにキャッシュする
    thread_local! {
        static P0_AI: OnceLock<RefCell<Box<dyn AI + Send>>> = OnceLock::new();
        static P1_AI: OnceLock<RefCell<Box<dyn AI + Send>>> = OnceLock::new();
    }

    // 対戦の実行（並列化）
    let seq_number = AtomicI32::new(start_seq as i32);
    let battle_results = (0..opts.games)
        .into_par_iter()
        .map(|_| {
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

                    let i = seq_number.fetch_add(1, AtomicOrdering::SeqCst);
                    // スコアを出力
                    println!(
                        "対戦 #{}: スコア - {}: {}, {}: {}",
                        i, state.player0_name, scores[0], state.player1_name, scores[1]
                    );

                    // 勝者を決定
                    let winner = match scores[0].cmp(&scores[1]) {
                        CmpOrdering::Greater => Some(0),
                        CmpOrdering::Less => Some(1),
                        CmpOrdering::Equal => None,
                    };

                    // 結果の保存
                    let seq_num = start_seq + i as u32;
                    let filename =
                        format!("{}/{}_{:05}.json", opts.result_dir, opts.prefix, seq_num);
                    fs::write(&filename, serde_json::to_string(&result)?)?;

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

fn create_ai(model_path: Option<&str>) -> Result<Box<dyn AI + Send>> {
    match model_path {
        Some(path) if path.contains("/") || path.contains("\\") => {
            Ok(Box::new(SneuaiolakeAI::new(path)?))
        }
        Some("random") => Ok(Box::new(RandomAI::new())),
        Some("trivial") => Ok(Box::new(TrivialAI::new())),
        Some("statiolake") => Ok(Box::new(StatiolakeAI::new())),
        Some("montplusa") => Ok(Box::new(MontplusaAI::new())),
        Some(name) => bail!("Invalid model specified: {name}"),
        None => Ok(Box::new(RandomAI::new())),
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
