package sneuaiolake

import (
	"fmt"
	"time"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
	"github.com/patrikeh/go-deep/training"
)

// GameRecord stores data from a played game for training
type GameRecord struct {
	Features [][]float64 // Game states as feature vectors
	Rewards  []float64   // Final rewards for each state
}

// TrainingConfig specifies parameters for self-play training
type TrainingConfig struct {
	Episodes             int           // Number of self-play games
	BatchSize            int           // Batch size for neural network updates
	ReportInterval       int           // How often to report progress (default every 10 episodes)
	Name                 string        // Name of the output weights file
	InitialTemperature   float64       // Starting exploration temperature
	FinalTemperature     float64       // Final exploration temperature
	LearningRate         float64       // Learning rate for the neural network
	InitialNetworkConfig NetworkConfig // Initial network configuration
	OpponentAI           game.AI       // Opponent AI for self-play
}

// TrainingStats tracks metrics during training
type TrainingStats struct {
	Wins         int
	Losses       int
	Draws        int
	TotalMyScore float64
	TotalOpScore float64
	TotalTurns   int
	StartTime    time.Time
}

// Train performs self-play reinforcement learning
func Train(config TrainingConfig) error {
	if config.ReportInterval <= 0 {
		return fmt.Errorf("report interval must be greater than 0")
	}

	myAI, err := New(config.InitialNetworkConfig)
	if err != nil {
		return fmt.Errorf("failed to create AI: %w", err)
	}

	opAI := config.OpponentAI
	fmt.Println("Opponent:", opAI.Name())

	fmt.Println("Starting self-play training...")
	fmt.Println("- Episodes:", config.Episodes)
	fmt.Println("- Batch size:", config.BatchSize)
	fmt.Println("- Temperature:", config.InitialTemperature, "->", config.FinalTemperature)
	fmt.Println("- Report interval:", config.ReportInterval)
	fmt.Println()

	// Training statistics
	// Setup training data collection
	var trainingData training.Examples
	stats := TrainingStats{
		StartTime: time.Now(),
	}
	lastReportTime := time.Now()

	// Main training loop
	for episode := 0; episode < config.Episodes; episode++ {
		// Update temperature schedule (linear decay)
		runner := game.NewGameRunner(myAI, opAI)
		result := runner.Run()

		// Calculate final scores
		myScore := calculateScore(result.FinalState, 0)
		opScore := calculateScore(result.FinalState, 1)

		if myScore > opScore {
			stats.Wins++
		} else if myScore < opScore {
			stats.Losses++
		} else {
			stats.Draws++
		}

		stats.TotalMyScore += myScore
		stats.TotalOpScore += opScore
		eval := computeEval(myScore, opScore)

		// 特徴抽出
		for _, move := range result.Moves {
			// 行動前の対象プレイヤー
			turn := 1 - move.State.Turn
			features := stateToFeatures(move.State, turn)
			evalSign := 1.0
			if turn == 1 {
				evalSign = -1.0
			}
			reward := []float64{eval * evalSign}

			trainingData = append(trainingData, training.Example{
				Input:    features,
				Response: reward,
			})
		}

		// Train the network periodically
		if len(trainingData) >= config.BatchSize {
			// デバッグ情報を表示
			trainStart := time.Now()
			fmt.Printf("Training on %d examples... ", len(trainingData))

			// Shuffle training data
			trainingData.Shuffle()

			// Create trainer with adjusted parameters
			trainer := training.NewTrainer(training.NewSGD(config.LearningRate, 0.5, 0.0, false), 1)

			// Train on batch
			iterations := len(trainingData)/config.BatchSize + 1
			trainer.Train(myAI.network, trainingData, nil, iterations)

			// トレーニング時間を表示
			trainTime := time.Since(trainStart)
			fmt.Printf("done in %s\n", formatDuration(trainTime))

			// Clear training data after using it
			trainingData = nil
		}

		// Report progress at specified intervals
		if (episode+1)%config.ReportInterval == 0 || episode == config.Episodes-1 {
			now := time.Now()
			elapsed := now.Sub(lastReportTime)
			totalElapsed := now.Sub(stats.StartTime)

			// Calculate statistics
			winRate := float64(stats.Wins) / float64(stats.Wins+stats.Losses+stats.Draws) * 100
			avgTurns := float64(stats.TotalTurns) / float64(episode+1)
			avgScore0 := stats.TotalMyScore / float64(episode+1)
			avgScore1 := stats.TotalOpScore / float64(episode+1)
			gamesPerSecond := float64(config.ReportInterval) / elapsed.Seconds()

			// Calculate ETA
			gamesRemaining := config.Episodes - (episode + 1)
			etaSeconds := float64(gamesRemaining) / gamesPerSecond
			eta := time.Duration(etaSeconds) * time.Second

			fmt.Printf("[%d/%d] Win: %.1f%% (W:%d L:%d D:%d) | Turns: %.1f | Score: %.1f vs %.1f | %.2f games/sec | Elapsed: %s | ETA: %s\n",
				episode+1, config.Episodes, winRate, stats.Wins, stats.Losses, stats.Draws,
				avgTurns, avgScore0, avgScore1, gamesPerSecond,
				formatDuration(totalElapsed), formatDuration(eta))

			lastReportTime = now

			fmt.Printf("Episode %d/%d - Saving model to %s...\n",
				episode+1, config.Episodes, myAI.config.Name)

			// Extract and save weights
			myAI.config.Weights = myAI.network.Dump().Weights
			if err := myAI.config.export(); err != nil {
				return err
			}
		}
	}

	// Final stats
	totalElapsed := time.Since(stats.StartTime)
	fmt.Println("\nTraining completed!")
	fmt.Printf("Total training time: %s\n", formatDuration(totalElapsed))
	fmt.Printf("Final win rate: %.1f%% (W:%d L:%d D:%d)\n",
		float64(stats.Wins)/float64(stats.Wins+stats.Losses+stats.Draws)*100,
		stats.Wins, stats.Losses, stats.Draws)
	fmt.Printf("Average game length: %.1f turns\n", float64(stats.TotalTurns)/float64(config.Episodes))
	fmt.Printf("Average scores: %.1f vs %.1f\n",
		stats.TotalMyScore/float64(config.Episodes),
		stats.TotalOpScore/float64(config.Episodes))
	fmt.Printf("Model saved to: %s\n", config.Name)

	return nil
}

// formatDuration returns a human-readable string for a duration
func formatDuration(d time.Duration) string {
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second

	if h > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", h, m, s)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}

// calculateScore computes the total score for a player
func calculateScore(state *game.GameState, player int) float64 {
	boardSize := len(state.Board)
	score := 0.0

	for y := 0; y < boardSize; y++ {
		for x := 0; x < boardSize; x++ {
			if state.Colors[y][x] == player {
				score += float64(state.Board[y][x])
			}
		}
	}

	return score
}

func computeEval(myScore, opScore float64) float64 {
	diff := myScore - opScore
	diff = max(min(diff, 20000.0), -20000.0)
	return normalizeFeature(diff, -20000.0, 20000.0)
}
