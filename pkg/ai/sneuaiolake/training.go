package sneuaiolake

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
	"github.com/patrikeh/go-deep/training"
)

// GameRecord stores data from a played game for training
type GameRecord struct {
	States  [][]float64 // Game states as feature vectors
	Rewards []float64   // Final rewards for each state
}

// TrainingConfig specifies parameters for self-play training
type TrainingConfig struct {
	Episodes              int           // Number of self-play games
	BatchSize             int           // Batch size for neural network updates
	SaveInterval          int           // Save model every N episodes
	Name                  string        // Name of the output weights file
	InitialTemperature    float64       // Starting exploration temperature
	FinalTemperature      float64       // Final exploration temperature
	BoardSize             int           // Size of the game board (N×N)
	OpponentNetworkConfig NetworkConfig // Path to opponent weights file (empty for zeroed weights)
	ReportInterval        int           // How often to report progress (default every 10 episodes)
	UseMCTS               bool          // Whether to use MCTS for exploration (very slow)
	MCTSSimulations       int           // Number of MCTS simulations if enabled
}

// Default values
const (
	defaultMCTSSimulations = 50 // Reduced from 400 for better performance
)

// TrainingStats tracks metrics during training
type TrainingStats struct {
	Wins        int
	Losses      int
	Draws       int
	TotalScore0 float64
	TotalScore1 float64
	TotalTurns  int
	StartTime   time.Time
}

// Train performs self-play reinforcement learning
func (s *Sneuaiolake) Train(config TrainingConfig) error {
	if config.BoardSize <= 0 {
		config.BoardSize = 20
	}

	if config.ReportInterval <= 0 {
		config.ReportInterval = 10
	}

	// Setup training data collection
	var trainingData training.Examples

	// Set up opponent model
	var opponent *Sneuaiolake
	var err error

	// 指定されたパスから対戦相手の重みを読み込む
	opponent, err = New(config.OpponentNetworkConfig)
	if err != nil {
		return fmt.Errorf("failed to load opponent weights: %v", err)
	}
	fmt.Println("Loaded opponent:", config.OpponentNetworkConfig.Name)

	fmt.Println("Starting self-play training...")
	fmt.Println("- Episodes:", config.Episodes)
	fmt.Println("- Batch size:", config.BatchSize)
	fmt.Println("- Board size:", config.BoardSize)
	fmt.Println("- Temperature:", config.InitialTemperature, "->", config.FinalTemperature)
	fmt.Println("- Save interval:", config.SaveInterval)
	fmt.Println("- Report interval:", config.ReportInterval)
	fmt.Println("- Using MCTS:", config.UseMCTS)
	if config.UseMCTS {
		fmt.Println("- MCTS simulations:", config.MCTSSimulations)
	}
	fmt.Println()

	// Training statistics
	stats := TrainingStats{
		StartTime: time.Now(),
	}
	lastReportTime := time.Now()

	// Main training loop
	for episode := 0; episode < config.Episodes; episode++ {
		// Update temperature schedule (linear decay)
		progress := float64(episode) / float64(config.Episodes)
		s.temperature = config.InitialTemperature - progress*(config.InitialTemperature-config.FinalTemperature)
		opponent.temperature = s.temperature // 対戦相手も同じ温度を使用

		// Initialize a game
		states := game.GenerateInitialStates(1, config.BoardSize)
		state := states[0]

		// Collect game data
		var gameStates [][]float64
		turn := 0

		// Play a complete game
		gameOver := false
		for !gameOver {
			// Get current player
			currentPlayer := state.Turn

			// Get legal moves
			moves := state.LegalMoves(currentPlayer)

			// Check if game is over
			if len(moves) == 0 {
				// Check if both players have no moves
				otherMoves := state.LegalMoves(1 - currentPlayer)
				if len(otherMoves) == 0 {
					gameOver = true
					break
				}

				// Current player has no moves, switch turn
				state.Turn = 1 - currentPlayer
				continue
			}

			// Record current state from both players' perspectives
			features0 := stateToFeatures(state, 0)
			features1 := stateToFeatures(state, 1)

			gameStates = append(gameStates, features0, features1)

			// Evaluate all moves
			var bestMove game.Move
			var bestScore float64

			// Add some exploration during training
			useExploration := rand.Float64() < s.temperature

			// 現在のプレイヤーに応じて使用するAIを選択
			currentAI := s
			if currentPlayer == 1 {
				currentAI = opponent
			}

			// Choose move selection strategy based on configuration
			if config.UseMCTS && useExploration && rand.Float64() < 0.3 {
				// Use MCTS with neural network evaluation for guided exploration (only 30% of the time)
				bestMove = currentAI.MCTS(state, currentPlayer, config.MCTSSimulations)
			} else if useExploration {
				// Two exploration strategies:
				// 1. Pure random moves (less frequent)
				// 2. Softmax selection based on evaluation scores (more frequent)
				if rand.Float64() < 0.3 {
					// Pure random exploration (30% of exploration)
					bestMove = moves[rand.Intn(len(moves))]
				} else {
					// Softmax exploration based on evaluation scores (70% of exploration)
					// This is much faster than MCTS but still provides good exploration
					scores := make([]float64, len(moves))
					maxScore := -1e9

					// Evaluate all moves
					for i, move := range moves {
						scores[i] = currentAI.Evaluate(move.State, currentPlayer)
						if scores[i] > maxScore {
							maxScore = scores[i]
						}
					}

					// Apply temperature and calculate softmax probabilities
					var sum float64
					probs := make([]float64, len(scores))

					for i, score := range scores {
						// Apply temperature (higher temp = more uniform distribution)
						probs[i] = math.Exp((score - maxScore) / currentAI.temperature)
						sum += probs[i]
					}

					// Normalize probabilities
					for i := range probs {
						probs[i] /= sum
					}

					// Select move based on probabilities
					r := rand.Float64()
					cumProb := 0.0

					for i, prob := range probs {
						cumProb += prob
						if r <= cumProb {
							bestMove = moves[i]
							break
						}
					}

					// Fallback if something went wrong with probability selection
					if bestMove.State == nil && len(moves) > 0 {
						bestMove = moves[0]
					}
				}
			} else {
				// Greedy evaluation using the neural network (no exploration)
				for _, move := range moves {
					nextState := move.State
					score := currentAI.Evaluate(nextState, currentPlayer)

					if score > bestScore || bestMove.State == nil {
						bestScore = score
						bestMove = move
					}
				}
			}

			// Apply the selected move
			state = state.ApplyMove(bestMove)
			turn++
		}

		// Calculate final scores
		player0Score := calculateScore(state, 0)
		player1Score := calculateScore(state, 1)

		// Update statistics
		stats.TotalScore0 += player0Score
		stats.TotalScore1 += player1Score
		stats.TotalTurns += turn

		// Calculate rewards (win/loss/draw)
		var reward0, reward1 float64
		if player0Score > player1Score {
			reward0 = 1.0
			reward1 = -1.0
			stats.Wins++
		} else if player0Score < player1Score {
			reward0 = -1.0
			reward1 = 1.0
			stats.Losses++
		} else {
			reward0 = 0.0
			reward1 = 0.0
			stats.Draws++
		}

		// Create training examples - プレイヤー0の視点からのみ学習（自分のモデル）
		for i := 0; i < len(gameStates); i += 2 {
			if i+1 >= len(gameStates) {
				break
			}

			// Player 0's perspective
			trainingData = append(trainingData, training.Example{
				Input:    gameStates[i],
				Response: []float64{reward0},
			})
		}

		_ = reward1

		// Train the network periodically
		if len(trainingData) >= config.BatchSize {
			// デバッグ情報を表示
			trainStart := time.Now()
			fmt.Printf("Training on %d examples... ", len(trainingData))

			// Shuffle training data
			trainingData.Shuffle()

			// Create trainer with adjusted parameters
			epoch := 1
			trainer := training.NewBatchTrainer(training.NewSGD(s.config.LearningRate, 0.5, 0.0, false), epoch, config.BatchSize, 6)

			// Train on batch
			trainer.Train(s.network, trainingData, nil, 10)

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
			avgScore0 := stats.TotalScore0 / float64(episode+1)
			avgScore1 := stats.TotalScore1 / float64(episode+1)
			gamesPerSecond := float64(config.ReportInterval) / elapsed.Seconds()

			// Calculate ETA
			gamesRemaining := config.Episodes - (episode + 1)
			etaSeconds := float64(gamesRemaining) / gamesPerSecond
			eta := time.Duration(etaSeconds) * time.Second

			fmt.Printf("[%d/%d] Win: %.1f%% (W:%d L:%d D:%d) | Temp: %.2f | Turns: %.1f | Score: %.1f vs %.1f | %.2f games/sec | Elapsed: %s | ETA: %s\n",
				episode+1, config.Episodes, winRate, stats.Wins, stats.Losses, stats.Draws,
				s.temperature, avgTurns, avgScore0, avgScore1, gamesPerSecond,
				formatDuration(totalElapsed), formatDuration(eta))

			lastReportTime = now
		}

		// Save model periodically
		if (episode+1)%config.SaveInterval == 0 || episode == config.Episodes-1 {
			fmt.Printf("Episode %d/%d - Saving model to %s...\n",
				episode+1, config.Episodes, config.Name)

			// Extract and save weights
			s.config.Weights = s.network.Dump().Weights
			if err := saveConfig(s.config); err != nil {
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
		stats.TotalScore0/float64(config.Episodes),
		stats.TotalScore1/float64(config.Episodes))
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
