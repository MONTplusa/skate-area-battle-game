package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake/networks"
)

func main() {
	// Parse command line flags
	episodes := flag.Int("episodes", 1000, "Number of training episodes")
	batchSize := flag.Int("batch", 32, "Training batch size")
	saveInterval := flag.Int("save", 5, "Save interval")
	name := flag.String("name", "sample", "Name of output weights")
	initialTemp := flag.Float64("temp-init", 1.0, "Initial exploration temperature")
	finalTemp := flag.Float64("temp-final", 0.1, "Final exploration temperature")
	boardSize := flag.Int("board", 20, "Board size (smaller is faster)")
	reportInterval := flag.Int("report", 5, "Report progress every N episodes")
	useMCTS := flag.Bool("mcts", false, "Use MCTS for exploration (very slow)")
	mctsSimulations := flag.Int("mcts-sims", 20, "Number of MCTS simulations if enabled")
	learningRate := flag.Float64("lr", 0.01, "Learning rate for neural network training")

	flag.Parse()

	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	// Create the Sneuaiolake AI
	// networkConfig := sneuaiolake.DefaultNetworkConfig()
	// networkConfig.Name = *name
	networkConfig := networks.LoadConfig_init()
	ai, err := sneuaiolake.New(networkConfig)
	if err != nil {
		log.Fatalf("Failed to create AI: %v", err)
	}

	// Set custom learning rate if specified
	if *learningRate != 0.01 {
		ai.SetLearningRate(*learningRate)
	}

	// Configure training
	config := sneuaiolake.TrainingConfig{
		Episodes:              *episodes,
		BatchSize:             *batchSize,
		SaveInterval:          *saveInterval,
		Name:                  *name,
		InitialTemperature:    *initialTemp,
		FinalTemperature:      *finalTemp,
		BoardSize:             *boardSize,
		OpponentNetworkConfig: sneuaiolake.DefaultNetworkConfig(),
		ReportInterval:        *reportInterval,
		UseMCTS:               *useMCTS,
		MCTSSimulations:       *mctsSimulations,
	}

	// Run training
	fmt.Println("Starting training...")
	if err := ai.Train(config); err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Println("Training complete! New weight name is:", *name)
}
