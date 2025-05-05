package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/montplusa/skate-area-battle-game/pkg/ai/montplusa"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake"
	"github.com/montplusa/skate-area-battle-game/pkg/ai/sneuaiolake/networks"
)

func main() {
	// Parse command line flags
	episodes := flag.Int("episodes", 1000, "Number of training episodes")
	batchSize := flag.Int("batch", 32, "Training batch size")
	name := flag.String("name", "sample", "Name of output weights")
	initialTemp := flag.Float64("temp-init", 1.0, "Initial exploration temperature")
	finalTemp := flag.Float64("temp-final", 0.1, "Final exploration temperature")
	reportInterval := flag.Int("report", 5, "Report progress every N episodes")
	learningRate := flag.Float64("lr", 0.01, "Learning rate for neural network training")

	flag.Parse()

	// Initialize random seed

	// Create the Sneuaiolake AI
	networkConfig := networks.LoadConfig_init()
	networkConfig.Name = *name
	networkConfig.Weights = nil

	// Configure training
	config := sneuaiolake.TrainingConfig{
		Episodes:             *episodes,
		BatchSize:            *batchSize,
		Name:                 *name,
		InitialTemperature:   *initialTemp,
		FinalTemperature:     *finalTemp,
		ReportInterval:       *reportInterval,
		LearningRate:         *learningRate,
		InitialNetworkConfig: networkConfig,
		OpponentAI:           montplusa.New(),
	}

	// Run training
	fmt.Println("Starting training...")
	if err := sneuaiolake.Train(config); err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Println("Training complete!", *name)
}
