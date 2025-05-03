package sneuaiolake

import (
	"math"
	"math/rand"

	"github.com/montplusa/skate-area-battle-game/pkg/game"
)

// MCTSNode represents a node in the Monte Carlo search tree
type MCTSNode struct {
	state        *game.GameState
	parent       *MCTSNode
	children     []*MCTSNode
	visits       int
	totalReward  float64
	unexplored   []game.Move
	playerToMove int
}

// MCTS performs Monte Carlo Tree Search to find the best move
func (s *Sneuaiolake) MCTS(state *game.GameState, player int, simulations int) game.Move {
	// Create root node
	root := &MCTSNode{
		state:        state,
		unexplored:   state.LegalMoves(player),
		playerToMove: player,
	}

	// Run simulations
	for i := 0; i < simulations; i++ {
		// Selection and expansion
		node := s.selectNode(root)

		// Simulation
		reward := s.simulate(node.state, node.playerToMove)

		// Backpropagation
		s.backpropagate(node, reward)
	}

	// Choose best child of root
	var bestChild *MCTSNode
	var bestScore float64

	for _, child := range root.children {
		// Use exploitation only (not exploration) for final selection
		score := child.totalReward / float64(child.visits)

		if bestChild == nil || score > bestScore {
			bestChild = child
			bestScore = score
		}
	}

	// Find the move that led to this child
	for _, move := range state.LegalMoves(player) {
		if moveEqual(move, bestChild.state) {
			return move
		}
	}

	// Fallback to first legal move if something went wrong
	moves := state.LegalMoves(player)
	if len(moves) > 0 {
		return moves[0]
	}

	// This should never happen
	panic("No legal moves found in MCTS")
}

// selectNode selects a node for expansion using UCB1
func (s *Sneuaiolake) selectNode(node *MCTSNode) *MCTSNode {
	// If node is not fully expanded, expand it
	if len(node.unexplored) > 0 {
		// Choose a random unexplored move
		index := rand.Intn(len(node.unexplored))
		move := node.unexplored[index]

		// Remove the move from unexplored list
		node.unexplored = append(node.unexplored[:index], node.unexplored[index+1:]...)

		// Create a new child node
		childState := move.State
		childNode := &MCTSNode{
			state:        childState,
			parent:       node,
			unexplored:   childState.LegalMoves(childState.Turn),
			playerToMove: childState.Turn,
		}

		// Add child to parent
		node.children = append(node.children, childNode)

		return childNode
	}

	// If all moves are explored, select best child using UCB1
	if len(node.children) == 0 {
		return node // Terminal node
	}

	// Find child with highest UCB1 value
	var bestChild *MCTSNode
	var bestUCB float64

	// Exploration parameter
	c := 1.414 // sqrt(2)

	for _, child := range node.children {
		// Calculate UCB1
		exploitation := child.totalReward / float64(child.visits)
		exploration := c * math.Sqrt(math.Log(float64(node.visits))/float64(child.visits))
		ucb := exploitation + exploration

		if bestChild == nil || ucb > bestUCB {
			bestChild = child
			bestUCB = ucb
		}
	}

	// Recursively select from best child
	return s.selectNode(bestChild)
}

// simulate runs a simulation from the given state
func (s *Sneuaiolake) simulate(state *game.GameState, player int) float64 {
	// Make a copy of the state
	simState := state.Clone()
	simPlayer := player

	// Run a random simulation until game ends
	for {
		// Get legal moves for current player
		moves := simState.LegalMoves(simPlayer)

		// If no moves, try other player
		if len(moves) == 0 {
			otherMoves := simState.LegalMoves(1 - simPlayer)
			if len(otherMoves) == 0 {
				// Game is over, calculate scores
				score0 := calculateScore(simState, 0)
				score1 := calculateScore(simState, 1)

				// Return reward from perspective of original player
				if player == 0 {
					return (score0 - score1) / 1000.0 // Normalize score
				} else {
					return (score1 - score0) / 1000.0
				}
			}

			// Switch player
			simPlayer = 1 - simPlayer
			continue
		}

		// Choose a random move
		move := moves[rand.Intn(len(moves))]

		// Apply the move
		simState = move.State

		// Switch player
		simPlayer = simState.Turn
	}
}

// backpropagate updates the statistics for all nodes in the path
func (s *Sneuaiolake) backpropagate(node *MCTSNode, reward float64) {
	// Update statistics for this node
	node.visits++
	node.totalReward += reward

	// Recursively update parent
	if node.parent != nil {
		// Invert reward for parent (opponent's perspective)
		s.backpropagate(node.parent, -reward)
	}
}

// moveEqual checks if a move leads to the given state
func moveEqual(move game.Move, state *game.GameState) bool {
	return move.State.Player0.X == state.Player0.X &&
		move.State.Player0.Y == state.Player0.Y &&
		move.State.Player1.X == state.Player1.X &&
		move.State.Player1.Y == state.Player1.Y
}
