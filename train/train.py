import os
import pickle
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from pv_model import PVModel
from game_state import GameState, Position
from self_play import legal_moves

# Constants
BOARD_SIZE = 20
MAX_STEPS = BOARD_SIZE
NUM_ACTIONS = 4 * MAX_STEPS

class GameStateDataset(Dataset):
    def __init__(self, data_file=None):
        # デフォルトは同ディレクトリの data/dataset.pkl
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(base_dir, 'data', 'dataset.pkl')
        self.data_file = data_file or default_path

        # データファイルがなければ空リストで初期化
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'wb') as f:
                pickle.dump([], f)

        # データ読み込み
        with open(self.data_file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        st, pol_list, value = self.data[idx]
        # Build input tensor: [5, BOARD_SIZE, BOARD_SIZE]
        board = torch.tensor(st.board, dtype=torch.float32)
        colors = torch.tensor(st.colors, dtype=torch.float32)
        rocks = torch.tensor(st.rocks, dtype=torch.float32)
        p0 = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
        p1 = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
        p0[st.player0.y, st.player0.x] = 1.0
        p1[st.player1.y, st.player1.x] = 1.0
        inp = torch.stack([board, colors, rocks, p0, p1], dim=0)

        # Build fixed-length policy vector
        policy_vec = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        moves = legal_moves(st)
        for i, prob in enumerate(pol_list):
            if i < len(moves):
                dir_idx, steps = moves[i]
                action_id = dir_idx * MAX_STEPS + (steps - 1)
                if 0 <= action_id < NUM_ACTIONS:
                    policy_vec[action_id] = prob

        # Value label
        value_label = torch.tensor(value, dtype=torch.float32)
        return inp, policy_vec, value_label

if __name__ == '__main__':
    # データセット準備
    dataset = GameStateDataset()
    if len(dataset) == 0:
        print('No training data found in', dataset.data_file)
        sys.exit(0)

    # DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # CNN-based PVModel
    model = PVModel(
        board_size=BOARD_SIZE,
        in_channels=5,
        num_res_blocks=3,
        filter_size=8,
        max_steps=MAX_STEPS,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
    value_loss_fn = nn.MSELoss()

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        total_pl, total_vl = 0.0, 0.0
        for inp, policy, value in loader:
            optimizer.zero_grad()
            p_pred, v_pred = model(inp)
            # Policy loss: KL divergence
            pl = policy_loss_fn(torch.log(p_pred + 1e-8), policy)
            # Value loss: MSE
            vl = value_loss_fn(v_pred.squeeze(), value)
            loss = pl + vl
            loss.backward()
            optimizer.step()
            total_pl += pl.item()
            total_vl += vl.item()
        print(f"Epoch {epoch+1}: Policy Loss={total_pl/len(loader):.4f}, Value Loss={total_vl/len(loader):.4f}")

    # Save model weights
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcts_policy_value_model.pt')
    torch.save(model.state_dict(), save_path)
    print('Saved model weights to', save_path)
