"""
agent.py  —  DQN agent with experience replay and epsilon-greedy policy.

One DQNAgent plays BOTH sides.  The environment always encodes the board
from the current player's perspective, so the network sees a consistent
"my cells / enemy cells" view regardless of colour.

Bellman target (1-step zero-sum)
---------------------------------
    target = r + gamma * (1 - done) * (-max Q_hat(s'))

The minus sign is essential: next_state is always from the OPPONENT's
perspective.  A state great for them is bad for us.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import DQN


class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (s, a, r, s', done, next_mask).
    s' is always from the opponent's perspective (1-step zero-sum).

    Args:
        capacity : maximum number of transitions kept (oldest discarded)
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        """Append one transition."""
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        """Random mini-batch — returns six numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            np.array(next_masks,  dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent built around the spatial CNN (model.DQN).

    Unlike the old MLP version, this agent is board-size agnostic: loading
    a checkpoint trained on 6×6 and calling it on a 9×9 board just works —
    the CNN applies the same filters to a larger grid.

    Args:
        rows, cols    : board dimensions (determines reshape inside DQN)
        lr            : Adam learning rate
        gamma         : discount factor
        buffer_size   : replay buffer capacity
        batch_size    : SGD mini-batch size
        target_update : gradient steps between hard target-net copies
        channels      : hidden conv channels in the DQN
        device        : "cuda", "cpu", or "auto"
    """

    def __init__(
        self,
        rows:          int,
        cols:          int,
        lr:            float = 1e-4,
        gamma:         float = 0.99,
        buffer_size:   int   = 100_000,
        batch_size:    int   = 64,
        target_update: int   = 1_000,
        channels:      int   = 64,
        device:        str   = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.rows         = rows
        self.cols         = cols
        self.state_size   = 4 * rows * cols
        self.action_size  = rows * cols
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update = target_update

        self.q_net      = DQN(rows, cols, channels).to(self.device)
        self.target_net = DQN(rows, cols, channels).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer     = ReplayBuffer(buffer_size)
        self.steps_done = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state:      np.ndarray,
        valid_mask: np.ndarray,
        epsilon:    float = 0.0,
    ) -> int:
        """
        Epsilon-greedy action selection with invalid-action masking.

        Args:
            state      : flat float32 state array (length 3*rows*cols)
            valid_mask : boolean array (length rows*cols); True = legal move
            epsilon    : exploration rate (0 = greedy)

        Returns:
            action index
        """
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid actions — board is full?")

        if random.random() < epsilon:
            return int(np.random.choice(valid_indices))

        self.q_net.eval()
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(state_t).squeeze(0).cpu().numpy()
        self.q_net.train()

        q_vals[~valid_mask] = -1e9
        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def push(self, state, action, reward, next_state, done, next_mask):
        """Push one transition to the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done, next_mask)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """
        One gradient descent step using a random mini-batch.

        Zero-sum Bellman (1-step):
            target = r + gamma * (1 - done) * (-max Q_hat(s'))
        s' is always from the opponent's perspective — negate their best value.

        Returns scalar loss, or None if the buffer is too small to sample.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = \
            self.buffer.sample(self.batch_size)

        s  = torch.tensor(states,      device=self.device)
        a  = torch.tensor(actions,     device=self.device)
        r  = torch.tensor(rewards,     device=self.device)
        s_ = torch.tensor(next_states, device=self.device)
        d  = torch.tensor(dones,       device=self.device)
        m  = torch.tensor(next_masks,  device=self.device)

        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q              = self.target_net(s_)
            next_q[~m]          = -1e9
            best_next_q         = next_q.max(dim=1).values
            target_q            = r + self.gamma * (1.0 - d) * (-best_next_q)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self._update_target()

        return loss.item()

    def _update_target(self):
        """Hard-copy online network weights into the frozen target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save weights and metadata to a .pt checkpoint."""
        torch.save({
            "rows":        self.rows,
            "cols":        self.cols,
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "steps_done":  self.steps_done,
        }, path)

    def load(self, path: str):
        """Load weights from a checkpoint (must match current rows/cols)."""
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> "DQNAgent":
        """
        Load a fully-initialised agent from a checkpoint.

        The board dimensions are read from the checkpoint, so this works
        even when the checkpoint was trained on a different grid size than
        the one currently in use.  The CNN weights will still be applied
        correctly to any grid at inference time.
        """
        ckpt  = torch.load(path, map_location="cpu")
        agent = cls(rows=ckpt["rows"], cols=ckpt["cols"], device=device)
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        agent.steps_done = ckpt.get("steps_done", 0)
        agent.q_net.eval()
        return agent
