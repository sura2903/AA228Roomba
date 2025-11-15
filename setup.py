# Setup
import gym
from gym import spaces
import numpy as np

class RoombaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, room_width=10, room_height=10, roomba_size=1, render_mode=None):
        super().__init__()

        # Grid-world dimensions
        self.room_width = room_width
        self.room_height = room_height
        self.roomba_size = roomba_size
        self.grid_width = room_width // roomba_size
        self.grid_height = room_height // roomba_size

        # State: (x, y, theta) where theta ∈ {0, 1, 2, 3}
        self.theta_vals = 4
        self.state_space_shape = (self.grid_width, self.grid_height, self.theta_vals)

        # Observation space is noisy estimate of state
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.grid_width, self.grid_height, 3], dtype=np.float32)
        )

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Transition probabilities
        self.p_intended = 0.7
        self.p_adjacent = 0.1   # left or right relative to intended
        self.p_fail = 0.1       # stays in place

        # Visit counter for reward shaping
        self.visit_counts = np.zeros((self.grid_width, self.grid_height), dtype=np.int32)

        # Obstacles: represented as grid cells
        self.obstacles = self._generate_static_obstacles()

        # Episode termination settings
        self.coverage_goal = 0.95
        self.max_steps = 500

        self.render_mode = render_mode
        self.current_step = 0
        self.state = None
        
        self.time_penalty = 0.01 # Tune this later if needed
        self.steps_taken = 0


    def _generate_static_obstacles(self):
        grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int32)
        # Example: add a few rectangular obstacles
        grid[2:4, 5:7] = 1
        grid[6:7, 1:3] = 1
        return grid

    def _sample_observation(self, state):
        x, y, theta = state
        noisy_x = np.random.normal(x, 0.3)
        noisy_y = np.random.normal(y, 0.3)
        noisy_theta = np.random.normal(theta, 0.1)
        return np.array([noisy_x, noisy_y, noisy_theta], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.visit_counts[:] = 0

        # Start in a random free cell
        free_cells = np.argwhere(self.obstacles == 0)
        idx = np.random.choice(len(free_cells))
        x, y = free_cells[idx]
        theta = np.random.choice(self.theta_vals)
        self.state = (x, y, theta)

        obs = self._sample_observation(self.state)
        return obs, {}

    def _transition_state(self, action):
        x, y, theta = self.state

        # Candidate moves: intended + 2 adjacent + staying
        moves = {
            0: (0, 1),   # up
            1: (0, -1),  # down
            2: (-1, 0),  # left
            3: (1, 0),   # right
        }

        # Build probability distribution
        choices = []
        probs = []

        # Intended action
        choices.append(action)
        probs.append(self.p_intended)

        # Adjacent actions (rotate left/right in grid)
        choices.append((action - 1) % 4)
        choices.append((action + 1) % 4)
        probs.extend([self.p_adjacent, self.p_adjacent])

        # Failure to move
        choices.append(None)
        probs.append(self.p_fail)

        picked = np.random.choice(len(choices), p=probs)
        act = choices[picked]

        if act is None:
            return self.state

        dx, dy = moves[act]
        new_x = x + dx
        new_y = y + dy

        # Check boundaries and obstacles
        if (
            new_x < 0
            or new_x >= self.grid_width
            or new_y < 0
            or new_y >= self.grid_height
            or self.obstacles[new_x, new_y] == 1
        ):
            return self.state

        return (new_x, new_y, act)  # theta replaced by direction moved

    def _reward(self, state):
        x, y, _ = state
        # Penalty for obstacle or wall handled in transition (no movement)
        if self.obstacles[x, y] == 1:
            return -5.0

        base_reward = 1.0 / (1 + self.visit_counts[x, y])
        base_reward -= self.time_penalty * self.steps_taken
        return base_reward

    def step(self, action):
        self.current_step += 1

        next_state = self._transition_state(action)
        self.state = next_state

        x, y, _ = next_state
        self.visit_counts[x, y] += 1

        reward = self._reward(next_state)

        # Coverage metric
        explored = np.sum(self.visit_counts > 0) / (self.grid_width * self.grid_height)
        terminated = explored >= self.coverage_goal or self.current_step >= self.max_steps

        obs = self._sample_observation(next_state)
        return obs, reward, terminated, False, {}

    def render(self):
        pass

    def close(self):
        pass
