import gym
import numpy as np
from gym import spaces

class RoombaSoftObstacleEnv(gym.Env):
    """
    Roomba simulator with soft obstacles represented by hidden (d, k).
    Observations: noisy (x, y, theta)
    State: internal grid cell + hidden variables
    """

    def __init__(self,
                 width=20,
                 height=20,
                 max_depth=3,
                 step_penalty=0.01,
                 noise_std=0.1):

        super().__init__()

        self.W = width
        self.H = height
        self.max_depth = max_depth
        self.noise_std = noise_std
        self.step_penalty = step_penalty

        # Actions: 0=forward, 1=backward, 2=left, 3=right, 4=turn_left, 5=turn_right
        self.action_space = spaces.Discrete(6)

        # Observation: noisy x, y, theta (continuous)
        low = np.array([0.0, 0.0, -np.pi])
        high = np.array([width, height, np.pi])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Hard obstacles (static for now)
        self.hard_obstacles = np.zeros((height, width), dtype=np.int32)

        # Visit counter for rewards
        self.visit_count = np.zeros((height, width))

        # Soft obstacle map (cells with soft obstacles)
        self.soft_obstacles = np.zeros((height, width), dtype=np.int32)

        # Pre-fill some soft obstacles
        for _ in range(10):
            x = np.random.randint(3, width-3)
            y = np.random.randint(3, height-3)
            self.soft_obstacles[y, x] = 1

        self.reset()

    def reset(self):
        # Random start in free space
        while True:
            self.x = np.random.randint(0, self.W)
            self.y = np.random.randint(0, self.H)
            if self.hard_obstacles[self.y, self.x] == 0:
                break

        self.theta = np.random.uniform(-np.pi, np.pi)

        # Hidden variables
        self.k = 0               # depth 0 = free
        self.d = None            # entry direction

        self.steps = 0
        self.visit_count[:] = 0

        return self._get_obs()

    def _get_obs(self):
        # Noisy observation of pose only
        return np.array([
            self.x + np.random.randn()*self.noise_std,
            self.y + np.random.randn()*self.noise_std,
            self.theta + np.random.randn()*self.noise_std*0.1
        ], dtype=np.float32)

    def _forward_dir(self):
        # Map theta to grid direction
        angle = self.theta % (2*np.pi)
        if angle < np.pi/4 or angle > 7*np.pi/4:
            return (1, 0)   # east
        if angle < 3*np.pi/4:
            return (0, 1)   # south
        if angle < 5*np.pi/4:
            return (-1, 0)  # west
        return (0, -1)      # north

    def _reverse_dir(self, d):
        if d == (1,0): return (-1,0)
        if d == (-1,0): return (1,0)
        if d == (0,1): return (0,-1)
        if d == (0,-1): return (0,1)
        return (0,0)

    def _direction_to_cardinal(self, dx, dy):
        if dx == 1: return "E"
        if dx == -1: return "W"
        if dy == 1: return "S"
        if dy == -1: return "N"
        return None

    def step(self, action):
        self.steps += 1

        reward = 0.0

        # Movement logic
        dx, dy = 0, 0

        if action == 0:  # forward
            dx, dy = self._forward_dir()
        elif action == 1:  # backward
            fdx, fdy = self._forward_dir()
            dx, dy = -fdx, -fdy
        elif action == 2:  # left
            dx, dy = -1, 0
        elif action == 3:  # right
            dx, dy = 1, 0
        elif action == 4:  # rotate left
            self.theta -= np.pi/4
        elif action == 5:  # rotate right
            self.theta += np.pi/4

        # Normalize theta
        self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi

        # --- Hidden state transitions ---
        if self.k == 0:
            # We are free
            nx = self.x + dx
            ny = self.y + dy

            # Check hard obstacle or wall collision
            if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H or self.hard_obstacles[ny, nx] == 1:
                reward -= 5
                nx, ny = self.x, self.y  # bounce back

            # Check if entering soft obstacle
            elif self.soft_obstacles[ny, nx] == 1:
                self.k = 1
                self.d = self._direction_to_cardinal(dx, dy)
                reward -= 1  # small penalty for entering soft obstacle

            self.x, self.y = nx, ny

        else:
            # STUCK BEHAVIOR
            # Action effects depend on d and k
            if action == 0:  # forward: push deeper
                self.k = min(self.k + 1, self.max_depth)

            elif action == 1:  # backward
                if self._direction_to_cardinal(dx, dy) == self.d:
                    self.k = max(self.k - 1, 0)

            # Movement is strongly damped:
            damping = 1.0 / (self.k + 1)
            nx = int(self.x + dx * damping)
            ny = int(self.y + dy * damping)

            # If k reached 0, we are free again
            if self.k == 0:
                self.d = None

            self.x = nx
            self.y = ny

            reward -= 0.5  # being stuck is bad

        # Step penalty for time
        reward -= self.step_penalty

        # Reward for exploring
        reward += 1.0 / (1.0 + self.visit_count[self.y, self.x])
        self.visit_count[self.y, self.x] += 1

        # Termination condition: coverage or step limit
        coverage = np.sum(self.visit_count > 0) / (self.W * self.H)
        done = coverage >= 0.95 or self.steps >= 2000

        return self._get_obs(), reward, done, {}


# --- Register environment ---
def make_env():
    return RoombaSoftObstacleEnv()
