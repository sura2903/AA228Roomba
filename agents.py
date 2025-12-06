import numpy as np
import csv
import pandas as pd
from collections import defaultdict

class TransitionLogger:
    """
    Log all MDP transitions to CSV.
    
    CSV FORMAT:
    s_x, s_y, s_theta, s_k, action, reward, sp_x, sp_y, sp_theta, sp_k, episode, step
    
    This format is perfect for:
    - Analyzing reward function
    - Debugging transitions
    - Training other models (supervised learning)
    - Visualizing state space coverage
    """
    
    def __init__(self, filepath="transitions.csv"):
        self.filepath = filepath
        self.transitions = []
        self.episode_num = 0
        
        # Column names
        self.columns = [
            's_x', 's_y', 's_theta', 's_k', 's_d',  # Current state
            'action',                         # Action taken
            'reward',                         # Immediate reward
            'sp_x', 'sp_y', 'sp_theta', 'sp_k', 'sp_d',  # Next state
            'episode',                        # Episode number
            'step',                          # Step within episode
            'done',                          # Terminal state flag
            'coverage'                       # Coverage fraction
        ]
        
        # Initialize CSV file with header
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
        
        print(f"✓ Initialized transition logger: {self.filepath}")
    
    def log_transition(self, state, action, reward, next_state, step, done=False, coverage=0.0):
        """
        Log a single transition.
        
        Args:
            state: (x, y, theta, k)
            action: int [0-3]
            reward: float
            next_state: (x', y', theta', k, d')
            step: int - step number in episode
            done: bool - episode terminated
            coverage: float - fraction of map covered
        """
        transition = [
            state[0], state[1], state[2], state[3],state[4],     # s
            action,                                      # a
            reward,                                      # r
            next_state[0], next_state[1], next_state[2], next_state[3], next_state[4],  # s'
            self.episode_num,                           # episode
            step,                                       # step
            1 if done else 0,                          # done
            coverage                                   # coverage
        ]
        
        self.transitions.append(transition)
    
    def save_batch(self):
        """Write buffered transitions to CSV."""
        if not self.transitions:
            return
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.transitions)
        
        self.transitions = []
    
    def increment_episode(self):
        """Move to next episode."""
        self.save_batch()
        self.episode_num += 1
    
    def get_dataframe(self):
        """Load logged data as pandas DataFrame."""
        self.save_batch()
        return pd.read_csv(self.filepath)

# ==================== BASE AGENT ====================

class BaseAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 log_transitions=True, log_filepath="transitions.csv", log_filepath_eval = "transitions_eval.csv"):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.n_actions = env.action_space.n
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

        self.episode_rewards = []
        self.episode_lengths = []
        self.coverage_history = []

        if log_transitions:
            self.logger = TransitionLogger(log_filepath)
            self.logger_eval = TransitionLogger(log_filepath_eval)
        else:
            self.logger = None
            self.logger_eval = None

    def get_state(self):
        """Return current environment state as tuple."""
        return (self.env.x, self.env.y, self.env.theta, self.env.k, self.env.d)

    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, n_episodes=500, eval_every=100, verbose=False, steps=2000):
        """Generic training loop for any agent with train_episode()."""
        print("="*80)
        print(f"Training {self.__class__.__name__}")
        print("="*80)

        self.episode_rewards = []
        self.episode_lengths = []
        self.coverage_history = []

        for ep in range(n_episodes):
            reward, length, coverage, stuck_count = self.train_episode(max_steps = steps, verbose=verbose and ep % 200 == 0)
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.coverage_history.append(coverage)

            self.decay_epsilon()

            if ep % eval_every == 0:
                window = min(eval_every, len(self.episode_rewards))
                avg_r = sum(self.episode_rewards[-window:]) / window
                avg_l = sum(self.episode_lengths[-window:]) / window
                avg_c = sum(self.coverage_history[-window:]) / window
                print(f"[Episode {ep}] Reward={avg_r:.2f} Length={avg_l:.1f} Cover={avg_c:.3f} Eps={self.epsilon:.3f}")

        print("Training complete.")
        if self.logger:
            self.logger.save_batch()

    def eval(self, steps=100):
        print("inside eval")
        """Train one episode using SARSA."""
        obs = self.env.reset_eval()
        state = self.get_state()
        action = self.choose_action(state, training=False)

        episode_reward = 0
        episode_td_errors = []
        stuck_count = 0

        for step in range(steps):
            # print("step: ", step)
            if getattr(self.env, "_is_gymnasium", False):
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            next_state = self.get_state()
            next_action = self.choose_action(next_state, training=False)
            # print(next_action)

            # Log transition
            if self.logger_eval:
                # print("yes")
                coverage = info.get("explored_fraction", 0) if isinstance(info, dict) else 0
                self.logger_eval.log_transition(state, action, reward, next_state, step, done, coverage)

            # Check termination
            explored_frac = info.get("explored_fraction", 0) if isinstance(info, dict) else 0
            if explored_frac >= self.env.coverage_goal or done:
                break

            state = next_state
            action = next_action
        if self.logger_eval:
            self.logger_eval.increment_episode()

        return episode_reward


# ==================== SARSA AGENT ====================

class SARSAAgent(BaseAgent):
    """SARSA agent with epsilon-greedy exploration and transition logging."""

    def update_q(self, state, action, reward, next_state, next_action):
        """SARSA update rule."""
        current_q = self.Q[state][action]
        next_q = self.Q[next_state][next_action]
        td_error = reward + self.gamma * next_q - current_q
        self.Q[state][action] = current_q + self.alpha * td_error
        return td_error

    def train_episode(self, max_steps=2000, verbose=False):
        """Train one episode using SARSA."""
        obs = self.env.reset() if not getattr(self.env, "_is_gymnasium", False) else self.env.reset()[0]
        state = self.get_state()
        action = self.choose_action(state, training=True)

        episode_reward = 0
        episode_td_errors = []
        stuck_count = 0

        for step in range(max_steps):
            if getattr(self.env, "_is_gymnasium", False):
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, info = self.env.step(action)

            next_state = self.get_state()
            next_action = self.choose_action(next_state, training=True)

            # Log transition
            if self.logger:
                coverage = info.get("explored_fraction", 0) if isinstance(info, dict) else 0
                self.logger.log_transition(state, action, reward, next_state, step, done, coverage)

            # SARSA Q update
            td_error = self.update_q(state, action, reward, next_state, next_action)
            episode_td_errors.append(abs(td_error))

            if next_state[3] > 0:
                stuck_count += 1

            episode_reward += reward

            if verbose and step % 200 == 0:
                print(f"  Step {step}: s={state}, a={action}, r={reward:.2f}, s'={next_state}")

            # Check termination
            explored_frac = info.get("explored_fraction", 0) if isinstance(info, dict) else 0
            if explored_frac >= self.env.coverage_goal or done:
                break

            state = next_state
            action = next_action

        if self.logger:
            self.logger.increment_episode()
        # print("Abhat")
        # print("Stuck count: ",self.env.stuck_count)
        return episode_reward, step + 1, explored_frac, stuck_count

# ==================== Q-LEARNING AGENT ====================

class QLearningAgent(BaseAgent):
    """Q-learning agent with epsilon-greedy exploration and transition logging."""

    def update_q(self, state, action, reward, next_state):
        """Q-learning update rule."""
        current_q = self.Q[state][action]
        max_next_q = np.max(self.Q[next_state])
        td_error = reward + self.gamma * max_next_q - current_q
        self.Q[state][action] = current_q + self.alpha * td_error
        return td_error

    def train_episode(self, max_steps=2000, verbose=False):
        """Train one episode using Q-learning."""
        obs = self.env.reset() if not getattr(self.env, "_is_gymnasium", False) else self.env.reset()[0]
        state = self.get_state()

        episode_reward = 0
        stuck_count = 0

        for step in range(max_steps):
            action = self.choose_action(state, training=True)

            if getattr(self.env, "_is_gymnasium", False):
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, info = self.env.step(action)

            next_state = self.get_state()

            # Log transition
            if self.logger:
                coverage = info.get("explored_fraction", 0) if isinstance(info, dict) else 0
                self.logger.log_transition(state, action, reward, next_state, step, done, coverage)

            # Q-learning update
            td_error = self.update_q(state, action, reward, next_state)

            if next_state[3] > 0:
                stuck_count += 1

            episode_reward += reward

            if verbose and step % 200 == 0:
                print(f"  Step {step}: s={state}, a={action}, r={reward:.2f}, s'={next_state}")

            explored_frac = info.get("explored_fraction", 0) if isinstance(info, dict) else 0
            if explored_frac >= self.env.coverage_goal or done:
                break

            state = next_state

        if self.logger:
            self.logger.increment_episode()

        return episode_reward, step + 1, explored_frac, stuck_count
