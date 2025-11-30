"""
SARSA with CSV Transition Logging
Each row: state(x,y,theta,k), action, reward, next_state(x',y',theta',k')
"""

import numpy as np
import pandas as pd
import csv
import os
from collections import defaultdict
from setup import RoombaSoftPOMDPEnv
from generate_test_scenarios import load_scenario, generate_all_scenarios

# ==================== TRANSITION LOGGER ====================

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
            's_x', 's_y', 's_theta', 's_k',  # Current state
            'action',                         # Action taken
            'reward',                         # Immediate reward
            'sp_x', 'sp_y', 'sp_theta', 'sp_k',  # Next state
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
            next_state: (x', y', theta', k')
            step: int - step number in episode
            done: bool - episode terminated
            coverage: float - fraction of map covered
        """
        transition = [
            state[0], state[1], state[2], state[3],     # s
            action,                                      # a
            reward,                                      # r
            next_state[0], next_state[1], next_state[2], next_state[3],  # s'
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


# ==================== SARSA AGENT WITH LOGGING ====================

class SARSAAgent:    
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        log_transitions=True,
        log_filepath="transitions.csv"
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.n_actions = env.action_space.n
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.coverage_history = []
        self.td_errors = []
        self.stuck_encounters = []
        
        # Transition logger
        self.log_transitions = log_transitions
        if log_transitions:
            self.logger = TransitionLogger(log_filepath)
        else:
            self.logger = None
    
    def get_state(self):
        """Get current state tuple."""
        return (self.env.x, self.env.y, self.env.theta, self.env.k)
    
    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update_q(self, state, action, reward, next_state, next_action):
        """SARSA update."""
        current_q = self.Q[state][action]
        next_q = self.Q[next_state][next_action]
        td_error = reward + self.gamma * next_q - current_q
        self.Q[state][action] = current_q + self.alpha * td_error
        return td_error
    
    def train_episode(self, max_steps=2000, verbose=False):
        """Train one episode with transition logging."""
        # Reset
        if getattr(self.env, '_is_gymnasium', False):
            obs, info = self.env.reset()
        else:
            obs = self.env.reset()
        
        state = self.get_state()
        action = self.choose_action(state, training=True)
        
        episode_reward = 0
        episode_td_errors = []
        stuck_count = 0
        
        for step in range(max_steps):
            # Execute action
            if getattr(self.env, '_is_gymnasium', False):
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, info = self.env.step(action)
            
            next_state = self.get_state()
            next_action = self.choose_action(next_state, training=True)
            
            # Log transition
            if self.log_transitions:
                coverage = info.get('explored_fraction', 0) if isinstance(info, dict) else 0
                self.logger.log_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    step=step,
                    done=done,
                    coverage=coverage
                )
            
            # SARSA update
            td_error = self.update_q(state, action, reward, next_state, next_action)
            episode_td_errors.append(abs(td_error))
            
            if next_state[3] > 0:
                stuck_count += 1
            
            episode_reward += reward
            
            if verbose and step % 200 == 0:
                print(f"  Step {step}: s={state}, a={action}, r={reward:.2f}, s'={next_state}")
            
            # Check termination
            explored_frac = info.get('explored_fraction', 0) if isinstance(info, dict) else 0
            if explored_frac >= self.env.coverage_goal or done:
                break
            
            state = next_state
            action = next_action
        
        # End of episode
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if self.log_transitions:
            self.logger.increment_episode()
        
        # Statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(step + 1)
        self.coverage_history.append(explored_frac)
        self.td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)
        self.stuck_encounters.append(stuck_count)
        
        return episode_reward, step + 1, explored_frac, stuck_count
    
    def train(self, n_episodes=2000, eval_every=100, verbose=True):
        """Train for multiple episodes."""
        print(f"{'='*80}")
        print(f"SARSA TRAINING WITH TRANSITION LOGGING")
        print(f"{'='*80}")
        print(f"Episodes: {n_episodes}")
        print(f"Logging: {self.logger.filepath if self.log_transitions else 'Disabled'}")
        print()
        
        for episode in range(n_episodes):
            ep_reward, ep_length, coverage, stuck = self.train_episode(
                verbose=(verbose and episode % 200 == 0)
            )
            
            if episode % eval_every == 0:
                window = min(eval_every, len(self.episode_rewards))
                avg_reward = np.mean(self.episode_rewards[-window:])
                avg_length = np.mean(self.episode_lengths[-window:])
                avg_coverage = np.mean(self.coverage_history[-window:])
                
                print(f"Episode {episode:4d}/{n_episodes}")
                print(f"  Reward:   {avg_reward:8.2f}")
                print(f"  Length:   {avg_length:8.0f}")
                print(f"  Coverage: {avg_coverage:8.3f}")
                print(f"  Epsilon:  {self.epsilon:8.4f}")
                print(f"  Q-states: {len(self.Q):8d}")
                if self.log_transitions:
                    print(f"  Logged:   {self.logger.episode_num} episodes")
                print()
        
        print("✓ Training complete!")
        
        if self.log_transitions:
            self.logger.save_batch()
            print(f"✓ Transitions saved to: {self.logger.filepath}")
        
        return self.episode_rewards, self.episode_lengths, self.coverage_history


# ==================== TRANSITION ANALYZER ====================

class TransitionAnalyzer:
    """
    Analyze logged transitions to understand reward function behavior.
    """
    
    def __init__(self, csv_path="transitions.csv"):
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} transitions from {csv_path}")
    
    def summary_statistics(self):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"TRANSITION DATA SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Total transitions: {len(self.df)}")
        print(f"Episodes: {self.df['episode'].max() + 1}")
        print(f"Unique states visited: {len(self.df[['s_x', 's_y', 's_theta', 's_k']].drop_duplicates())}")
        print()
        
        print(f"Reward Statistics:")
        print(f"  Mean:   {self.df['reward'].mean():8.3f}")
        print(f"  Std:    {self.df['reward'].std():8.3f}")
        print(f"  Min:    {self.df['reward'].min():8.3f}")
        print(f"  Max:    {self.df['reward'].max():8.3f}")
        print(f"  Median: {self.df['reward'].median():8.3f}")
        print()
        
        print(f"Action Distribution:")
        for action in range(4):
            count = len(self.df[self.df['action'] == action])
            pct = 100 * count / len(self.df)
            action_names = ['Forward', 'Backward', 'TurnLeft', 'TurnRight']
            print(f"  {action_names[action]:<10}: {count:6d} ({pct:5.1f}%)")
        print()
        
        print(f"Stuck Depth Distribution:")
        for k in range(4):
            count = len(self.df[self.df['s_k'] == k])
            pct = 100 * count / len(self.df)
            print(f"  k={k}: {count:6d} ({pct:5.1f}%)")
        print()
    
    def reward_by_stuck_depth(self):
        """Analyze rewards by stuck depth."""
        print(f"\n{'='*80}")
        print(f"REWARD BY STUCK DEPTH")
        print(f"{'='*80}\n")
        
        for k in range(4):
            subset = self.df[self.df['s_k'] == k]
            if len(subset) > 0:
                print(f"k={k} (n={len(subset)}):")
                print(f"  Mean reward: {subset['reward'].mean():7.3f}")
                print(f"  Std reward:  {subset['reward'].std():7.3f}")
                print()
    
    def reward_by_action(self):
        """Analyze rewards by action type."""
        print(f"\n{'='*80}")
        print(f"REWARD BY ACTION")
        print(f"{'='*80}\n")
        
        action_names = ['Forward', 'Backward', 'TurnLeft', 'TurnRight']
        for action in range(4):
            subset = self.df[self.df['action'] == action]
            if len(subset) > 0:
                print(f"{action_names[action]} (n={len(subset)}):")
                print(f"  Mean reward: {subset['reward'].mean():7.3f}")
                print(f"  Std reward:  {subset['reward'].std():7.3f}")
                print()
    
    def analyze_stuck_transitions(self):
        """Analyze transitions involving stuck states."""
        print(f"\n{'='*80}")
        print(f"STUCK STATE TRANSITIONS")
        print(f"{'='*80}\n")
        
        # Entering stuck state (k: 0 -> 1+)
        entering = self.df[(self.df['s_k'] == 0) & (self.df['sp_k'] > 0)]
        print(f"Entering stuck state (k: 0→1+): {len(entering)} transitions")
        if len(entering) > 0:
            print(f"  Mean reward: {entering['reward'].mean():.3f}")
            print()
        
        # Exiting stuck state (k: 1+ -> 0)
        exiting = self.df[(self.df['s_k'] > 0) & (self.df['sp_k'] == 0)]
        print(f"Exiting stuck state (k: 1+→0): {len(exiting)} transitions")
        if len(exiting) > 0:
            print(f"  Mean reward: {exiting['reward'].mean():.3f}")
            print()
        
        # Getting more stuck (k increases)
        worsening = self.df[self.df['sp_k'] > self.df['s_k']]
        print(f"Getting more stuck (k increases): {len(worsening)} transitions")
        if len(worsening) > 0:
            print(f"  Mean reward: {worsening['reward'].mean():.3f}")
            print()
    
    def plot_reward_distribution(self, save_path=None):
        """Plot reward distribution."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall reward distribution
        axes[0, 0].hist(self.df['reward'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.df['reward'].mean(), color='r', linestyle='--', label='Mean')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Reward Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reward by stuck depth
        for k in range(4):
            subset = self.df[self.df['s_k'] == k]['reward']
            if len(subset) > 0:
                axes[0, 1].hist(subset, bins=30, alpha=0.5, label=f'k={k}')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Reward Distribution by Stuck Depth')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Reward by action
        action_names = ['Forward', 'Backward', 'TurnLeft', 'TurnRight']
        for action in range(4):
            subset = self.df[self.df['action'] == action]['reward']
            if len(subset) > 0:
                axes[1, 0].hist(subset, bins=30, alpha=0.5, label=action_names[action])
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reward Distribution by Action')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Reward over episodes (smoothed)
        episode_rewards = self.df.groupby('episode')['reward'].mean()
        window = min(50, len(episode_rewards) // 10)
        if window > 1:
            smoothed = episode_rewards.rolling(window=window).mean()
            axes[1, 1].plot(smoothed, label='Smoothed', linewidth=2)
        axes[1, 1].plot(episode_rewards, alpha=0.3, label='Raw')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Mean Reward per Step')
        axes[1, 1].set_title('Average Reward per Episode')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved reward distribution plot to {save_path}")
        
        plt.show()
    
    def export_summary(self, output_path="transition_analysis.txt"):
        """Export analysis to text file."""
        import sys
        
        original_stdout = sys.stdout
        with open(output_path, 'w') as f:
            sys.stdout = f
            
            self.summary_statistics()
            self.reward_by_stuck_depth()
            self.reward_by_action()
            # self.analyze_stuck_transitions()
            
            sys.stdout = original_stdout
        
        print(f"✓ Exported analysis to {output_path}")
    
    def get_state_action_rewards(self):
        """Get average reward for each state-action pair."""
        state_action_rewards = self.df.groupby(
            ['s_x', 's_y', 's_theta', 's_k', 'action']
        )['reward'].agg(['mean', 'count', 'std']).reset_index()
        
        return state_action_rewards


# ==================== SCENARIO TESTER WITH CSV ====================

class ScenarioTesterCSV:
    """Test SARSA on scenarios with CSV logging."""
    
    def __init__(self, scenarios_dir="./test_scenarios", output_dir="./scenario_results"):
        self.scenarios_dir = scenarios_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def test_scenario(
        self,
        scenario_name,
        n_train_episodes=500,
        n_eval_episodes=5,
        alpha=0.1,
        gamma=0.95
    ):
        """Test SARSA on a scenario with CSV logging."""
        print(f"\n{'='*80}")
        print(f"TESTING: {scenario_name}")
        print(f"{'='*80}\n")
        
        # Load scenario
        try:
            map_array, info = load_scenario(scenario_name, self.scenarios_dir)
        except FileNotFoundError:
            print(f"✗ Scenario not found: {scenario_name}")
            return None
        
        print(f"Map: {info['width']}×{info['height']}")
        print(f"Description: {info['description']}\n")
        
        # Create environment
        env = RoombaSoftPOMDPEnv(
            width=info['width'],
            height=info['height'],
            map_array=map_array,
            K=3,
            exit_mode="reset",
            p_intended=0.7,
            p_adj=0.1,
            p_stay=0.1,
            p_exit_base=0.9,
            p_exit_alpha=0.5,
            coverage_goal=0.85,
            max_steps=7000,
            seed=42
        )
        
        # Output paths
        csv_path = os.path.join(self.output_dir, f"{scenario_name}_transitions.csv")
        analysis_path = os.path.join(self.output_dir, f"{scenario_name}_analysis.txt")
        plot_path = os.path.join(self.output_dir, f"{scenario_name}_rewards.png")
        
        # Create agent with logging
        agent = SARSAAgent(
            env,
            alpha=alpha,
            gamma=gamma,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            log_transitions=True,
            log_filepath=csv_path
        )
        
        # Train
        agent.train(n_episodes=n_train_episodes, eval_every=100, verbose=False)
        
        # Analyze transitions
        print(f"\nAnalyzing transitions...")
        analyzer = TransitionAnalyzer(csv_path)
        analyzer.summary_statistics()
        analyzer.reward_by_stuck_depth()
        analyzer.reward_by_action()
        # analyzer.analyze_stuck_transitions()
        analyzer.export_summary(analysis_path)
        analyzer.plot_reward_distribution(save_path=plot_path)
        
        print(f"\n✓ Results saved to {self.output_dir}/")
        print(f"  - {scenario_name}_transitions.csv")
        print(f"  - {scenario_name}_analysis.txt")
        print(f"  - {scenario_name}_rewards.png")
        
        return {
            'scenario': scenario_name,
            'csv_path': csv_path,
            'analysis_path': analysis_path,
            'plot_path': plot_path,
            'agent': agent,
            'analyzer': analyzer
        }


# ==================== MAIN ====================

def main():
    """Main script with CSV logging."""
    print(f"\n{'#'*80}")
    print(f"# SARSA WITH CSV TRANSITION LOGGING")
    print(f"# Format: s_x, s_y, s_theta, s_k, action, reward, sp_x, sp_y, sp_theta, sp_k")
    print(f"{'#'*80}\n")
    
    # Ensure scenarios exist
    if not os.path.exists("./test_scenarios/summary.json"):
        print("Generating test scenarios...")
        generate_all_scenarios()
    
    # Test single scenario with detailed logging
    tester = ScenarioTesterCSV()
    
    result = tester.test_scenario(
        "single_soft_center",
        n_train_episodes=500,
        n_eval_episodes=5
    )
    
    if result:
        print(f"\n{'='*80}")
        print(f"CSV FILE PREVIEW")
        print(f"{'='*80}\n")
        
        # Show first few rows
        df = pd.read_csv(result['csv_path'])
        print(df.head(10).to_string(index=False))
        print(f"\n... ({len(df)} total rows)")
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE TRANSITIONS")
        print(f"{'='*80}\n")
        
        # Show interesting transitions
        print("1. Entering stuck state:")
        entering = df[(df['s_k'] == 0) & (df['sp_k'] > 0)].head(3)
        print(entering[['s_x', 's_y', 's_theta', 's_k', 'action', 'reward', 
                       'sp_x', 'sp_y', 'sp_theta', 'sp_k']].to_string(index=False))
        
        print("\n2. Exiting stuck state:")
        exiting = df[(df['s_k'] > 0) & (df['sp_k'] == 0)].head(3)
        print(exiting[['s_x', 's_y', 's_theta', 's_k', 'action', 'reward',
                      'sp_x', 'sp_y', 'sp_theta', 'sp_k']].to_string(index=False))
        
        print("\n3. High reward transitions:")
        high_reward = df.nlargest(3, 'reward')
        print(high_reward[['s_x', 's_y', 's_theta', 's_k', 'action', 'reward',
                          'sp_x', 'sp_y', 'sp_theta', 'sp_k']].to_string(index=False))
        
        print("\n4. Low reward transitions:")
        low_reward = df.nsmallest(3, 'reward')
        print(low_reward[['s_x', 's_y', 's_theta', 's_k', 'action', 'reward',
                         'sp_x', 'sp_y', 'sp_theta', 'sp_k']].to_string(index=False))
    
    # Test multiple scenarios
    print(f"\n{'='*80}")
    print(f"TEST MULTIPLE SCENARIOS")
    print(f"{'='*80}\n")
    
    response = input("Test all scenarios? (y/n): ")
    if response.lower() == 'y':
        scenarios = ['scattered_soft_seed123', 'mixed_obstacles_seed42', 
                    'four_rooms']
        
        for scenario in scenarios:
            tester.test_scenario(scenario, n_train_episodes=800)
    
    print(f"\n{'#'*80}")
    print(f"# COMPLETE!")
    print(f"# CSV files contain full transition history: (s, a, r, s')")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()