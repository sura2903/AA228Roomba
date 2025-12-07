import numpy as np
import pandas as pd
import csv
import os
import json

from agents import SARSAAgent, QLearningAgent
from setup import RoombaSoftPOMDPEnv
from generate_test_scenarios import load_scenario
from plot_heatmap import save_heatmaps

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
        
        # plt.show()
    
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


class ScenarioTester:

    def __init__(self, scenarios_dir="./test_scenarios", output_dir="./scenario_results"):
        self.scenarios_dir = scenarios_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def test(
        self,
        scenario_name,
        method="sarsa",
        n_train_episodes=500,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        log_transitions=True,
        save_dir="./",
        steps=2000,
        visit_reward_lambda=0.07,
        w_explore = 1.0,
        w_collision = 1.0,
        w_stuck = 1.0,
        eval_steps = 1000
    ):
        
        log_filepath = os.path.join(save_dir, "transition_train.csv")
        log_filepath_eval = os.path.join(save_dir,"transition_eval.csv")

        # load scenario
        map_array, info = load_scenario(scenario_name, self.scenarios_dir)

        from agents import SARSAAgent, QLearningAgent   # lazy import avoids circular

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
            seed=42,
            visit_reward_lambda = visit_reward_lambda,
            w_explore=w_explore, w_stuck=w_stuck, w_collision=w_collision
        )

        # choose method
        if method == "sarsa":
            agent = SARSAAgent(env,
                               alpha=alpha, gamma=gamma,
                               epsilon=epsilon, epsilon_decay=epsilon_decay,
                               epsilon_min=epsilon_min,
                               log_transitions=log_transitions,
                               log_filepath=log_filepath,
                               log_filepath_eval = log_filepath_eval)
        else:
            agent = QLearningAgent(env,
                                   alpha=alpha, gamma=gamma,
                                   epsilon=epsilon, epsilon_decay=epsilon_decay,
                                   epsilon_min=epsilon_min,
                                   log_transitions=log_transitions,
                                   log_filepath=log_filepath,
                                   log_filepath_eval = log_filepath_eval)

        agent.train(n_episodes=n_train_episodes, eval_every=100, verbose=False, steps=steps)

        # states = sorted(agent.Q.keys())
        # q_array = np.vstack([agent.Q[s] for s in states])
        # np.save(os.path.join(save_dir,"q_table.npy"), q_array)
        qtable = {str(state):list(agent.Q[state]) for state in agent.Q}
        json_path = os.path.join(save_dir,"q_table.json")
        with open(json_path, 'w') as f:
            json.dump(qtable, f)


        episode_reward = agent.eval(steps=eval_steps)
        print("FINAL REWARD: ", episode_reward)

        
        # Run analyzer afterward
        if log_transitions:
            analyzer = TransitionAnalyzer(log_filepath)
            analyzer.summary_statistics()
            analysis_path = os.path.join(save_dir, 'analysis_train.txt')
            plot_path = os.path.join(save_dir, 'plots_rewards_train.png')
            analyzer.export_summary(analysis_path)
            analyzer.plot_reward_distribution(save_path=plot_path)

            analyzer = TransitionAnalyzer(log_filepath_eval)
            analyzer.summary_statistics()
            save_heatmaps(log_filepath_eval, save_dir, scenario_name)

        return agent