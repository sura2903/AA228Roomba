"""
Standalone script to compare SARSA and Q-Learning results.
Hardcode the paths to your CSV files and run this script.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ========== HARDCODE YOUR PATHS HERE ==========
SARSA_CSV = "four_rooms/sarsa/2025-12-07-00-30-03/transition_train.csv"
QLEARNING_CSV = "four_rooms/qlearning/2025-12-07-00-29-26/transition_train.csv"
OUTPUT_PATH = "four_rooms/comparison_plot.png"
# ===============================================

def plot_comparison(sarsa_csv, qlearning_csv, output_path):
    """
    Create comparison plots for SARSA vs Q-Learning.
    """
    # Load data
    print(f"Loading SARSA data from: {sarsa_csv}")
    if not os.path.exists(sarsa_csv):
        print(f"ERROR: SARSA CSV not found at {sarsa_csv}")
        return
    
    print(f"Loading Q-Learning data from: {qlearning_csv}")
    if not os.path.exists(qlearning_csv):
        print(f"ERROR: Q-Learning CSV not found at {qlearning_csv}")
        return
    
    df_sarsa = pd.read_csv(sarsa_csv)
    df_qlearning = pd.read_csv(qlearning_csv)
    
    print(f"✓ Loaded {len(df_sarsa)} SARSA transitions")
    print(f"✓ Loaded {len(df_qlearning)} Q-Learning transitions")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall reward distribution comparison
    axes[0, 0].hist(df_sarsa['reward'], bins=50, alpha=0.6, edgecolor='black', 
                    color='blue', label='SARSA')
    axes[0, 0].hist(df_qlearning['reward'], bins=50, alpha=0.6, edgecolor='black', 
                    color='red', label='Q-Learning')
    axes[0, 0].axvline(df_sarsa['reward'].mean(), color='blue', linestyle='--', 
                       linewidth=2, label=f'SARSA Mean')
    axes[0, 0].axvline(df_qlearning['reward'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Q-Learning Mean')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Reward by stuck depth - SARSA
    for k in range(4):
        subset = df_sarsa[df_sarsa['s_k'] == k]['reward']
        if len(subset) > 0:
            axes[0, 1].hist(subset, bins=30, alpha=0.5, label=f'SARSA k={k}')
    axes[0, 1].set_xlabel('Reward')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('SARSA: Reward Distribution by Stuck Depth')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reward by action comparison
    action_names = ['Forward', 'Backward', 'TurnLeft', 'TurnRight']
    sarsa_actions = df_sarsa.groupby('action')['reward'].mean()
    qlearning_actions = df_qlearning.groupby('action')['reward'].mean()
    
    x = range(4)
    width = 0.35
    axes[1, 0].bar([i - width/2 for i in x], 
                   [sarsa_actions.get(i, 0) for i in range(4)], 
                   width, label='SARSA', alpha=0.8, color='blue')
    axes[1, 0].bar([i + width/2 for i in x], 
                   [qlearning_actions.get(i, 0) for i in range(4)], 
                   width, label='Q-Learning', alpha=0.8, color='red')
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Mean Reward')
    axes[1, 0].set_title('Mean Reward by Action')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(action_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Reward over episodes - OVERLAID PLOT (THE MAIN ONE!)
    episode_rewards_sarsa = df_sarsa.groupby('episode')['reward'].mean()
    episode_rewards_qlearning = df_qlearning.groupby('episode')['reward'].mean()
    
    window = min(50, len(episode_rewards_sarsa) // 10)
    
    # Plot SARSA
    if window > 1:
        smoothed_sarsa = episode_rewards_sarsa.rolling(window=window).mean()
        axes[1, 1].plot(smoothed_sarsa, label='SARSA (Smoothed)', 
                       linewidth=2.5, color='blue', alpha=0.9)
    axes[1, 1].plot(episode_rewards_sarsa, alpha=0.15, linewidth=0.8, color='blue')
    
    # Plot Q-Learning
    window_q = min(50, len(episode_rewards_qlearning) // 10)
    if window_q > 1:
        smoothed_qlearning = episode_rewards_qlearning.rolling(window=window_q).mean()
        axes[1, 1].plot(smoothed_qlearning, label='Q-Learning (Smoothed)', 
                       linewidth=2.5, color='red', alpha=0.9)
    axes[1, 1].plot(episode_rewards_qlearning, alpha=0.15, linewidth=0.8, color='red')
    
    axes[1, 1].set_xlabel('Episode', fontsize=11)
    axes[1, 1].set_ylabel('Mean Reward per Step', fontsize=11)
    axes[1, 1].set_title('Average Reward per Episode - SARSA vs Q-Learning', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)
    print(f"\nSARSA:")
    print(f"  Episodes: {df_sarsa['episode'].max() + 1}")
    print(f"  Mean Reward: {df_sarsa['reward'].mean():.4f}")
    print(f"  Std Reward: {df_sarsa['reward'].std():.4f}")
    print(f"  Final Episode Mean Reward: {episode_rewards_sarsa.iloc[-1]:.4f}")
    
    print(f"\nQ-Learning:")
    print(f"  Episodes: {df_qlearning['episode'].max() + 1}")
    print(f"  Mean Reward: {df_qlearning['reward'].mean():.4f}")
    print(f"  Std Reward: {df_qlearning['reward'].std():.4f}")
    print(f"  Final Episode Mean Reward: {episode_rewards_qlearning.iloc[-1]:.4f}")
    print("="*60)


if __name__ == "__main__":
    plot_comparison(SARSA_CSV, QLEARNING_CSV, OUTPUT_PATH)