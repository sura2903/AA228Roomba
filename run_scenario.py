from generate_test_scenarios import load_scenario
from setup import RoombaSoftPOMDPEnv, DIR2DELTA
import os
import numpy as np

random_runs = 1000

# Parameters
scenario = "four_rooms"
steps = 1000
alpha = 0.1
gamma = 0.95
visit_reward_lambda = 0.07
w_explore = 4.0
w_stuck = 1.0
w_collision = 2.0
eval_steps = 3000
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Load scenario
map_array, info = load_scenario(scenario)

# Create environment with specified parameters
env = RoombaSoftPOMDPEnv(
    width=info["width"],
    height=info["height"],
    map_array=map_array,
    visit_reward_lambda=visit_reward_lambda,
    w_explore=w_explore,
    w_stuck=w_stuck,
    w_collision=w_collision,
    max_steps=steps,
)


# Store per-run metrics for mean and std calculation
rewards = []
hard_collisions_list = []
soft_collisions_list = []
coverage_list = []
entropy_list = []

metrics = {
    "scenario": scenario,
    "steps": steps,
    "visit_reward_lambda": visit_reward_lambda,
    "w_explore": w_explore,
    "w_stuck": w_stuck,
    "w_collision": w_collision,
    "gamma": gamma,
    "alpha": alpha,
    "epsilon": epsilon,
    "epsilon_decay": epsilon_decay,
    "epsilon_min": epsilon_min,
}

obs = env.reset()
total_reward = 0.0
visited = np.zeros_like(env.map, dtype=bool)
results_dir = os.path.join(os.getcwd(), "random_policy_results", scenario)
os.makedirs(results_dir, exist_ok=True)
hard_collisions = 0
soft_collisions = 0


for i in range(random_runs):
    obs = env.reset()
    total_reward = 0.0
    visited = np.zeros_like(env.map, dtype=bool)
    hard_collisions = 0
    soft_collisions = 0

    for t in range(steps):
        prev_x, prev_y = env.x, env.y
        action = env.action_space.sample()
        step_ret = env.step(action)
        if getattr(env, "_is_gymnasium", False):
            obs, r, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = step_ret
        total_reward += r
        visited[env.y, env.x] = True
        # Check for collisions
        if (env.x == prev_x and env.y == prev_y) and action in (0, 1):
            dx, dy = 0, 0
            if action == 0:
                dx, dy = DIR2DELTA[env.theta]
            elif action == 1:
                dx, dy = DIR2DELTA[(env.theta + 2) % 4]
            candx, candy = prev_x + dx, prev_y + dy
            if env.in_bounds(candx, candy):
                if env.map[candy, candx] == 1:
                    hard_collisions += 1
                elif env.map[candy, candx] == 2:
                    soft_collisions += 1
        if done:
            break

    # Calculate coverage as fraction of non-obstacle cells visited
    non_obstacle = env.map != 1
    coverage = float(np.sum(visited & non_obstacle)) / float(np.sum(non_obstacle))

    # Calculate entropy of visit distribution over non-obstacle cells
    visit_counts = visited & non_obstacle
    # For entropy, treat each visited cell as 1, others as 0, normalize over non-obstacle cells
    prob = (
        visit_counts.astype(float) / np.sum(visit_counts)
        if np.sum(visit_counts) > 0
        else np.zeros_like(visit_counts, dtype=float)
    )
    # Only consider non-obstacle cells
    prob_flat = prob[non_obstacle]
    # Avoid log(0) by masking zeros
    prob_nonzero = prob_flat[prob_flat > 0]
    entropy = (
        -np.sum(prob_nonzero * np.log(prob_nonzero)) if prob_nonzero.size > 0 else 0.0
    )

    rewards.append(total_reward)
    hard_collisions_list.append(hard_collisions)
    soft_collisions_list.append(soft_collisions)
    coverage_list.append(coverage)
    entropy_list.append(entropy)


# Compute mean and std for each metric
metrics["mean_reward"] = float(np.mean(rewards))
metrics["std_reward"] = float(np.std(rewards))
metrics["mean_hard_collisions"] = float(np.mean(hard_collisions_list))
metrics["std_hard_collisions"] = float(np.std(hard_collisions_list))
metrics["mean_soft_collisions"] = float(np.mean(soft_collisions_list))
metrics["std_soft_collisions"] = float(np.std(soft_collisions_list))
metrics["mean_coverage"] = float(np.mean(coverage_list))
metrics["std_coverage"] = float(np.std(coverage_list))
metrics["mean_entropy"] = float(np.mean(entropy_list))
metrics["std_entropy"] = float(np.std(entropy_list))

import json

with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("=== Random Policy Results ===")
for k, v in metrics.items():
    print(f"{k}: {v}")
