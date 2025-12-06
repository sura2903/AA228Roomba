import argparse
import os
from rltools import ScenarioTester
from datetime import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--method", choices=["sarsa", "qlearning"], default="sarsa")
    parser.add_argument("-s", "--scenario", default="single_soft_center")
    parser.add_argument("-e", "--episodes", type=int, default=500)
    parser.add_argument("-step", "--steps", type=int, default=2000)


    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)

    parser.add_argument("--visit_reward_lambda", type=float, default=0.07)


    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)

    parser.add_argument("--log-transitions", action="store_true")
    parser.add_argument("--log-folder", type=str, default="./scenario_results")
    

    return parser.parse_args()


def main():
    args = parse_args()

    tester = ScenarioTester()

    log_filepath = os.path.join(args.log_folder, args.scenario, args.method)
    os.makedirs(log_filepath, exist_ok=True)
    file_prefix = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(log_filepath, file_prefix)
    os.makedirs(save_dir, exist_ok=True)

    agent = tester.test(
        scenario_name=args.scenario,
        method=args.method,
        n_train_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        log_transitions=args.log_transitions,
        save_dir=save_dir,
        steps=args.steps,
        visit_reward_lambda = args.visit_reward_lambda
    )

    # Save h params
    with open(os.path.join(save_dir,"args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")



if __name__ == "__main__":
    main()
