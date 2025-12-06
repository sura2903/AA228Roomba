python main.py --method qlearning --scenario mixed_obstacles_seed42 --episodes 500 --log-transitions --steps 500
python main.py --method sarsa --scenario mixed_obstacles_seed42 --episodes 500 --log-transitions --steps 500 --visit_reward_lambda 0.1
python main.py --method sarsa --scenario mixed_obstacles_seed42 --episodes 500 --log-transitions --steps 500 --visit_reward_lambda 0.15


