python main.py --method qlearning --scenario mixed_obstacles_seed42 --episodes 500 --log-transitions --steps 500
python main.py --method sarsa --scenario mixed_obstacles_seed42 --episodes 500 --log-transitions --steps 500 --visit_reward_lambda 0.1
python main.py --method sarsa --scenario mixed_obstacles_seed42 --episodes 500 --log-transitions --steps 500 --visit_reward_lambda 0.15




python main.py --method sarsa --scenario mixed_obstacles_seed42 --episodes 1000 --log-transitions --steps 1000 --w_explore 4 --w_collision 2 --w_soft 1
python main.py --method sarsa --scenario mixed_obstacles_seed42 --episodes 1000 --log-transitions --steps 1000 --w_explore 5 --w_collision 2 --w_soft 2
python main.py --method sarsa --scenario mixed_obstacles_seed42 --episodes 1000 --log-transitions --steps 1000 --w_explore 4 --w_collision 1 --w_soft 1

