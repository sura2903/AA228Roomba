import math
import random
import time
from collections import defaultdict


class POMCPNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = (
            {}
        )  # obs -> child node for belief nodes, or action->node at root
        self.N = 0
        self.Q = 0.0
        self.action_children = {}  # action -> (N, Q, obs_children dict)

    def ucb_score(self, total_visits, action_N, c):
        if action_N == 0:
            return float("inf")
        return (self.action_children[action][1] / action_N) + c * math.sqrt(
            math.log(total_visits + 1) / action_N
        )


class POMCPTree:
    def __init__(
        self,
        action_space_n,
        generative,
        rollout_policy,
        sims=200,
        max_depth=30,
        gamma=0.99,
        c=1.4,
        rng=None,
    ):
        self.action_space_n = action_space_n
        self.generative = generative
        self.rollout_policy = rollout_policy
        self.sims = sims
        self.max_depth = max_depth
        self.gamma = gamma
        self.c = c
        self.root = POMCPNode()
        self.rng = rng or random.Random()

    def reset(self):
        self.root = POMCPNode()

    def search(
        self,
        belief_particles,
        sim_counts_baseline=None,
        total_visitable=None,
        coverage_goal=None,
    ):
        # belief_particles: list of (state, hidden, weight) or just state tuples depending on PF
        for _ in range(self.sims):
            # sample a root particle
            weights = [p[2] for p in belief_particles]
            particle = self.rng.choices(belief_particles, weights=weights, k=1)[0]
            state, hidden, w = particle
            # initialize per-simulation visit counts (copy baseline if provided)
            if sim_counts_baseline is None:
                sim_counts = {}
            else:
                sim_counts = dict(sim_counts_baseline)
            self._simulate(
                state,
                hidden,
                self.root,
                depth=0,
                sim_counts=sim_counts,
                total_visitable=total_visitable,
                coverage_goal=coverage_goal,
            )

        # select best action by highest average Q
        best_a = None
        best_q = -float("inf")
        for a in range(self.action_space_n):
            entry = self.root.action_children.get(a)
            if entry is None:
                # untried action -> prefer it
                return a
            N_a, Q_a, _ = entry
            q = Q_a / N_a if N_a > 0 else -float("inf")
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def _simulate(
        self,
        state,
        hidden,
        node,
        depth,
        sim_counts=None,
        total_visitable=None,
        coverage_goal=None,
    ):
        if depth >= self.max_depth:
            return 0.0

        # Choose action
        total_N = node.N
        # Initialize untried actions lazily
        untried = [
            a for a in range(self.action_space_n) if a not in node.action_children
        ]
        if untried:
            a = self.rng.choice(untried)
            # expand action entry
            node.action_children[a] = [0, 0.0, {}]
        else:
            # UCB over actions
            best_score = -float("inf")
            best_a = None
            for a, (N_a, Q_a, obs_children) in node.action_children.items():
                if N_a == 0:
                    score = float("inf")
                else:
                    score = (Q_a / N_a) + self.c * math.sqrt(
                        math.log(total_N + 1) / N_a
                    )
                if score > best_score:
                    best_score = score
                    best_a = a
            a = best_a

        # simulate generative model; pass sim_counts so generative can compute simulated rewards
        next_state, next_hidden, obs, reward, done = self.generative(
            state, hidden, a, sim_counts, total_visitable, coverage_goal
        )

        # find or create child node for observation under the chosen action
        entry = node.action_children[a]
        N_a, Q_a, obs_children = entry
        if obs not in obs_children:
            obs_children[obs] = POMCPNode(parent=node, action=a)

        # recursive simulate or rollout
        if done:
            R = reward
        else:
            child_node = obs_children[obs]
            R = reward + self.gamma * self._simulate(
                next_state,
                next_hidden,
                child_node,
                depth + 1,
                sim_counts=sim_counts,
                total_visitable=total_visitable,
                coverage_goal=coverage_goal,
            )

        # backpropagate
        node.N += 1
        entry[0] += 1
        entry[1] += R
        return R

    def rollout(self, state, hidden, depth):
        # not used separately; included for future enhancements
        return 0.0
