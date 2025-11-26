import os
import time
import numpy as np
from particle_filter import ParticleFilter, render_particles
from pf_heatmap import make_video_from_dir
from pomcp import POMCPTree
from setup import RoombaSoftPOMDPEnv


def generative_from_env(env):
    """Return a generative model closure compatible with POMCPTree.generative.

    The generated function has signature (state, hidden, action) -> (next_state, next_hidden, obs_key, reward, done)
    where obs_key is a hashable observation (we'll use a rounded-tuple) suitable for dict keys.
    """

    def gen(state, hidden, action):
        (nx, ny, ntheta), (nd, nk) = env.sample_transition(state, hidden, int(action))
        # synthesize noisy observation according to env noise params
        x_obs = float(nx) + np.random.normal(scale=env.obs_sigma)
        y_obs = float(ny) + np.random.normal(scale=env.obs_sigma)
        th_obs = float(ntheta) + np.random.normal(scale=env.theta_obs_sigma)
        # clamp
        x_obs = float(np.clip(x_obs, 0.0, float(env.W - 1)))
        y_obs = float(np.clip(y_obs, 0.0, float(env.H - 1)))
        th_obs = float(np.clip(th_obs, 0.0, 3.0))
        # use rounded observation as key to avoid floating point dict issues
        obs_key = (round(x_obs, 2), round(y_obs, 2), round(th_obs, 2))
        # reward is computed using env.reward_model (prev state -> next state)
        prev_state = state
        prev_hidden = hidden
        next_state = (int(nx), int(ny), int(ntheta) % 4)
        next_hidden = (int(nd), int(nk))
        reward = env.reward_model(
            prev_state, prev_hidden, action, next_state, next_hidden
        )
        # done flag: we won't end rollouts early by coverage, only by depth or explicit terminal cells
        done = False
        return next_state, next_hidden, obs_key, float(reward), done

    return gen


def pf_particles_as_belief(pf: ParticleFilter):
    # return list of (state, hidden, weight)
    N = pf.N
    particles = []
    for i in range(N):
        s = (int(pf.xs[i]), int(pf.ys[i]), int(pf.thetas[i]))
        h = (int(pf.ds[i]), int(pf.ks[i]))
        w = float(pf.ws[i])
        particles.append((s, h, w))
    # ensure weights normalized
    total = sum(p[2] for p in particles)
    if total > 0:
        particles = [(s, h, w / total) for (s, h, w) in particles]
    else:
        N = len(particles)
        particles = [(s, h, 1.0 / float(N)) for (s, h, w) in particles]
    return particles


def run_pomcp_demo(
    width=20,
    height=20,
    seed=0,
    N_particles=500,
    sims=500,
    max_steps=10000,
    save_every=50,
    save_dir=None,
):
    env = RoombaSoftPOMDPEnv(
        width=width,
        height=height,
        seed=seed,
        random_obstacles=True,
        max_steps=max_steps,
    )
    obs = env.reset()

    # particle filter
    pf = ParticleFilter(env, N=N_particles, use_hidden=True)
    pf.init_particles()

    # POMCP
    gen = generative_from_env(env)

    # rollout policy: random
    def rollout_policy(state, hidden):
        return np.random.randint(0, env.action_space.n)

    tree = POMCPTree(
        action_space_n=env.action_space.n,
        generative=gen,
        rollout_policy=rollout_policy,
        sims=sims,
        max_depth=30,
        gamma=0.99,
        c=1.4,
    )

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "render_frames", "pomcp_run")
    os.makedirs(save_dir, exist_ok=True)

    total = 0.0
    for t in range(max_steps):
        # construct belief list from PF
        belief = pf_particles_as_belief(pf)

        # run search
        action = tree.search(belief)

        # execute action in env
        step_ret = env.step(action)
        if getattr(env, "_is_gymnasium", False):
            obs, r, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = step_ret

        total += r

        # PF update: propagate then weight
        pf.predict(action)
        pf.update(obs)

        # save frame occasionally
        if t % save_every == 0:
            path = os.path.join(save_dir, f"frame_{t:06d}.png")
            render_particles(env, pf, save_path=path, show_visits=True)

        # rebuild tree root each step (no reuse)
        tree.reset()

        # termination
        explored = info.get("explored_fraction") if isinstance(info, dict) else None
        if explored is not None and explored >= env.coverage_goal:
            print(f"coverage goal reached at step {t}: explored={explored:.4f}")
            break
        if done:
            print("env done at", t)
            break

    print("total reward", total)

    # attempt to stitch saved frames into a video
    try:
        outvid = make_video_from_dir(
            save_dir, output_path=os.path.join(save_dir, "pomcp_run_anim.mp4"), fps=8
        )
        print("Saved POMCP run video:", outvid)
    except Exception as e:
        print("Could not create POMCP video:", e)


if __name__ == "__main__":
    run_pomcp_demo(
        width=20,
        height=20,
        seed=0,
        N_particles=200,
        sims=200,
        max_steps=2000,
        save_every=50,
    )
