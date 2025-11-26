import os
import numpy as np
from particle_filter import ParticleFilter
from pf_heatmap import heatmap_from_particles
from setup import RoombaSoftPOMDPEnv


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p.flatten()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return -np.sum(p * np.log(p))


def run_diagnostics(seed=1, N=300, steps=200, save_every=50):
    env = RoombaSoftPOMDPEnv(
        width=20, height=20, exit_mode="reset", seed=seed, random_obstacles=True
    )
    obs = env.reset()

    pf = ParticleFilter(env, N=N, use_hidden=True)
    pf.init_particles()

    print(f"env.obs_sigma={env.obs_sigma}, theta_obs_sigma={env.theta_obs_sigma}")
    print("step, ESS, entropy, max_prob, obs_x, obs_y, est_x, est_y")

    for t in range(steps):
        a = np.random.choice([0, 1, 2, 3], p=[0.5, 0.1, 0.2, 0.2])
        pf.predict(a)
        step_ret = env.step(a)
        if getattr(env, "_is_gymnasium", False):
            obs, r, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = step_ret

        pf.update(obs)

        # diagnostics
        ess = 1.0 / np.sum(pf.ws**2)
        heat = heatmap_from_particles(env, pf)
        ent = entropy(heat)
        maxp = float(np.max(heat))
        est = pf.estimate()
        print(
            f"{t:03d}, {ess:.1f}, {ent:.4f}, {maxp:.6f}, {obs[0]:.2f},{obs[1]:.2f}, {est[0]:.2f},{est[1]:.2f}"
        )

        if (t % save_every) == 0:
            # save a small diagnostic image
            out_dir = os.path.join(os.getcwd(), "render_frames", "pf_diag")
            os.makedirs(out_dir, exist_ok=True)
            # reuse existing rendering if available
            try:
                from particle_filter import render_particles

                render_particles(
                    env, pf, save_path=os.path.join(out_dir, f"diag_{t:03d}.png")
                )
            except Exception:
                pass

        if done:
            print("Env terminated at step", t)
            break


if __name__ == "__main__":
    run_diagnostics(seed=1, N=300, steps=200, save_every=50)
