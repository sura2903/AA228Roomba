import os
import numpy as np

from setup import RoombaSoftPOMDPEnv
from particle_filter import ParticleFilter, render_particles


def run_demo(
    seed=0,
    N=300,
    steps=200,
    save_every=5,
    target=None,
    target_radius=0.5,
    use_true=False,
):
    env = RoombaSoftPOMDPEnv(
        width=20, height=20, exit_mode="reset", seed=seed, random_obstacles=True
    )
    obs = env.reset()

    pf = ParticleFilter(env, N=N, use_hidden=True)
    pf.init_particles()

    save_dir = os.path.join(os.getcwd(), "render_frames", "pf_run")
    os.makedirs(save_dir, exist_ok=True)

    total = 0.0
    for t in range(steps):
        a = np.random.choice([0, 1, 2, 3], p=[0.5, 0.1, 0.2, 0.2])

        # predict
        pf.predict(a)

        # step env
        step_ret = env.step(a)
        if getattr(env, "_is_gymnasium", False):
            obs, r, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = step_ret

        total += float(r)

        # update PF with observation
        pf.update(obs)

        if t % save_every == 0:
            out = os.path.join(save_dir, f"pf_{t:05d}.png")
            render_particles(env, pf, save_path=out)

        # termination: either environment done or target reached
        if done:
            print("Env done at step", t)
            break

        if target is not None:
            tx, ty = target
            if use_true:
                # check using true state
                true_x, true_y = env.x, env.y
                dist = ((true_x - tx) ** 2 + (true_y - ty) ** 2) ** 0.5
                if dist <= target_radius:
                    print(f"Target reached (true) at step {t}, dist={dist:.3f}")
                    break
            else:
                # check using particle filter estimate
                est = pf.estimate()
                ex, ey = est[0], est[1]
                dist = ((ex - tx) ** 2 + (ey - ty) ** 2) ** 0.5
                if dist <= target_radius:
                    print(f"Target reached (belief) at step {t}, dist={dist:.3f}")
                    break

    print("Total reward", total)


if __name__ == "__main__":
    run_demo(seed=1, N=300, steps=300, save_every=5)
