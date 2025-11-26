import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import colors

from setup import RoombaSoftPOMDPEnv
from particle_filter import ParticleFilter


def heatmap_from_particles(env, pf: ParticleFilter):
    """Aggregate particle weights onto the grid to form a belief heatmap.

    Returns an array shape (H, W) with belief mass per cell.
    """
    H, W = env.H, env.W
    heat = np.zeros((H, W), dtype=float)
    # particles have xs, ys, ws
    xs = np.asarray(pf.xs, dtype=int)
    ys = np.asarray(pf.ys, dtype=int)
    ws = np.asarray(pf.ws, dtype=float)
    # accumulate weights (ignore out-of-bound, but they should not occur)
    for x, y, w in zip(xs, ys, ws):
        if 0 <= x < W and 0 <= y < H:
            heat[y, x] += w
    # normalize
    s = heat.sum()
    if s > 0:
        heat = heat / s
    return heat


def render_heatmap(env, heatmap, save_path=None, figsize=(6, 6), show=False):
    """Render heatmap overlaid on the map and optionally save."""
    cmap_map = colors.ListedColormap(
        ["#ffffff", "#444444", "#ffcc99"]
    )  # free, hard, soft
    bounds = [0, 1, 2, 3]
    norm_map = colors.BoundaryNorm(bounds, cmap_map.N)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    # base map
    ax.imshow(env.map, origin="lower", cmap=cmap_map, norm=norm_map)

    # heatmap overlay (alpha): use a sequential colormap
    hm = ax.imshow(
        heatmap,
        origin="lower",
        cmap="magma",
        alpha=0.8,
        interpolation="nearest",
        vmin=0.0,
        vmax=np.max(heatmap) if np.max(heatmap) > 0 else 1.0,
    )
    plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)

    # true robot position
    if hasattr(env, "x") and env.x is not None:
        ax.scatter([env.x], [env.y], c="green", s=120, marker="o", edgecolors="k")

    ax.set_xticks(np.arange(-0.5, env.W, 1.0))
    ax.set_yticks(np.arange(-0.5, env.H, 1.0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="both", color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, env.W - 0.5)
    ax.set_ylim(-0.5, env.H - 0.5)
    ax.set_aspect("equal")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def make_video_from_dir(save_dir, output_path=None, fps=8):
    if output_path is None:
        output_path = os.path.join(save_dir, "heatmap_anim.mp4")
    frames = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(".png")])
    if len(frames) == 0:
        raise RuntimeError("No PNG frames found in " + save_dir)

    # First pass: read all images and determine the maximum width/height
    imgs = []
    max_h = 0
    max_w = 0
    for f in frames:
        img = imageio.imread(os.path.join(save_dir, f))
        imgs.append(img)
        h, w = img.shape[:2]
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

    # Round up to multiple of 16 for codec compatibility
    target_h = ((max_h + 15) // 16) * 16
    target_w = ((max_w + 15) // 16) * 16

    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for img in imgs:
            h, w = img.shape[:2]
            pad_h = target_h - h
            pad_w = target_w - w
            if pad_h != 0 or pad_w != 0:
                if img.ndim == 3:
                    img = np.pad(
                        img,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    img = np.pad(
                        img,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )
            writer.append_data(img)
    finally:
        writer.close()
    return output_path


def run_heatmap(
    seed=0,
    N=500,
    steps=400,
    save_every=5,
    save_dir=None,
    target=None,
    target_radius=0.5,
    use_true=False,
):
    env = RoombaSoftPOMDPEnv(
        width=20, height=20, exit_mode="reset", seed=seed, random_obstacles=True
    )
    env.reset()

    pf = ParticleFilter(env, N=N, use_hidden=True)
    pf.init_particles()

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "render_frames", "pf_heatmap")
    os.makedirs(save_dir, exist_ok=True)

    for t in range(steps):
        # sample random action for demo
        a = np.random.choice([0, 1, 2, 3], p=[0.5, 0.1, 0.2, 0.2])
        pf.predict(a)
        step_ret = env.step(a)
        if getattr(env, "_is_gymnasium", False):
            obs, r, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = step_ret

        pf.update(obs)

        if (t % save_every) == 0:
            heat = heatmap_from_particles(env, pf)
            out = os.path.join(save_dir, f"heat_{t:05d}.png")
            render_heatmap(env, heat, save_path=out)

        # termination: environment done or target reached
        if done:
            break

        if target is not None:
            tx, ty = target
            if use_true:
                true_x, true_y = env.x, env.y
                dist = ((true_x - tx) ** 2 + (true_y - ty) ** 2) ** 0.5
                if dist <= target_radius:
                    print(f"Target reached (true) at step {t}, dist={dist:.3f}")
                    break
            else:
                heat = heatmap_from_particles(env, pf)
                # belief-based terminate: if cumulative mass within radius of target exceeds threshold
                # compute cumulative mass in a disk of radius `target_radius` centered at tx,ty
                W = env.W
                H = env.H
                xs = np.arange(W)
                ys = np.arange(H)
                gx, gy = np.meshgrid(xs, ys)
                d2 = (gx - tx) ** 2 + (gy - ty) ** 2
                mask = d2 <= (target_radius**2)
                mass = float(np.sum(heat[mask]))
                if mass >= 0.5:
                    print(f"Target reached (belief mass) at step {t}, mass={mass:.3f}")
                    break

    # try to make video
    try:
        outvid = make_video_from_dir(
            save_dir, output_path=os.path.join(save_dir, "pf_heatmap_anim.mp4"), fps=8
        )
        print("Saved heatmap video:", outvid)
    except Exception as e:
        print("Could not create heatmap video:", e)


if __name__ == "__main__":
    run_heatmap(seed=1, N=2000, steps=200, save_every=10)
