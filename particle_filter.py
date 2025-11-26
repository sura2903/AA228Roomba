import numpy as np
from collections import Counter
import os

# TODO: add option later optionally to add jittering to particles during resampling


def circular_diff(a, b, period=4.0):
    """Minimal difference a - b on a circular domain [0,period)."""
    d = a - b
    d = (d + 0.5 * period) % period - 0.5 * period
    return d


class ParticleFilter:
    """Simple particle filter for the Roomba POMDP.

    Particles represent full hidden state (x,y,theta,d,k) by default.
    The filter uses the environment's `sample_transition` to propagate particles
    and a Gaussian observation likelihood using `env.obs_sigma` and
    `env.theta_obs_sigma`.
    """

    def __init__(self, env, N=500, use_hidden=True, rng=None, jitter_std=0.0):
        self.env = env
        self.N = int(N)
        self.use_hidden = use_hidden
        self.rng = np.random.RandomState() if rng is None else rng
        # jitter_std: after resampling, add gaussian jitter (in cell units) to x,y then round
        self.jitter_std = float(jitter_std)

        # particle arrays (initialized in init_particles)
        self.xs = None
        self.ys = None
        self.thetas = None
        self.ds = None
        self.ks = None
        self.ws = None

    def init_particles(self, mode="uniform"):
        """Initialize particles.

        mode: currently only 'uniform' supported (uniform over free cells and headings).
        """
        free = np.argwhere(self.env.map == 0)
        if len(free) == 0:
            raise RuntimeError("No free cells to initialize particles")

        # sample with replacement from free cell indices
        idx = self.rng.randint(0, len(free), size=self.N)
        samples = free[idx]
        ys = samples[:, 0]
        xs = samples[:, 1]

        thetas = self.rng.randint(0, 4, size=self.N)

        self.xs = xs.astype(int).copy()
        self.ys = ys.astype(int).copy()
        self.thetas = thetas.astype(int).copy()
        if self.use_hidden:
            self.ds = -1 * np.ones(self.N, dtype=int)
            self.ks = np.zeros(self.N, dtype=int)
        else:
            self.ds = -1 * np.ones(self.N, dtype=int)
            self.ks = np.zeros(self.N, dtype=int)

        self.ws = np.ones(self.N, dtype=np.float64) / float(self.N)

    def predict(self, action):
        """Propagate particles through the environment transition model."""
        for i in range(self.N):
            state = (int(self.xs[i]), int(self.ys[i]), int(self.thetas[i]))
            hidden = (int(self.ds[i]), int(self.ks[i]))
            (nx, ny, ntheta), (nd, nk) = self.env.sample_transition(
                state, hidden, int(action)
            )
            self.xs[i] = int(nx)
            self.ys[i] = int(ny)
            self.thetas[i] = int(ntheta) % 4
            self.ds[i] = int(nd)
            self.ks[i] = int(nk)

    def update(self, obs, resample_thresh=None):
        """Weight particles by observation likelihood and optionally resample.

        obs: array-like [x_obs, y_obs, theta_obs]
        resample_thresh: if provided, resample when effective sample size < thresh
        """
        obs = np.asarray(obs, dtype=float)
        sx = float(self.env.obs_sigma)
        st = float(self.env.theta_obs_sigma)

        # compute log-likelihoods for numerical stability
        dx = obs[0] - self.xs
        dy = obs[1] - self.ys
        dt = circular_diff(obs[2], self.thetas, period=4.0)

        # gaussian log-likelihoods (unnormalized)
        logp_xy = -0.5 * ((dx * dx + dy * dy) / (sx * sx + 1e-12))
        logp_t = -0.5 * ((dt * dt) / (st * st + 1e-12))
        logw = logp_xy + logp_t

        # subtract max for stability
        logw = logw - np.max(logw)
        w = np.exp(logw)
        w = w + 1e-300
        w = w / np.sum(w)
        self.ws = w

        # effective sample size
        ess = 1.0 / np.sum(self.ws**2)
        if resample_thresh is None:
            resample_thresh = 0.5 * float(self.N)
        if ess < resample_thresh:
            self.resample()

    def resample(self):
        """Systematic resampling."""
        N = self.N
        positions = (np.arange(N) + self.rng.random_sample()) / float(N)
        cumulative = np.cumsum(self.ws)
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        # resample arrays
        self.xs = self.xs[indexes].copy()
        self.ys = self.ys[indexes].copy()
        self.thetas = self.thetas[indexes].copy()
        self.ds = self.ds[indexes].copy()
        self.ks = self.ks[indexes].copy()
        self.ws = np.ones(N, dtype=np.float64) / float(N)

        # optional jitter to maintain diversity: add small gaussian noise and re-discretize
        if self.jitter_std and self.jitter_std > 0.0:
            # store originals to fallback if jitter lands in hard cell or OOB
            orig_xs = self.xs.copy()
            orig_ys = self.ys.copy()
            # add continuous gaussian noise then round
            noisy_x = self.xs.astype(float) + self.rng.normal(
                scale=self.jitter_std, size=N
            )
            noisy_y = self.ys.astype(float) + self.rng.normal(
                scale=self.jitter_std, size=N
            )
            new_x = np.round(noisy_x).astype(int)
            new_y = np.round(noisy_y).astype(int)
            # clip to bounds
            new_x = np.clip(new_x, 0, self.env.W - 1)
            new_y = np.clip(new_y, 0, self.env.H - 1)
            # if jittered cell is hard, revert to original
            for ii in range(N):
                if not (0 <= new_x[ii] < self.env.W and 0 <= new_y[ii] < self.env.H):
                    new_x[ii] = orig_xs[ii]
                    new_y[ii] = orig_ys[ii]
                elif self.env.is_hard(new_x[ii], new_y[ii]):
                    new_x[ii] = orig_xs[ii]
                    new_y[ii] = orig_ys[ii]
            self.xs = new_x
            self.ys = new_y

    def estimate(self):
        """Return a point estimate (mean x,y and MAP theta)."""
        mean_x = float(np.sum(self.ws * self.xs))
        mean_y = float(np.sum(self.ws * self.ys))
        # MAP theta
        theta_counts = Counter()
        for th, w in zip(self.thetas, self.ws):
            theta_counts[int(th)] += float(w)
        map_theta = max(theta_counts.items(), key=lambda kv: kv[1])[0]
        return (mean_x, mean_y, float(map_theta))


def render_particles(
    env,
    pf: ParticleFilter,
    save_path=None,
    figsize=(6, 6),
    show=False,
    show_visits=False,
):
    """Render environment map with particle overlay and optional save.

    - `pf` must have attributes `xs, ys, thetas, ws`.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    display = np.copy(env.map).astype(np.int8)
    cmap = colors.ListedColormap(["#ffffff", "#444444", "#ffcc99"])  # free, hard, soft
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(display, origin="lower", cmap=cmap, norm=norm)

    # plot particles (weight-scaled)
    if pf is not None and pf.xs is not None:
        # normalize weights to [0,1]
        ws = np.asarray(pf.ws, dtype=float)
        if ws.sum() <= 0:
            ws = np.ones_like(ws) / float(len(ws))
        wnorm = ws / (np.max(ws) + 1e-12)
        # sizes and alpha: emphasize heavier particles
        max_marker = 80
        sizes = 2.0 + (max_marker - 2.0) * (wnorm**0.6)
        alphas = 0.15 + 0.7 * (wnorm**0.5)
        # plot in small batches grouped by rounded weight to reduce draw calls
        rounded = np.round(wnorm, 3)
        unique_vals = np.unique(rounded)
        for u in unique_vals:
            if u <= 0:
                continue
            idx = np.where(rounded == u)[0]
            if idx.size == 0:
                continue
            ax.scatter(
                pf.xs[idx],
                pf.ys[idx],
                c="blue",
                s=sizes[idx],
                alpha=float(alphas[idx][0]),
                edgecolors="none",
            )

    # estimated mean
    est = pf.estimate() if pf is not None else None
    if est is not None:
        mx, my, mth = est
        ax.scatter([mx], [my], c="red", s=80, marker="x")

    # robot true state overlay (if available)
    if hasattr(env, "x") and env.x is not None:
        ax.scatter([env.x], [env.y], c="green", s=120, marker="o", edgecolors="k")

    # visits overlay: small text in blue (optional)
    if show_visits and hasattr(env, "visit_counts"):
        for yy in range(env.H):
            for xx in range(env.W):
                v = int(env.visit_counts[yy, xx])
                if v > 0:
                    ax.text(
                        xx,
                        yy,
                        str(min(9, v)),
                        color="blue",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

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
