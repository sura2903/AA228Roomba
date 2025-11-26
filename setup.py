# roomba_soft_pomdp.py
# Gym environment: Roomba POMDP with soft regions (S2), state=(x,y,theta,d,k).
# Grid encoding: 0 = free, 1 = hard obstacle, 2 = soft region
# Theta encoding: 0=N, 1=E, 2=S, 3=W

try:
    import gymnasium as gym
except Exception:
    import gym

# Use the `spaces` object from whichever gym package was imported (gymnasium or gym)
spaces = gym.spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio.v2 as imageio
import os
import time

# Direction utilities: 0=N,1=E,2=S,3=W
DIR2DELTA = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
LEFT_OF = {0: 3, 1: 0, 2: 1, 3: 2}
RIGHT_OF = {0: 1, 1: 2, 2: 3, 3: 0}
REVERSE_DIR = {0: 2, 1: 3, 2: 0, 3: 1}


class RoombaSoftPOMDPEnv(gym.Env):
    """
    Roomba POMDP environment (soft-region S2).
    - State (internal): (x, y, theta, d, k)
      d = -1 means not currently tracked (k==0)
    - Observation (returned): noisy (x_obs, y_obs, theta_obs) only.
    - Action set: {0: forward, 1: backward, 2: turn_left, 3: turn_right}
    - Transition model implemented in sample_transition()
    - Reward implemented in reward_model()
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        width=20,
        height=20,
        map_array=None,
        p_hard=0.05,
        p_soft=0.1,
        random_obstacles=True,
        exit_mode="reset",  # "reset" or "decrement"
        K=3,
        p_intended=0.7,
        p_adj=0.1,
        p_stay=0.1,
        p_worsen=0.2,  # used if you want stochastic worsen
        p_exit_base=0.9,
        p_exit_alpha=0.5,
        obs_sigma=0.3,
        theta_obs_sigma=0.05,
        time_penalty=0.01,
        stuck_penalty=0.5,
        collision_penalty=0.5,
        coverage_goal=0.95,
        max_steps=1000,
        seed=None,
    ):
        super().__init__()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # detect whether we're running under Gymnasium (newer API)
        self._is_gymnasium = getattr(gym, "__name__", "") == "gymnasium"

        self.W = width
        self.H = height

        # Map: H x W (row-major: y, x)
        if map_array is None:
            if random_obstacles:
                # random map with hard and soft obstacles
                self.map = np.zeros((self.H, self.W), dtype=np.int8)
                rand = np.random.rand(self.H, self.W)
                self.map[rand < p_hard] = 1
                self.map[(rand >= p_hard) & (rand < p_hard + p_soft)] = 2
            else:
                # default simple map: mostly free with a soft region and some hard obstacles
                self.map = np.zeros((self.H, self.W), dtype=np.int8)
                # create a soft region (connected block)
                sx0, sy0 = max(2, self.W // 3), max(2, self.H // 3)
                for yy in range(sy0, sy0 + 4):
                    for xx in range(sx0, sx0 + 4):
                        if 0 <= yy < self.H and 0 <= xx < self.W:
                            self.map[yy, xx] = 2
                # some hard obstacles
                for yy in range(3, 7):
                    if 10 < self.W:
                        self.map[yy, 10] = 1
        else:
            assert map_array.shape == (self.H, self.W)
            self.map = np.array(map_array, dtype=np.int8)

        # internal state
        self.x = None
        self.y = None
        self.theta = None  # 0..3
        self.d = -1  # -1 means no stored entry direction
        self.k = 0

        # transition params
        self.exit_mode = exit_mode
        assert self.exit_mode in ("reset", "decrement")
        self.K = K
        self.p_intended = p_intended
        self.p_adj = p_adj
        self.p_stay = p_stay
        assert abs(self.p_intended + 2 * self.p_adj + self.p_stay - 1.0) < 1e-6
        self.p_worsen = p_worsen
        self.p_exit_base = p_exit_base
        self.p_exit_alpha = p_exit_alpha

        # observation noise
        self.obs_sigma = obs_sigma
        self.theta_obs_sigma = theta_obs_sigma

        # rewards
        self.time_penalty = time_penalty
        self.stuck_penalty = stuck_penalty
        self.collision_penalty = collision_penalty

        # termination
        self.coverage_goal = coverage_goal
        self.max_steps = max_steps

        # bookkeeping
        self.visit_counts = np.zeros((self.H, self.W), dtype=np.int32)
        self.steps = 0

        # gym spaces
        # actions: 0..3 as described above
        self.action_space = spaces.Discrete(4)
        # observation: [x_obs, y_obs, theta_obs] continuous
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array(
                [float(self.W - 1), float(self.H - 1), 3.0], dtype=np.float32
            ),
            dtype=np.float32,
        )

        # initialize
        self.reset()

    # ---------------------
    # Helper utilities
    # ---------------------
    def in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def is_hard(self, x, y):
        if not self.in_bounds(x, y):
            return True
        return self.map[y, x] == 1

    def is_soft(self, x, y):
        if not self.in_bounds(x, y):
            return False
        return self.map[y, x] == 2

    def p_exit(self, k):
        if k <= 1:
            return self.p_exit_base
        return self.p_exit_base / (1.0 + self.p_exit_alpha * (k - 1))

    def sample_executed_relative(self, intended_rel):
        """
        Sample executed relative direction given intended relative direction.
        intended_rel is in {0: forward, 1: left, 2: right, 3: stay} encoded relative to current heading.
        But to be consistent with earlier spec, we'll use choices = [intended, left, right, stay].
        Return one of {0: intended, 1: left, 2: right, 3: stay} (relative codes).
        """
        choices = [0, 1, 2, 3]
        probs = [self.p_intended, self.p_adj, self.p_adj, self.p_stay]
        return np.random.choice(choices, p=probs)

    # ---------------------
    # Core transition model
    # ---------------------
    def sample_transition(self, state, hidden, action):
        """
        Given state (x,y,theta) and hidden (d,k) and intended action,
        sample (state', hidden') according to the specified kernel.

        Movement noise applies to forward/back moves; turns are deterministic.
        The function implements S2 semantics: soft obstacles are regions (connected cells).
        """
        x, y, theta = state
        d, k = hidden

        # Default next = stay
        nx, ny, ntheta = x, y, theta
        nd, nk = d, k

        # Turning actions are deterministic and do not change x,y,k,d
        if action == 2:  # TURN_LEFT
            ntheta = (theta - 1) % 4
            return (nx, ny, ntheta), (nd, nk)
        if action == 3:  # TURN_RIGHT
            ntheta = (theta + 1) % 4
            return (nx, ny, ntheta), (nd, nk)

        # Movement actions: forward or backward relative to heading
        if action == 0:  # forward -> intended absolute direction = theta
            intended_abs = theta
        elif action == 1:  # backward -> intended absolute direction = theta + 2
            intended_abs = (theta + 2) % 4
        else:
            # invalid action
            intended_abs = theta

        # sample executed relative outcome: 0=intended,1=left,2=right,3=stay
        rel = self.sample_executed_relative(intended_rel=0)
        if rel == 3:
            # stay
            return (nx, ny, ntheta), (nd, nk)

        # map relative to absolute executed direction
        if rel == 0:
            e = intended_abs
        elif rel == 1:
            e = LEFT_OF[intended_abs]
        elif rel == 2:
            e = RIGHT_OF[intended_abs]
        else:
            e = intended_abs

        # candidate cell
        dx, dy = DIR2DELTA[e]
        candx = x + dx
        candy = y + dy

        # ---------- CASE: k == 0 (free) ----------
        if k == 0:
            # blocked by hard or OOB -> stay
            if self.is_hard(candx, candy):
                return (nx, ny, ntheta), (nd, nk)

            # if entering soft region (S2): move into the soft cell and set d,k
            if self.is_soft(candx, candy):
                nx, ny, ntheta = candx, candy, e
                nd = REVERSE_DIR[e]  # store direction that leads out if robot reverses
                nk = 1
                return (nx, ny, ntheta), (nd, nk)

            # else free cell -> move and remain free
            nx, ny, ntheta = candx, candy, e
            nd, nk = -1, 0
            return (nx, ny, ntheta), (nd, nk)

        # ---------- CASE: k > 0 (inside soft region) ----------
        # If executed direction equals stored 'reverse' direction d (attempt to exit)
        if e == d:
            p_exit = self.p_exit(k)
            if np.random.rand() < p_exit:
                # attempt to exit: cell in direction d
                ex, ey = x + DIR2DELTA[d][0], y + DIR2DELTA[d][1]
                # if exit cell is hard/out-of-bounds -> fail to exit
                if self.is_hard(ex, ey):
                    nk = min(k + 1, self.K)
                    return (x, y, theta), (d, nk)
                # else we move into ex,ey (successful exit)
                nx, ny, ntheta = ex, ey, d  # per Option A, set theta' = d
                if self.exit_mode == "reset":
                    nd, nk = -1, 0
                else:  # "decrement"
                    nk = max(k - 1, 0)
                    nd = d if nk > 0 else -1
                return (nx, ny, ntheta), (nd, nk)
            else:
                # failed exit attempt: remain, depth increases (saturate)
                nk = min(k + 1, self.K)
                return (x, y, theta), (d, nk)

        # ---------- e != d (wrong direction while stuck) ----------
        # If moving into hard cell -> stay but maybe update orientation
        if self.is_hard(candx, candy):
            ntheta = e
            return (x, y, ntheta), (d, k)

        # else allow move (into soft region or free), and increase depth
        nx, ny, ntheta = candx, candy, e
        nk = min(k + 1, self.K)
        return (nx, ny, ntheta), (d, nk)

    # ---------------------
    # Reward model
    # ---------------------
    def reward_model(self, prev_state, prev_hidden, action, next_state, next_hidden):
        """
        Reward composition:
         - time penalty
         - exploration reward: 1/(1 + visits) for the next cell
         - stuck penalty proportional to k (next_hidden)
         - collision penalty if attempted move into hard obstacle
        """
        px, py, ptheta = prev_state
        pd, pk = prev_hidden
        nx, ny, ntheta = next_state
        nd, nk = next_hidden

        r = -self.time_penalty

        # exploration reward (encourage visiting new cells), measured on next cell
        visits = self.visit_counts[ny, nx]
        r += 1.0 / (1.0 + visits)

        # stuck penalty (based on next hidden depth)
        if nk > 0:
            r -= self.stuck_penalty * float(nk)

        # collision penalty: if attempted cell (intended move) was hard and agent stayed
        # detect: agent intended forward/back and next pos equals prev pos while intended would have changed pos
        if action in (0, 1):
            intended_abs = ptheta if action == 0 else (ptheta + 2) % 4
            intended_dx, intended_dy = DIR2DELTA[intended_abs]
            intended_x = px + intended_dx
            intended_y = py + intended_dy
            # if intended cell is hard and agent didn't change cells, penalize
            if self.is_hard(intended_x, intended_y) and (nx == px and ny == py):
                r -= self.collision_penalty

        return r

    # ---------------------
    # Observation model
    # ---------------------
    def observe(self):
        """
        Return noisy observation: (x_obs, y_obs, theta_obs)
        Note: theta_obs is returned as continuous float in [0,3], agent may discretize if needed.
        """
        x_obs = float(self.x) + np.random.normal(scale=self.obs_sigma)
        y_obs = float(self.y) + np.random.normal(scale=self.obs_sigma)
        theta_obs = float(self.theta) + np.random.normal(scale=self.theta_obs_sigma)
        # clamp
        x_obs = np.clip(x_obs, 0.0, float(self.W - 1))
        y_obs = np.clip(y_obs, 0.0, float(self.H - 1))
        theta_obs = np.clip(theta_obs, 0.0, 3.0)
        return np.array([x_obs, y_obs, theta_obs], dtype=np.float32)

    # ---------------------
    # Gym API: reset
    # ---------------------
    def reset(self, seed=None, options=None):
        # call parent reset (Gym/Gymnasium compatibility)
        try:
            parent_ret = super().reset(seed=seed)
        except TypeError:
            # older gym may not accept seed as keyword
            parent_ret = super().reset(seed)
        free_cells = np.argwhere(self.map == 0)
        if len(free_cells) == 0:
            # fallback
            self.x, self.y = 0, 0
        else:
            idx = np.random.choice(len(free_cells))
            yy, xx = free_cells[idx]
            self.x, self.y = int(xx), int(yy)
        self.theta = int(np.random.choice([0, 1, 2, 3]))
        self.d = -1
        self.k = 0
        self.visit_counts[:] = 0
        self.visit_counts[self.y, self.x] = 1
        self.steps = 0
        if self._is_gymnasium:
            # Gymnasium expects (obs, info)
            return self.observe(), {}
        return self.observe()

    # ---------------------
    # Gym API: step
    # ---------------------
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        prev_state = (self.x, self.y, self.theta)
        prev_hidden = (self.d, self.k)

        (nx, ny, ntheta), (nd, nk) = self.sample_transition(
            prev_state, prev_hidden, action
        )

        # update true state
        self.x, self.y, self.theta = int(nx), int(ny), int(ntheta) % 4
        self.d, self.k = int(nd), int(nk)

        # update visit counts
        self.visit_counts[self.y, self.x] += 1

        # compute reward
        next_state = (self.x, self.y, self.theta)
        next_hidden = (self.d, self.k)
        reward = self.reward_model(
            prev_state, prev_hidden, action, next_state, next_hidden
        )

        self.steps += 1

        # termination: coverage or max_steps
        explored = float(np.sum(self.visit_counts > 0)) / float(self.W * self.H)
        done = (explored >= self.coverage_goal) or (self.steps >= self.max_steps)

        # info: include true hidden state for debugging only (agent should not use it)
        info = {
            "true_state": (self.x, self.y, self.theta),
            "true_hidden": (self.d, self.k),
            "explored_fraction": explored,
        }

        obs = self.observe()
        if self._is_gymnasium:
            # Gymnasium step API: obs, reward, terminated, truncated, info
            terminated = bool(done)
            truncated = False
            return obs, float(reward), terminated, truncated, info
        return obs, float(reward), bool(done), info

    # ---------------------
    # Render (ASCII)
    # ---------------------
    def render(
        self,
        mode="human",
        save=False,
        save_dir=None,
        prefix=None,
        figsize=(6, 6),
        show_visits=True,
    ):
        """
        Primary renderer: delegates to the matplotlib renderer.

        - `mode`: same as `render_matplotlib` (`"human"` or `"rgb_array").
        - `save`: if True, also save the frame to `save_dir` using `save_frame`.
        - `save_dir`: directory to save frames (defaults to `./render_frames`).
        - `prefix`: filename prefix for saved frames.
        - `figsize`, `show_visits`: passed to `render_matplotlib`.
        """
        # If saving, render off-screen and save without opening a GUI window.
        if save:
            out_path = self.save_frame(save_dir=save_dir, prefix=prefix)
            # if user requested an array, provide it; otherwise return saved path
            if mode == "rgb_array":
                return self.render_matplotlib(
                    mode="rgb_array",
                    figsize=figsize,
                    show_visits=show_visits,
                    show=False,
                )
            return out_path

        # default: render to screen or return array
        return self.render_matplotlib(
            mode=mode, figsize=figsize, show_visits=show_visits
        )

    def render_matplotlib(
        self, mode="human", figsize=(6, 6), show_visits=True, cmap=None, show=True
    ):
        """
        Matplotlib renderer.
        - mode="human": show non-blocking interactive window.
        - mode="rgb_array": return an HxWx3 uint8 RGB image.
        """
        # build display grid: 0=free,1=hard,2=soft
        display = np.copy(self.map).astype(np.int8)

        # default colormap: free=white, hard=dark gray, soft=light orange
        if cmap is None:
            cmap = colors.ListedColormap(
                ["#ffffff", "#444444", "#ffcc99"]
            )  # free, hard, soft
            bounds = [0, 1, 2, 3]
            norm = colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.imshow(display, origin="lower", cmap=cmap, norm=norm)

        # visits overlay: small text in blue
        if show_visits:
            for yy in range(self.H):
                for xx in range(self.W):
                    v = int(self.visit_counts[yy, xx])
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

        # robot marker and heading arrow
        if self.in_bounds(self.x, self.y):
            dx, dy = DIR2DELTA[self.theta]
            ax.scatter(
                [self.x], [self.y], c="red", s=120, marker="o", edgecolors="k", zorder=3
            )
            ax.arrow(
                self.x - 0.3 * dx,
                self.y - 0.3 * dy,
                0.6 * dx,
                0.6 * dy,
                head_width=0.2,
                head_length=0.2,
                fc="k",
                ec="k",
                zorder=4,
            )

        ax.set_xticks(np.arange(-0.5, self.W, 1.0))
        ax.set_yticks(np.arange(-0.5, self.H, 1.0))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(which="both", color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.set_xlim(-0.5, self.W - 0.5)
        ax.set_ylim(-0.5, self.H - 0.5)
        ax.set_aspect("equal")

        if mode == "human":
            if show:
                plt.show(block=False)
                plt.pause(0.001)
        elif mode == "rgb_array":
            # Use Agg canvas for reliable off-screen rendering and RGB buffer
            canvas = FigureCanvas(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            # Use ARG B buffer then convert to RGB to be compatible across backends
            buf = canvas.tostring_argb()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            # ARGB -> RGBA
            arr = arr[:, :, [1, 2, 3, 0]]
            img = arr[:, :, :3].copy()
            plt.close(fig)
            return img
        else:
            if show:
                plt.show()

    def save_frame(self, save_dir=None, prefix=None):
        """
        Render and save a single frame (PNG) to `save_dir`.
        - `save_dir`: directory path; defaults to `./render_frames`.
        - `prefix`: filename prefix, default `frame`.
        Returns the saved filepath.
        """
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "render_frames")
        os.makedirs(save_dir, exist_ok=True)

        img = self.render_matplotlib(mode="rgb_array", show=False)
        prefix = prefix or "frame"
        filename = f"{prefix}_{self.steps:06d}.png"
        out_path = os.path.join(save_dir, filename)
        plt.imsave(out_path, img)
        return out_path

    def make_video(self, save_dir=None, output_path=None, fps=10):
        """
        Stitch saved frames in `save_dir` into a video at `output_path` using imageio.
        - `save_dir`: directory containing frames saved by `save_frame` (default `./render_frames`).
        - `output_path`: file path to write (default: `render_animation.mp4` in save_dir).
        - `fps`: frames per second.
        Returns `output_path` on success.
        """
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "render_frames")
        os.makedirs(save_dir, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(save_dir, "render_animation.mp4")

        frames = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(".png")])
        if len(frames) == 0:
            raise RuntimeError(f"No PNG frames found in {save_dir}")

        writer = imageio.get_writer(output_path, fps=fps)
        try:
            for fname in frames:
                full = os.path.join(save_dir, fname)
                img = imageio.imread(full)
                # ensure dimensions are divisible by 16 (macro block size) to avoid ffmpeg resizing
                h, w = img.shape[:2]
                new_h = ((h + 15) // 16) * 16
                new_w = ((w + 15) // 16) * 16
                if new_h != h or new_w != w:
                    pad_h = new_h - h
                    pad_w = new_w - w
                    # pad bottom and right with black pixels
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


# minimal demo when run directly
if __name__ == "__main__":
    env = RoombaSoftPOMDPEnv(
        width=20, height=20, exit_mode="reset", seed=0, random_obstacles=True
    )
    obs = env.reset()
    print("initial obs", obs)
    total = 0.0
    # Record frames every `save_every` steps and stitch into a video afterwards.
    save_every = 10
    total_steps = 2000
    save_dir = os.path.join(os.getcwd(), "render_frames", "run1")
    os.makedirs(save_dir, exist_ok=True)
    prefix = "run1"

    for t in range(total_steps):
        # random policy demo: 50% move forward, 10% back, 20% left turn, 20% right turn
        a = np.random.choice([0, 1, 2, 3], p=[0.5, 0.1, 0.2, 0.2])
        step_ret = env.step(a)

        # Unpack step return depending on Gym vs Gymnasium API
        if getattr(env, "_is_gymnasium", False):
            # Gymnasium: obs, reward, terminated, truncated, info
            obs, r, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            # Gym: obs, reward, done, info
            obs, r, done, info = step_ret

        total += r

        # save frame off-screen (does not open a GUI window)
        if t % save_every == 0:
            env.save_frame(save_dir=save_dir, prefix=prefix)

        if done:
            print("done at", t)
            break

    print("total reward", total)

    # Attempt to make a video from saved frames
    try:
        out_path = env.make_video(
            save_dir=save_dir,
            output_path=os.path.join(save_dir, f"{prefix}_anim.mp4"),
            fps=5,
        )
        print(f"Saved video to: {out_path}")
    except Exception as e:
        print("Could not create video:", e)
