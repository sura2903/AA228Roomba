from generate_test_scenarios import load_scenario
from setup import RoombaSoftPOMDPEnv
import os

# Load scenario
map_array, info = load_scenario("empty_room")

# Create environment
env = RoombaSoftPOMDPEnv(
    width=info['width'],
    height=info['height'],
    map_array=map_array
)

# Test
obs = env.reset()
total = 0.0
save_every = 50
save_dir = os.path.join(os.getcwd(), "render_frames", "run1")
os.makedirs(save_dir, exist_ok=True)
prefix = "run1"
for t in range(100):
    action = env.action_space.sample()
    step_ret = env.step(action)
    # Unpack step return depending on Gym vs Gymnasium API
    if getattr(env, "_is_gymnasium", False):
        # Gymnasium: obs, reward, terminated, truncated, info
        obs, r, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
    else:
        # Gym: obs, reward, done, info
        obs, r, done, info = step_ret

    total += r

    # if the environment reports explored fraction, terminate when coverage goal reached
    explored = info.get("explored_fraction") if isinstance(info, dict) else None
    if explored is not None and explored >= env.coverage_goal:
        print(f"coverage goal reached at step {t}: explored={explored:.4f}")
        break

    # save frame off-screen (does not open a GUI window)
    if t % save_every == 0:
        env.save_frame(save_dir=save_dir, prefix=prefix)

    if done:
        print("done at", t)
        break

    print("total reward", total)