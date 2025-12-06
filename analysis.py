#!/usr/bin/env python3
"""Analysis utilities for Roomba experiment CSVs.

Reads CSVs with headers:
s_x,s_y,s_theta,s_k,action,reward,sp_x,sp_y,sp_theta,sp_k,episode,step,done,coverage

Produces:
- metrics_summary.csv (final reward, coverage %, runtime if available)
- plots: rewards, reward breakdown (if map provided), entropy of visits
- diagrams: initial map (if provided) and final exploration heatmap

Usage:
    python analysis.py --csvs results/*.csv --outdir analysis_out [--map MAPFILE]

If no map is provided, obstacle-dependent reward components (collision) are not recomputed
and a proxy map (visited cells) will be used for diagrams.
"""

import os
import argparse
import csv
import math
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from setup import RoombaSoftPOMDPEnv


def read_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert numeric fields
            nr = {}
            for k, v in r.items():
                if v is None or v == "":
                    nr[k] = None
                    continue
                # try int, then float
                try:
                    nv = int(v)
                except Exception:
                    try:
                        nv = float(v)
                    except Exception:
                        nv = v
                nr[k] = nv
            rows.append(nr)
    return rows


def group_by_episode(rows):
    eps = defaultdict(list)
    for r in rows:
        ep = int(r.get("episode", 0) if r.get("episode", 0) is not None else 0)
        eps[ep].append(r)
    # sort episodes
    return [
        sorted(eps[k], key=lambda x: int(x.get("step", 0))) for k in sorted(eps.keys())
    ]


def compute_metrics_for_episode(ep_rows, env=None):
    # final reward = sum of reward column
    rewards = [float(r["reward"]) for r in ep_rows]
    total_reward = float(np.sum(rewards))

    # coverage: prefer explicit column 'coverage' if present in last row
    last = ep_rows[-1]
    coverage = last.get("coverage") if "coverage" in last else None
    if coverage is None:
        # compute coverage as unique sp_x,sp_y visited / total_visitable
        visited = set()
        for r in ep_rows:
            sx = int(r.get("sp_x", r.get("s_x", 0)))
            sy = int(r.get("sp_y", r.get("s_y", 0)))
            visited.add((sx, sy))
        if env is not None:
            total_visitable = int(np.sum(env.map != 1))
            coverage = (
                float(len([c for c in visited if env.map[c[1], c[0]] != 1]))
                / float(total_visitable)
                if total_visitable > 0
                else 1.0
            )
        else:
            # assume grid size from max coordinates
            xs = [int(r.get("sp_x", r.get("s_x", 0))) for r in ep_rows]
            ys = [int(r.get("sp_y", r.get("s_y", 0))) for r in ep_rows]
            W = max(xs) + 1
            H = max(ys) + 1
            coverage = float(len(visited)) / float(W * H) if W * H > 0 else 0.0

    # runtime: try to read 'runtime' or 'time' column else None
    runtime = None
    if "runtime" in last:
        runtime = last.get("runtime")
    elif "time" in last:
        runtime = last.get("time")

    # visit counts heatmap
    xs = [int(r.get("sp_x", r.get("s_x", 0))) for r in ep_rows]
    ys = [int(r.get("sp_y", r.get("s_y", 0))) for r in ep_rows]
    max_x = max(xs) if xs else 0
    max_y = max(ys) if ys else 0
    W = env.W if env is not None else max_x + 1
    H = env.H if env is not None else max_y + 1
    heat = np.zeros((H, W), dtype=float)
    for x, y in zip(xs, ys):
        if 0 <= x < W and 0 <= y < H:
            heat[y, x] += 1.0

    # entropy of visit distribution
    p = heat.flatten()
    s = p.sum()
    if s > 0:
        p = p / s
        mask = p > 0
        entropy = -float(np.sum(p[mask] * np.log(p[mask])))
    else:
        entropy = 0.0

    return dict(
        total_reward=total_reward,
        coverage=coverage,
        runtime=runtime,
        heatmap=heat,
        entropy=entropy,
        per_step_rewards=rewards,
    )


def plot_rewards(all_stats, outpath, title="Rewards per step"):
    plt.figure(figsize=(8, 5))
    for name, stats in all_stats.items():
        # aggregate per-step reward by episode average (pad episodes to same length)
        ep_rs = stats["per_episode_rewards"]
        maxlen = max(len(e) for e in ep_rs)
        arr = np.zeros((len(ep_rs), maxlen), dtype=float)
        for i, e in enumerate(ep_rs):
            arr[i, : len(e)] = e
            if len(e) < maxlen:
                arr[i, len(e) :] = np.nan
        mean = np.nanmean(arr, axis=0)
        plt.plot(mean, label=name)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_entropy(all_stats, outpath, title="Visit entropy (final)"):
    plt.figure(figsize=(8, 5))
    names = []
    vals = []
    for name, stats in all_stats.items():
        names.append(name)
        vals.append(stats["entropy"])
    plt.bar(names, vals)
    plt.ylabel("Entropy (nats)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def render_map(env, outpath, title="Map"):
    cmap = colors.ListedColormap(["#ffffff", "#444444", "#ffcc99"])
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(env.map, origin="lower", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(-0.5, env.W, 1.0))
    ax.set_yticks(np.arange(-0.5, env.H, 1.0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="both", color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, env.W - 0.5)
    ax.set_ylim(-0.5, env.H - 0.5)
    ax.set_aspect("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def render_heatmap(env, heat, outpath, title="Exploration heatmap", vmax=None):
    H, W = heat.shape
    # Create side-by-side figure: left = heatmap overlaid on map, right = map only
    fig, (ax_heat, ax_map) = plt.subplots(1, 2, figsize=(14, 6))

    # Colormap for map (0=free white, 1=hard black, 2=soft brown)
    cmap_map = colors.ListedColormap(["#ffffff", "#444444", "#ffcc99"])
    norm_map = colors.BoundaryNorm([0, 1, 2, 3], cmap_map.N)

    # LEFT: Heatmap overlaid on map
    ax_heat.imshow(env.map, origin="lower", cmap=cmap_map, norm=norm_map)
    hm = ax_heat.imshow(heat, origin="lower", cmap="magma", alpha=0.7, vmax=vmax)
    plt.colorbar(hm, ax=ax_heat, fraction=0.046, pad=0.04, label="Visit Count")
    ax_heat.set_xticks(np.arange(-0.5, W, 1.0))
    ax_heat.set_yticks(np.arange(-0.5, H, 1.0))
    ax_heat.set_xticklabels([])
    ax_heat.set_yticklabels([])
    ax_heat.grid(which="both", color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax_heat.set_xlim(-0.5, W - 0.5)
    ax_heat.set_ylim(-0.5, H - 0.5)
    ax_heat.set_aspect("equal")
    ax_heat.set_title("Exploration Heatmap + Obstacles")

    # RIGHT: Map only (no heatmap overlay) for reference
    ax_map.imshow(env.map, origin="lower", cmap=cmap_map, norm=norm_map)
    ax_map.set_xticks(np.arange(-0.5, W, 1.0))
    ax_map.set_yticks(np.arange(-0.5, H, 1.0))
    ax_map.set_xticklabels([])
    ax_map.set_yticklabels([])
    ax_map.grid(which="both", color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax_map.set_xlim(-0.5, W - 0.5)
    ax_map.set_ylim(-0.5, H - 0.5)
    ax_map.set_aspect("equal")
    ax_map.set_title("Map Reference\n(White=Free, Black=Hard, Brown=Soft)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", required=True, help="CSV files to analyze")
    p.add_argument("--outdir", default="analysis_out")
    p.add_argument(
        "--map", default=None, help="Optional numpy .npy map file to use for diagrams"
    )
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    env_map = None
    if args.map is not None:
        arr = np.load(args.map)
        env_map = arr

    all_stats = {}
    summary_rows = []

    for csvpath in args.csvs:
        name = os.path.splitext(os.path.basename(csvpath))[0]
        rows = read_csv(csvpath)
        episodes = group_by_episode(rows)
        per_ep_metrics = []
        per_episode_rewards = []
        heats = []
        entropies = []
        for ep_rows in episodes:
            if env_map is not None:
                env = RoombaSoftPOMDPEnv(
                    width=args.width,
                    height=args.height,
                    map_array=env_map,
                    random_obstacles=False,
                )
            else:
                env = None
            metrics = compute_metrics_for_episode(ep_rows, env=env)
            per_ep_metrics.append(metrics)
            per_episode_rewards.append(metrics["per_step_rewards"])
            heats.append(metrics["heatmap"])
            entropies.append(metrics["entropy"])

        # aggregate
        mean_total_reward = float(np.mean([m["total_reward"] for m in per_ep_metrics]))
        std_total_reward = float(np.std([m["total_reward"] for m in per_ep_metrics]))
        mean_coverage = float(np.mean([m["coverage"] for m in per_ep_metrics]))
        mean_entropy = float(np.mean(entropies))
        # runtime if available
        runtimes = [m["runtime"] for m in per_ep_metrics if m["runtime"] is not None]
        mean_runtime = float(np.mean(runtimes)) if runtimes else None

        all_stats[name] = dict(
            per_episode_metrics=per_ep_metrics,
            per_episode_rewards=per_episode_rewards,
            heatmaps=heats,
            entropy=mean_entropy,
        )

        summary_rows.append(
            dict(
                name=name,
                mean_total_reward=mean_total_reward,
                std_total_reward=std_total_reward,
                mean_coverage=mean_coverage,
                mean_entropy=mean_entropy,
                mean_runtime=mean_runtime,
            )
        )

        # save final heatmap of last episode
        final_heat = heats[-1]
        if env is None:
            # create proxy env with size
            H, W = final_heat.shape
            env = RoombaSoftPOMDPEnv(width=W, height=H)
        render_heatmap(
            env,
            final_heat,
            outpath=os.path.join(args.outdir, f"{name}_final_heatmap.png"),
            title=f"{name} final heatmap",
        )
        # if map provided, also render initial map
        if env_map is not None:
            render_map(
                env,
                outpath=os.path.join(args.outdir, f"{name}_map.png"),
                title=f"{name} initial map",
            )

    # write summary CSV
    summary_path = os.path.join(args.outdir, "metrics_summary.csv")
    keys = [
        "name",
        "mean_total_reward",
        "std_total_reward",
        "mean_coverage",
        "mean_entropy",
        "mean_runtime",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # plots
    plot_rewards(all_stats, outpath=os.path.join(args.outdir, "rewards_vs_steps.png"))
    plot_entropy(all_stats, outpath=os.path.join(args.outdir, "entropy_bar.png"))

    print("Analysis complete. Outputs in", args.outdir)


if __name__ == "__main__":
    main()
