# Setup imports and paths
import os, sys, glob, pprint
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

import analysis as analysis_module  # helper script
from setup import RoombaSoftPOMDPEnv


def save_heatmaps(csv, save_dir, map_name):
    # Analysis: specify a single map .npy and a single CSV file to analyze
    map_dir = './test_scenarios'
    MAP_NAME = map_name
    MAP_PATH = os.path.join(map_dir, f'{MAP_NAME}_map.npy')
    CSV_PATH = csv
    print('MAP_NAME=', MAP_NAME)
    print('MAP_PATH exists=', os.path.exists(MAP_PATH), 'CSV_PATH exists=', os.path.exists(CSV_PATH))

    # Run the analysis for the selected map+csv and display outputs inline.
    OUTDIR = save_dir
    # os.makedirs(OUTDIR, exist_ok=True)
    summary_rows = []
    # Single-file mode: use MAP_NAME, MAP_PATH, CSV_PATH
    map_key = MAP_NAME
    map_path = MAP_PATH
    csvp = CSV_PATH
    episode_nr = 0
    print('===>', map_key, '===')
    if not os.path.exists(map_path):
        print('Map file not found:', map_path)
    elif csvp is None or not os.path.exists(csvp):
        print('CSV file not found for', map_key, '(', csvp, ') -- skipping')
    else:
        print('Using CSV:', csvp)
        map_arr = np.load(map_path)
        H, W = map_arr.shape
        env = RoombaSoftPOMDPEnv(width=W, height=H, map_array=map_arr, random_obstacles=False)
        name = os.path.splitext(os.path.basename(csvp))[0]
        # out_sub = os.path.join(OUTDIR, map_key + '__' + name)
        # os.makedirs(out_sub, exist_ok=True)
        rows = analysis_module.read_csv(csvp)
        episodes = analysis_module.group_by_episode(rows)
        per_ep = []
        for i, ep_rows in enumerate(episodes):
            metrics = analysis_module.compute_metrics_for_episode(ep_rows, env=env)
            per_ep.append(metrics)
            # heatpath = os.path.join(out_sub, f'ep_{i:02d}_heat.png')
            # analysis_module.render_heatmap(env, metrics['heatmap'], outpath=heatpath, title=f'{name} ep {i} heatmap')
        dfrows = []
        for i, m in enumerate(per_ep):
            dfrows.append({'episode': i, 'total_reward': m['total_reward'], 'coverage': m['coverage'], 'entropy': m['entropy']})
        # df = pd.DataFrame(dfrows)
        # csv_out = os.path.join(out_sub, 'summary_per_episode.csv')
        # df.to_csv(csv_out, index=False)
        # print('Wrote', csv_out)
        # last_heat = os.path.join(out_sub, f'ep_{len(per_ep)-1:02d}_heat.png')
        # if os.path.exists(last_heat):
        #     display(Image(filename=last_heat))
        summary_rows.append({'map': map_key, 'csv': csvp, 'episodes': len(per_ep), 'mean_reward': float(np.mean([m['total_reward'] for m in per_ep])) if per_ep else None, 'mean_coverage': float(np.mean([m['coverage'] for m in per_ep])) if per_ep else None, 'mean_entropy': float(np.mean([m['entropy'] for m in per_ep])) if per_ep else None})
    
    # final summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTDIR, 'summary_all.csv')
    summary_df.to_csv(summary_csv, index=False)
    # print('Wrote overall summary to', summary_csv)
    display(summary_df)

    best_ep_idx = 0
    best_heat = per_ep[best_ep_idx]['heatmap']
    coverages = [m['coverage'] for m in per_ep]
    best_coverage = coverages[best_ep_idx]


    # Render with vmax=20
    heatpath_best = os.path.join(OUTDIR, 'heatmap_no_vmax.png')
    analysis_module.render_heatmap(env, best_heat, outpath=heatpath_best, title=f'{MAP_NAME} Episode {best_ep_idx} (Best Coverage: {best_coverage:.4f})')
    display(Image(filename=heatpath_best))

    heatpath_best = os.path.join(OUTDIR, 'heatmap_1_vmax.png')
    analysis_module.render_heatmap(env, best_heat, outpath=heatpath_best, title=f'{MAP_NAME} Episode {best_ep_idx} (Best Coverage: {best_coverage:.4f})', vmax=1)
    display(Image(filename=heatpath_best))
