#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eliezyer Fermino de Oliveira

Script for processing the data of HPC task learning, and plotting the results. Conditions A and B is home cage activity
pre- and post- task learning.
"""

# importing packages
import sys
import numpy as np
import pynapple as nap

import matplotlib.pyplot as plt

from contrastive import CPCA
from scipy.stats import zscore
from scipy.signal import savgol_filter
import mat73  # to load matlab v7.3 files

# parameters for processing
min_n_cell = 30  # min number of cells in the brain area to be used
min_fr = 0.01  # minimum firing rate to be included in the analysis
bin_size = 0.01  # 10 ms
std_conv = 2  # standard deviation for convolution (in num bins units)
wind_cov = 5  # window size for convolution (in num bins units)
roll_wind = 20  # window size for rolling mean

# parameters for plots
mrkr_size = 30
plt.rcParams['figure.dpi'] = 150
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12

# import custom modules
# change repo_dir to your gcPCA directory
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

# other directories
# directory with the HPC data
base_dir = '/mnt/probox/buzsakilab.nyumc.org/datasets/GirardeauG/'

# directory where to save the figures
save_fig_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/Figure3/source_plots/'

# defining ancillar functions
def extract_trials(temp, temp_spd):
    """Function to extract running trials from GG dataset where animals
    run in a linear track with airpuff. It returns left and right runs (trials) in the maze
    """

    max_len = 10  # max length of trial, in seconds
    min_len = 0.5  # min length of trial, in seconds
    min_clip, max_clip = 150, 450  # this is for clipping the trials in space
    min_dist = 150  # in cm (avoid runs where animal goes to half and back)
    min_speed = 3  # minimal speed is 3 cm/s in the trial

    # running speed
    speed = nap.Tsd(np.array(temp.index[:-1]), np.abs(np.diff(temp.values)) / np.diff(temp.index))
    logical_left = np.diff(temp.as_series().clip(lower=min_clip, upper=max_clip).values) < 0
    logical_right = np.diff(temp.as_series().clip(lower=min_clip, upper=max_clip).values) > 0

    logical_left = np.append(logical_left, False)
    logical_right = np.append(logical_right, False)
    logical_left = np.insert(logical_left, 0, False)
    logical_right = np.insert(logical_right, 0, False)
    ts = temp.as_series().clip(lower=min_clip, upper=max_clip).index

    # finding start and stop of left runs
    temp_st = np.argwhere(np.diff(logical_left.astype(int)) == 1) + 1
    temp_sp = np.argwhere(np.diff(logical_left.astype(int)) == -1)

    # picking only the intervals that lasted > min_len s and < max_len s
    start1 = ts.values[temp_st.flatten()]
    stop1 = ts.values[temp_sp.flatten()]
    int2keep = ((stop1 - start1) > min_len) * ((stop1 - start1) < max_len)
    start2 = start1[int2keep].copy()
    stop2 = stop1[int2keep].copy()
    trials2keep = []
    for a in np.arange(len(start2)):
        interval = nap.IntervalSet(start=start2[a], end=stop2[a])
        tempt = speed.restrict(interval).index[speed.restrict(interval).values > min_speed]
        if tempt.size > 0:
            new_interval = nap.IntervalSet(start=tempt[0], end=tempt[-1])
            if (temp.restrict(new_interval).as_series().max() - temp.restrict(
                    new_interval).as_series().min()) > min_dist:  # min distance in cm that the trial is getting
                trials2keep.append(True)
                start2[a], stop2[a] = tempt[0], tempt[-1]
            else:
                trials2keep.append(False)
        else:
            trials2keep.append(False)

    left_runs_interval = nap.IntervalSet(start=start2[trials2keep], end=stop2[trials2keep])

    # finding start and stop of right runs
    temp_st = np.argwhere(np.diff(logical_right.astype(int)) == 1) + 1
    temp_sp = np.argwhere(np.diff(logical_right.astype(int)) == -1)

    # picking only the intervals that lasted > 1 s and <3s
    start1 = ts.values[temp_st.flatten()]
    stop1 = ts.values[temp_sp.flatten()]
    int2keep = ((stop1 - start1) > min_len) * ((stop1 - start1) < max_len)
    start2 = start1[int2keep].copy()
    stop2 = stop1[int2keep].copy()

    trials2keep = []
    for a in np.arange(len(start2)):
        interval = nap.IntervalSet(start=start2[a], end=stop2[a])
        tempt = speed.restrict(interval).index[speed.restrict(interval).values > min_speed]
        if tempt.size > 0:
            new_interval = nap.IntervalSet(start=tempt[0], end=tempt[-1])
            if (temp.restrict(new_interval).as_series().max() - temp.restrict(
                    new_interval).as_series().min()) > min_dist:  # min distance in cm that the trial is getting
                trials2keep.append(True)
                start2[a], stop2[a] = tempt[0], tempt[-1]
            else:
                trials2keep.append(False)
        else:
            trials2keep.append(False)

    right_runs_interval = nap.IntervalSet(start=start2[trials2keep], end=stop2[trials2keep])
    return left_runs_interval, right_runs_interval


def trials_projection(proj_df, intervals):
    """Function to extract trials periods from the projected data. Useful for plotting"""

    # gcPCA projection on every left trial run
    for c, a in enumerate(intervals.values):
        temp_is = nap.IntervalSet(a[0], end=a[1])
        tmpr = proj_df.restrict(temp_is)
        tempdf = tmpr.as_dataframe()
        tempdf.columns = {'dim1', 'dim2'}

        # append a new column to dataframe with trial information
        tempdf.insert(loc=2,
                      column='trial',
                      value=c * np.ones((tmpr.shape[0], 1)))
        if c == 1:
            trials_proj = tempdf
        else:
            trials_proj = trials_proj.append(tempdf)

    return trials_proj

from matplotlib.collections import LineCollection

# functions for plotting
# sets of functions to make the line plot using colormap
def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# Interface to LineCollection:
def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3,
              alpha=1.0, ax=plt.gca()):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)

    return lc

# defining function to plot cPCA trajectories in each subplot
def plot_pc_trajectories(ax, x, y, color_edge, color_dot, alpha, pos_data, trial_interval, air_puff_loc, colormap):
    colorline(x, y, ax=ax, linewidth=1, alpha=alpha, cmap=colormap)
    ax.scatter(x[0], y[0], s=mrkr_size, c=color_dot[0], zorder=5, alpha=0.9,
                      edgecolors='k')
    ax.scatter(x[-1], y[-1], s=mrkr_size, c=color_dot[1], zorder=10, alpha=0.9,
                      edgecolors='k')
    # finding location index to plot
    temp_pos = pos_data.restrict(trial_interval)
    I = np.argmin(np.abs(temp_pos - air_puff_loc))
    if I < len(x):  # location of air puff was getting where gcPCA is nan
        ax.scatter(x[I], y[I], s=mrkr_size, c=color_edge[0],
                          edgecolors=color_edge[1], zorder=15, alpha=0.9)
    return ax

# x and y label adjustments
def plot_pc_adjustments(ax,xlabel,ylabel):
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.set_yticks([])
    ax.set_xticks([])
    return ax


# making custom gradient for plots
from matplotlib.colors import LinearSegmentedColormap

custom_gradient = LinearSegmentedColormap.from_list('custom_gradient', (
    (0.000, (0.0, 0.80, 0.80)),
    (0.250, (0.0, 0.65, 0.65)),
    (0.500, (1, 1, 1)),
    (0.750, (0.65, 0.0, 0.65)),
    (1.000, (0.80, 0, 0.80))))

# %% load the data structure
data_dict = mat73.loadmat(base_dir + 'hpc_bla_gg_dataset.mat')

# loop over sessions
session_list = np.arange(start=35, stop=36)
for ses in session_list:
    air_puff_times = data_dict['hpc_bla_gg_dataset']['air_puff_times'][ses]
    if len(air_puff_times) > 10:  # only run the session if it has 10 air puffs
        pos = data_dict['hpc_bla_gg_dataset']['location'][ses]  # position
        linspd = data_dict['hpc_bla_gg_dataset']['linspd'][ses]  # running speed
        pos_t = data_dict['hpc_bla_gg_dataset']['tracking_t'][ses]  # position time vector
        spikes = data_dict['hpc_bla_gg_dataset']['spikes_ts'][ses]  # spikes
        region = data_dict['hpc_bla_gg_dataset']['spikes_region'][ses]  # brain region

        if np.char.equal(region, 'hpc').sum() > min_n_cell:

            # getting time intervals of the session
            run_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['run_intervals']
            pre_sws_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['pre_sws_intervals']
            post_sws_temp = data_dict['hpc_bla_gg_dataset']['task'][ses]['post_sws_intervals']

            nap_pos = nap.Tsd(pos_t, d=pos[:, 0], time_units="s")
            nap_spd = nap.Tsd(pos_t, d=linspd, time_units="s")
            nap_air_puff = nap.Ts(air_puff_times, time_units="ms")
            run_intervals = nap.IntervalSet(start=run_temp[0], end=run_temp[1])
            pre_sws_intervals = nap.IntervalSet(start=pre_sws_temp[0], end=pre_sws_temp[1])
            post_sws_intervals = nap.IntervalSet(start=post_sws_temp[0], end=post_sws_temp[1])

            # preparing spikes
            cells_hpc = np.argwhere([np.char.equal(region, 'hpc')])
            if cells_hpc.size > 0:
                spks_times = {}
                for c, a in enumerate(cells_hpc[:, 1]):
                    spks_times[c] = spikes[a]

                # passing to pynapple ts group
                spikes_times = nap.TsGroup(spks_times)

                # separating in pre/run
                temp_r = spikes_times.restrict(run_intervals)  # run
                temp_presws = spikes_times.restrict(pre_sws_intervals)  # pre run SWS
                temp_postsws = spikes_times.restrict(post_sws_intervals)  # post run SWS

                # picking cells based on firing rate
                cells2keep = (temp_r.rates.values > min_fr)
                if sum(cells2keep) > min_n_cell:

                    # normalizing and smoothing data - run periods
                    temp_data = zscore(temp_r.count(bin_size).as_dataframe().rolling(window=wind_cov,
                                win_type='gaussian', center=True, min_periods=1).mean(std=std_conv).values)
                    run_time = np.array(temp_r.count(bin_size).index)
                    run_data = temp_data[:, cells2keep].copy()

                    # normalizing and smoothing data - sleep periods
                    temp_data = zscore(temp_presws.count(bin_size).as_dataframe().rolling(window=wind_cov,
                                win_type='gaussian', center=True, min_periods=1).mean(std=std_conv).values)
                    presws_time = np.array(temp_presws.count(bin_size).index)
                    presws_data = temp_data[:, cells2keep].copy()

                    # normalizing and smoothing data - sleep periods
                    temp_data = zscore(temp_postsws.count(bin_size).as_dataframe().rolling(window=wind_cov,
                                win_type='gaussian', center=True, min_periods=1).mean(std=std_conv).values)
                    postsws_time = np.array(temp_postsws.count(bin_size).index)
                    postsws_data = temp_data[:, cells2keep].copy()


                    run_pos = nap_pos.restrict(run_intervals).as_series()
                    run_spd = nap_spd.restrict(run_intervals).as_series()

                    # extracting left and right runs trials
                    temp = nap.Tsd(np.array(run_pos.index), savgol_filter(np.array(run_pos.values), 300, 3))
                    left_runs_int2, right_runs_int2 = extract_trials(temp, nap_spd)

                    # merging close intervals if necessary
                    left_runs_int = left_runs_int2.merge_close_intervals(threshold=1)
                    right_runs_int = right_runs_int2.merge_close_intervals(threshold=1)

                    # identifying which run is safe or dangerous
                    n_r = len(nap_air_puff.restrict(right_runs_int))
                    n_l = len(nap_air_puff.restrict(left_runs_int))
                    if n_l > n_r:
                        runs_type = {'left': 'danger', 'right': 'safe'}
                    else:
                        runs_type = {'left': 'safe', 'right': 'danger'}


                    # normalizing data for contrastive methods
                    N1 = zscore(postsws_data)
                    N1 = N1 / np.linalg.norm(N1, axis=0)
                    N2 = zscore(presws_data)
                    N2 = N2 / np.linalg.norm(N2, axis=0)

                    # normalizing run data
                    norm_run_data = zscore(run_data)

                    # gcPCA fitting
                    gcpca_mdl = gcPCA(method='v4')
                    gcpca_mdl.fit(N1, N2)

                    # gcPCA projection run data
                    run_gcpca = nap.TsdFrame(run_time, d=norm_run_data.dot(gcpca_mdl.loadings_[:, :2]))
                    run_gcpca_last = nap.TsdFrame(run_time, d=norm_run_data.dot(gcpca_mdl.loadings_[:, -2:]))

                    # cPCA gitting
                    # cPCA all components
                    cPCA_mdl = CPCA(n_components=postsws_data.shape[1])
                    cPCA_mdl.fit(N1, N2)
                    # cPCA 2 components
                    cPCA_mdl2 = CPCA(n_components=2)
                    cPCA_mdl2.fit(N1, N2)
                    np.random.seed(0)  # for reproducibility

                    # automated alpha finding for the different cPCA models
                    _, best_alphas_allPCs = cPCA_mdl.automated_cpca(N1, n_alphas=40, max_log_alpha=4,
                                                                    n_alphas_to_return=3)
                    _, best_alphas_2PCs = cPCA_mdl2.automated_cpca(N1, n_alphas=40, max_log_alpha=4,
                                                                   n_alphas_to_return=3)
                    # cPCA projection run data
                    projected_data_allPCs = []
                    for bbb in best_alphas_allPCs[1:]:
                        projected_data_allPCs.append(
                            cPCA_mdl.transform(norm_run_data, alpha_selection='manual', alpha_value=bbb))

                    projected_data_2PCs = []
                    for bbb in best_alphas_2PCs[1:]:
                        projected_data_2PCs.append(
                            cPCA_mdl2.transform(norm_run_data, alpha_selection='manual', alpha_value=bbb))

                    # passing to TsdFrame
                    run_CPCA_allpcs = []
                    for bbb in range(len(projected_data_allPCs)):
                        run_CPCA_allpcs.append(nap.TsdFrame(run_time, d=projected_data_allPCs[bbb]))

                    run_CPCA_2pcs = []
                    for bbb in range(len(projected_data_2PCs)):
                        run_CPCA_2pcs.append(nap.TsdFrame(run_time, d=projected_data_2PCs[bbb]))

                    run_pca = nap.TsdFrame(run_time, d=run_data.dot(V[:2, :].T))

                    #setting plot color according to whether run is safe or dange
                    if runs_type['left'] == 'safe':
                        color_l = 'k'
                        color_r = 'k'
                        color_ap_edge_r = ['firebrick', 'k']
                        color_ap_edge_l = ['white', 'orchid']
                        color_r_dots = ['whitesmoke', 'dimgray']
                        color_l_dots = ['whitesmoke', 'dimgray']

                    else:
                        color_l = 'k'
                        color_r = 'k'
                        color_ap_edge_l = ['firebrick', 'k']
                        color_ap_edge_r = ['white', 'orchid']
                        color_l_dots = ['whitesmoke', 'dimgray']
                        color_r_dots = ['whitesmoke', 'dimgray']

                    # find the airpuff in the position
                    temp_ap = nap.IntervalSet(start=nap_air_puff.index - 0.05, end=nap_air_puff.index + 0.05)
                    air_puff_loc = nap_pos.restrict(temp_ap).as_series().median()

                    # interpolating the run gcPCA data to match the position
                    # first dimensions
                    dim1 = np.interp(run_pos.index, run_gcpca.index, run_gcpca.values[:, 0])
                    dim2 = np.interp(run_pos.index, run_gcpca.index, run_gcpca.values[:, 1])
                    newd = np.concatenate((dim1[:, np.newaxis], dim2[:, np.newaxis]), axis=1)
                    new_r_gcpca = nap.TsdFrame(np.array(run_pos.index), d=newd)

                    # last dimensions
                    dim1 = np.interp(run_pos.index, run_gcpca_last.index,
                                     run_gcpca_last.values[:, 1])  # this is because the -2: indexing
                    dim2 = np.interp(run_pos.index, run_gcpca_last.index, run_gcpca_last.values[:, 0])
                    newd = np.concatenate((dim1[:, np.newaxis], dim2[:, np.newaxis]), axis=1)
                    new_run_gcpca_last = nap.TsdFrame(np.array(run_pos.index), d=newd)

                    # interpolating the CPCA multiple alpha to match the position
                    new_run_CPCA_allPCs_first = []
                    new_run_CPCA_2PCs_first = []
                    new_run_CPCA_allPCs_last = []
                    for bbb in range(len(run_CPCA_allpcs)):
                        dim1 = np.interp(run_pos.index, run_CPCA_allpcs[bbb].index, run_CPCA_allpcs[bbb].values[:, 0])
                        dim2 = np.interp(run_pos.index, run_CPCA_allpcs[bbb].index, run_CPCA_allpcs[bbb].values[:, 1])
                        newd = np.concatenate((dim1[:, np.newaxis], dim2[:, np.newaxis]), axis=1)
                        new_run_CPCA_allPCs_first.append(nap.TsdFrame(np.array(run_pos.index), d=newd))

                        dim_m1 = np.interp(run_pos.index, run_CPCA_allpcs[bbb].index, run_CPCA_allpcs[bbb].values[:, -1])
                        dim_m2 = np.interp(run_pos.index, run_CPCA_allpcs[bbb].index, run_CPCA_allpcs[bbb].values[:, -2])
                        newd = np.concatenate((dim_m1[:, np.newaxis], dim_m2[:, np.newaxis]), axis=1)
                        new_run_CPCA_allPCs_last.append(nap.TsdFrame(np.array(run_pos.index), d=newd))

                        dim1 = np.interp(run_pos.index, run_CPCA_2pcs[bbb].index, run_CPCA_2pcs[bbb].values[:, 0])
                        dim2 = np.interp(run_pos.index, run_CPCA_2pcs[bbb].index, run_CPCA_2pcs[bbb].values[:, 1])
                        newd = np.concatenate((dim1[:, np.newaxis], dim2[:, np.newaxis]), axis=1)
                        new_run_CPCA_2PCs_first.append(nap.TsdFrame(np.array(run_pos.index), d=newd))

                    # setting parameters rc to allow better pdf fonttype handling
                    plt.rcParams['pdf.fonttype'] = 42
                    plt.rcParams['ps.fonttype'] = 42

                    ###
                    # plot gcPCA first dimensions
                    cmap_str_danger = 'seismic'
                    cmap_str_safe = custom_gradient
                    fig, (axs) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(5, 5))
                    for a in left_runs_int.values[1:20, :]:  # plotting only the first 20 trials for better visualization
                        temp_is = nap.IntervalSet(a[0], end=a[1])
                        tempdf = new_r_gcpca.restrict(temp_is).as_dataframe().rolling(roll_wind, center=True).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                        axs[0, 0] = plot_pc_trajectories(axs[0, 0], x, y, color_ap_edge_l, color_l_dots, 0.5, nap_pos, temp_is, air_puff_loc, cmap_str_danger)
                        axs[0, 0] = plot_pc_adjustments(axs[0, 0], 'gcPC$_{1}$','gcPC$_{2}$')

                    for a in right_runs_int.values[1:20, :]:
                        temp_is = nap.IntervalSet(a[0], end=a[1])
                        tempdf = new_r_gcpca.restrict(temp_is).as_dataframe().rolling(roll_wind, center=True).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                        axs[1, 0] = plot_pc_trajectories(axs[1, 0], x, y, color_ap_edge_r ,color_r_dots, 0.5, nap_pos, temp_is, air_puff_loc, cmap_str_safe)
                        axs[1, 0] = plot_pc_adjustments(axs[1, 0], 'gcPC$_{1}$', 'gcPC$_{2}$')

                    ###
                    # plot gcPCA last dimensions
                    for a in left_runs_int.values[1:20, :]:
                        temp_is = nap.IntervalSet(a[0], end=a[1])
                        tempdf = new_run_gcpca_last.restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                             center=True).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                        axs[0, 1] = plot_pc_trajectories(axs[0, 1], x, y,color_ap_edge_l, color_l_dots, 0.5, nap_pos, temp_is,air_puff_loc, cmap_str_danger)
                        axs[0, 1] = plot_pc_adjustments(axs[0, 1], 'gcPC$_{last}$', 'gcPC$_{last-1}$')

                    for a in right_runs_int.values[1:20, :]:
                        temp_is = nap.IntervalSet(a[0], end=a[1])
                        tempdf = new_run_gcpca_last.restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                             center=True).mean()
                        x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0]
                        y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                        axs[1, 1] = plot_pc_trajectories(axs[1, 1], x, y, color_ap_edge_r, color_r_dots, 0.5, nap_pos, temp_is, air_puff_loc, cmap_str_safe)
                        axs[1, 1] = plot_pc_adjustments(axs[1, 1], 'gcPC$_{last}$', 'gcPC$_{last-1}$')
                    fig.subplots_adjust(left=0.1, top=0.90, right=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
                    fig.savefig(save_fig_dir + ses.astype(str) + "gcPCA_space_PRE_POST_v2.pdf", transparent=True)

                    #####################################################
                    # making a plot specifically for different alphas CPCA
                    cpc_str_xlabel = ['cPC$_1$', 'cPC$_{last}$']
                    cpc_str_ylabel = ['cPC$_2$', 'cPC$_{last-1}$']
                    plt.rcParams['font.size'] = 16
                    for bbb in range(len(new_run_CPCA_allPCs_first)):
                        fig, (axs) = plt.subplots(2, 2, figsize=(5, 5), sharex=True, sharey=True)
                        for a in left_runs_int.values[1:20, :]:
                            temp_is = nap.IntervalSet(a[0], end=a[1])
                            tempdf = new_run_CPCA_allPCs_first[bbb].restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                                center=True).mean()
                            x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0] * -1  # for flipping the sign
                            y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                            axs[0, 0] = plot_pc_trajectories(axs[0, 0], x, y, color_ap_edge_l, color_l_dots, 0.5, nap_pos, temp_is, air_puff_loc, cmap_str_danger)
                            axs[0, 0] = plot_pc_adjustments(axs[0, 0], cpc_str_xlabel[0],  cpc_str_ylabel[0])

                        for a in right_runs_int.values[1:20, :]:
                            temp_is = nap.IntervalSet(a[0], end=a[1])
                            tempdf = new_run_CPCA_allPCs_first[bbb].restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                                center=True).mean()
                            x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0] * -1  # for flipping the sign
                            y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                            axs[1, 0] = plot_pc_trajectories(axs[1, 0], x, y, color_ap_edge_r, color_r_dots, 0.5, nap_pos,
                                                         temp_is, air_puff_loc, cmap_str_safe)
                            axs[1, 0] = plot_pc_adjustments(axs[1, 0], cpc_str_xlabel[0], cpc_str_ylabel[0])

                        for a in left_runs_int.values[1:20, :]:
                            temp_is = nap.IntervalSet(a[0], end=a[1])
                            tempdf = new_run_CPCA_allPCs_last[bbb].restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                                center=True).mean()
                            x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0] * -1  # for flipping the sign
                            y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                            axs[0, 1] = plot_pc_trajectories(axs[0, 1], x, y, color_ap_edge_l, color_l_dots, 0.5, nap_pos, temp_is, air_puff_loc, cmap_str_danger)
                            axs[0, 1] = plot_pc_adjustments(axs[0, 1], cpc_str_xlabel[1],  cpc_str_ylabel[1])

                        for a in right_runs_int.values[1:20, :]:
                            temp_is = nap.IntervalSet(a[0], end=a[1])
                            tempdf = new_run_CPCA_allPCs_last[bbb].restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                                center=True).mean()
                            x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0] * -1  # for flipping the sign
                            y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                            axs[1, 1] = plot_pc_trajectories(axs[1, 1], x, y, color_ap_edge_r, color_r_dots, 0.5, nap_pos,
                                                         temp_is, air_puff_loc, cmap_str_safe)
                            axs[1, 1] = plot_pc_adjustments(axs[1, 1], cpc_str_xlabel[1], cpc_str_ylabel[1])

                        fig.subplots_adjust(left=0.1, top=0.90, right=0.95, bottom=0.1, hspace=0.2, wspace=0.2)
                        fig.suptitle('k='+str(postsws_data.shape[1])+', '+ r'$\alpha$ = ' + "{:.2f}".format(best_alphas_allPCs[bbb + 1]), x=0.55, y=0.97,
                                     fontweight='bold')
                        fig.savefig(save_fig_dir + ses.astype(str) + "PRE_POST_all_cPCs_RUN_v2_alpha_" + str(
                            best_alphas_allPCs[bbb + 1]) + ".pdf", transparent=True)

                    ###
                    # plot of the 2 cPCs model
                    for bbb in range(len(new_run_CPCA_2PCs_first)):
                        fig, (axs) = plt.subplots(2, 2, figsize=(5, 5), sharex=True, sharey=True)
                        for a in left_runs_int.values[1:20, :]:
                            temp_is = nap.IntervalSet(a[0], end=a[1])
                            tempdf = new_run_CPCA_2PCs_first[bbb].restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                                center=True).mean()
                            x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0] * -1  # for flipping the sign
                            y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                            axs[0, 0] = plot_pc_trajectories(axs[0, 0], x, y, color_ap_edge_l, color_l_dots, 0.5, nap_pos, temp_is, air_puff_loc, cmap_str_danger)
                            axs[0, 0] = plot_pc_adjustments(axs[0, 0], cpc_str_xlabel[0],  cpc_str_ylabel[0])

                        for a in right_runs_int.values[1:20, :]:
                            temp_is = nap.IntervalSet(a[0], end=a[1])
                            tempdf = new_run_CPCA_2PCs_first[bbb].restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                                                center=True).mean()
                            x = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 0])), 0] * -1  # for flipping the sign
                            y = tempdf.values[np.logical_not(np.isnan(tempdf.values[:, 1])), 1]
                            axs[1, 0] = plot_pc_trajectories(axs[1, 0], x, y, color_ap_edge_r, color_r_dots, 0.5, nap_pos,
                                                         temp_is, air_puff_loc, cmap_str_safe)
                            axs[1, 0] = plot_pc_adjustments(axs[1, 0], cpc_str_xlabel[0], cpc_str_ylabel[0])

                        # changing colors of matplotlib spines to white using a for loop
                        for spine in axs[0, 1].spines.values():
                            spine.set_edgecolor('white')
                        for spine in axs[1, 1].spines.values():
                            spine.set_edgecolor('white')

                        fig.subplots_adjust(left=0.1, top=0.90, right=0.95, bottom=0.1, hspace=0.2, wspace=0.2)
                        fig.suptitle('k=2, ' + r'$\alpha$ = ' + "{:.2f}".format(best_alphas_2PCs[bbb + 1]), x=0.29, y=0.97,
                                     fontweight='bold')
                        fig.savefig(save_fig_dir + ses.astype(str) + "PRE_POST_2_cPCs_RUN_v2_alpha_" + str(
                            best_alphas_2PCs[bbb + 1]) + ".pdf", transparent=True)
                    ###
                    # make raster plots of each condition and run
                    plt.rcParams['font.size'] = 16
                    pre_sws_interval_plot = nap.IntervalSet(start=pre_sws_temp[0], end=pre_sws_temp[0]+50)
                    post_sws_interval_plot = nap.IntervalSet(start=post_sws_temp[0], end=post_sws_temp[0]+50)
                    task_interval_plot = nap.IntervalSet(start=run_temp[0], end=run_temp[0]+50)
                    temp_spks_pre = spikes_times.restrict(pre_sws_interval_plot)
                    temp_spks_post = spikes_times.restrict(post_sws_interval_plot)
                    temp_spks_task = spikes_times.restrict(task_interval_plot)


                    #pre-task raster plot
                    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
                    for a in range(len(cells_hpc[cells2keep])):
                        axs.eventplot(temp_spks_pre[a].index,lineoffsets=a, color='k', linewidths=0.5, linelengths=0.5)
                    axs.set_title('Pre-task Recordings')
                    # axs.set_xticks([])
                    axs.set_ylabel('Neurons')
                    plt.tight_layout()
                    plt.savefig(save_fig_dir + ses.astype(str) + "_pre_task_raster.pdf", transparent=True)

                    # post-task raster plot
                    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
                    for a in range(len(cells_hpc[cells2keep])):
                        axs.eventplot(temp_spks_post[a].index, lineoffsets=a, color='k', linewidths=0.5, linelengths=0.5)
                    axs.set_title('Post-task Recordings')
                    # axs.set_xticks([])
                    axs.set_ylabel('Neurons')
                    plt.tight_layout()
                    plt.savefig(save_fig_dir + ses.astype(str) + "_post_task_raster.pdf", transparent=True)

                    # task plot
                    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
                    for a in range(len(cells_hpc[cells2keep])):
                        axs.eventplot(temp_spks_task[a].index, lineoffsets=a, color='k', linewidths=0.5, linelengths=0.5)
                    axs.set_title('Task')
                    # axs.set_xticks([])
                    axs.set_ylabel('Neurons')
                    plt.tight_layout()
                    plt.savefig(save_fig_dir + ses.astype(str) + "_task_raster.pdf", transparent=True)
plt.show()