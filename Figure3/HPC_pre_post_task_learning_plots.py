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
# import matplotlib as mpl    # DELETE THIS LATER
# mpl.use('TkAgg')    # DELETE THIS LATER

from contrastive import CPCA
from scipy.stats import zscore
from scipy.signal import savgol_filter
from scipy.io import loadmat

# parameters for processing
min_n_cell = 30  # min number of cells in the brain area to be used
min_fr = 0.01  # minimum firing rate to be included in the analysis
bin_size = 0.01  # 10 ms
std_conv = 2  # standard deviation for convolution (in num bins units)
wind_cov = 5  # window size for convolution (in num bins units)
roll_wind = 20  # window size for rolling mean

# parameters for plots
mrkr_size = 20
plt.rcParams.update({'figure.dpi':150,'font.size': 12,
                     'font.family':'Arial',
                     'pdf.fonttype':42, 'ps.fonttype':42})

# import custom modules
# change repo_dir to your gcPCA directory
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

# other directories
# directory with the HPC data
base_dir = '/home/eliezyer/Documents/github/deOliveira_gcPCA_2024/Figure3/'

# directory where to save the figures
save_fig_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/figure_manuscripts/figures_gcPCA/Figure3/source_plots/'


# defining ancillar functions
def extract_trials(smooth_pos):
    """Function to extract running trials from GG dataset where animals
    run in a linear track with airpuff. It returns left and right runs (trials) in the maze
    """

    max_len = 10  # max length of trial, in seconds
    min_len = 0.5  # min length of trial, in seconds
    min_clip, max_clip = 150, 450  # this is for clipping the trials in space
    min_dist = 150  # in cm (avoid runs where animal goes to half and back)
    min_speed = 3  # minimal speed is 3 cm/s in the trial

    # running speed
    speed = nap.Tsd(np.array(smooth_pos.index[:-1]), np.abs(np.diff(smooth_pos.values)) / np.diff(smooth_pos.index))
    logical_left = np.diff(smooth_pos.as_series().clip(lower=min_clip, upper=max_clip).values) < 0
    logical_right = np.diff(smooth_pos.as_series().clip(lower=min_clip, upper=max_clip).values) > 0

    logical_left = np.append(logical_left, False)
    logical_right = np.append(logical_right, False)
    logical_left = np.insert(logical_left, 0, False)
    logical_right = np.insert(logical_right, 0, False)
    ts = smooth_pos.as_series().clip(lower=min_clip, upper=max_clip).index

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
        tempt = speed.restrict(interval).index[speed.restrict(interval).values > min_speed]  # this is ensuring the running speed is reaching min
        if tempt.size > 0:
            new_interval = nap.IntervalSet(start=tempt[0], end=tempt[-1])
            if (smooth_pos.restrict(new_interval).as_series().max() - smooth_pos.restrict(
                    new_interval).as_series().min()) > min_dist:  # min distance in cm that the trial is getting
                trials2keep.append(True)
                start2[a], stop2[a] = tempt[0], tempt[-1]
            else:
                trials2keep.append(False)
        else:
            trials2keep.append(False)
    start3 = start2[trials2keep].copy()
    stop3 = stop2[trials2keep].copy()
    left_runs_interval = nap.IntervalSet(start=start3, end=stop3)

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
            if (smooth_pos.restrict(new_interval).as_series().max() - smooth_pos.restrict(
                    new_interval).as_series().min()) > min_dist:  # min distance in cm that the trial is getting
                trials2keep.append(True)
                start2[a], stop2[a] = tempt[0], tempt[-1]
            else:
                trials2keep.append(False)
        else:
            trials2keep.append(False)

    start3 = start2[trials2keep].copy()
    stop3 = stop2[trials2keep].copy()
    right_runs_interval = nap.IntervalSet(start=start3, end=stop3)

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
def colorline(x, y, t=None, min_max_t=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3,
              alpha=1.0, ax=plt.gca()):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if t is None:
        t = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if (t is not None) and (min_max_t is not None):
        t = t - min_max_t[0]
        t = t / (min_max_t[1] - min_max_t[0])
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=t, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)

    return lc

# defining function to interpolate data to match the position
def interpolate_data(run_pos, run_data):
    dim1 = np.interp(run_pos.index, run_data.index, run_data.values[:, 0])
    dim2 = np.interp(run_pos.index, run_data.index, run_data.values[:, 1])
    newd = np.concatenate((dim1[:, np.newaxis], dim2[:, np.newaxis]), axis=1)
    new_run_data = nap.TsdFrame(np.array(run_pos.index), d=newd)
    return new_run_data


# defining function to plot cPCA trajectories in each subplot
def plot_pc_trajectories(ax, x, y, t, min_max_t, color_edge, color_dot, alpha, pos_data, trial_interval, air_puff_loc, colormap,linewidth=1, plot_init=True, plot_ap=True):
    colorline(x, y, t, min_max_t, ax=ax, linewidth=linewidth, alpha=alpha, cmap=colormap)
    if plot_init==True:
        ax.scatter(x[0], y[0], s=mrkr_size, c=color_dot[0], zorder=5, alpha=0.8,
                   edgecolors='k')
        ax.scatter(x[-1], y[-1], s=mrkr_size, c=color_dot[1], zorder=10, alpha=0.8,
                   edgecolors='k')
    # finding location index to plot
    if plot_ap==True:
        temp_pos = pos_data.restrict(trial_interval)
        I = np.argmin(np.abs(temp_pos - air_puff_loc))
        if I < len(x):  # location of air puff was getting where gcPCA is nan
            ax.scatter(x[I], y[I], s=mrkr_size, c=color_edge[0],
                       edgecolors=color_edge[1], zorder=15, alpha=0.8)
    return ax

def prepare_and_plot(ax, interval, projection, min_max_t, roll_wind, position, ap_location, color_ap_edge, color_dots,xlabel_str, ylabel_str, cmap2use, plot_init=True, plot_ap=True):
    temp_is = nap.IntervalSet(interval[0], end=interval[1])
    tempdf = projection.restrict(temp_is).as_dataframe().rolling(roll_wind,
                                                                      center=True).mean()
    temp_pos = position.restrict(temp_is).values
    not_nan = np.logical_not(np.isnan(np.sum(tempdf.values, axis=1)))
    x = tempdf.values[not_nan, 0]
    y = tempdf.values[not_nan, 1]
    position_trial = temp_pos[not_nan]
    ax = plot_pc_trajectories(ax, x, y, position_trial, min_max_t, color_ap_edge, color_dots, 0.8, position,
                                     temp_is, ap_location, cmap2use,1, plot_init, plot_ap)
    ax = plot_pc_adjustments(ax, xlabel=xlabel_str, ylabel=ylabel_str)


# x and y label adjustments
def plot_pc_adjustments(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_yticks([])
    ax.set_xticks([])
    return ax


# making custom gradient for plots
from matplotlib.colors import LinearSegmentedColormap

custom_gradient = LinearSegmentedColormap.from_list('custom_gradient', (
    (0.000, (0.0, 0.80, 0.80)),
    (0.250, (0.0, 0.65, 0.65)),
    (0.500, (0.85, 0.85, 0.85)),
    (0.750, (0.65, 0.0, 0.65)),
    (1.000, (0.80, 0, 0.80))))

custom_gradient_danger = LinearSegmentedColormap.from_list('custom_gradient_danger', (
    (0.000, (0.0, 0.0, 0.65)),
    (0.250, (0.0, 0.0, 0.90)),
    (0.500, (0.85, 0.85, 0.85)),
    (0.750, (0.90, 0.0, 0.0)),
    (1.000, (0.65, 0, 0.0))))

# %% load the data structure
data_dict = loadmat(base_dir + 'hpc_gg_session.mat')

air_puff_times = data_dict['hpc_gg_session']['air_puff_times'][0][0]  # air puff times
if len(air_puff_times) > 10:  # only run the analysis if it has at least 10 air puffs events
    pos = data_dict['hpc_gg_session']['location'][0][0]  # position
    linspd = data_dict['hpc_gg_session']['linspd'][0][0]  # running speed
    pos_t = data_dict['hpc_gg_session']['tracking_t'][0][0]  # position time vector
    spikes = data_dict['hpc_gg_session']['spikes_ts'][0][0]  # spikes
    region = data_dict['hpc_gg_session']['spikes_region'][0]  # brain region
    # passion regions to numpy
    region = np.array([a[0] for a in region[0][0]])

    if np.char.equal(region, 'hpc').sum() > min_n_cell:

        # getting time intervals of the session
        run_temp = data_dict['hpc_gg_session']['task'][0][0]['run_intervals'][0][0][0]
        pre_task_temp = data_dict['hpc_gg_session']['task'][0][0]['pre_task_intervals'][0][0][0]
        post_task_temp = data_dict['hpc_gg_session']['task'][0][0]['post_task_intervals'][0][0][0]

        nap_pos = nap.Tsd(pos_t[:, 0], d=pos[:, 0], time_units="s")
        nap_spd = nap.Tsd(pos_t[:, 0], d=linspd[:, 0], time_units="s")
        nap_air_puff = nap.Ts(air_puff_times[:, 0], time_units="ms")
        run_intervals = nap.IntervalSet(start=run_temp[0], end=run_temp[1])
        pre_task_intervals = nap.IntervalSet(start=pre_task_temp[0], end=pre_task_temp[1])
        post_task_intervals = nap.IntervalSet(start=post_task_temp[0], end=post_task_temp[1])

        # preparing spikes
        cells_hpc = np.argwhere([np.char.equal(region, 'hpc')])
        if len(cells_hpc[:, 1]) > 0:
            spks_times = {}
            for c, a in enumerate(cells_hpc[:, 1]):
                spks_times[c] = spikes[0][a][:,0]

            # passing to pynapple ts group
            spikes_times = nap.TsGroup(spks_times)

            # separating in pre/run
            temp_r = spikes_times.restrict(run_intervals)  # run
            temp_pre_task = spikes_times.restrict(pre_task_intervals)  # pre run home cage
            temp_post_task = spikes_times.restrict(post_task_intervals)  # post run home cage

            # picking cells based on firing rate
            cells2keep = (temp_r.rates.values > min_fr)
            if sum(cells2keep) > min_n_cell:

                # normalizing and smoothing data - run periods
                temp_data = zscore(temp_r.count(bin_size).as_dataframe().rolling(window=wind_cov,
                                                                                 win_type='gaussian', center=True,
                                                                                 min_periods=1).mean(
                    std=std_conv).values)
                run_time = np.array(temp_r.count(bin_size).index)
                run_data = temp_data[:, cells2keep].copy()

                # normalizing and smoothing data - pre task periods
                temp_data = zscore(temp_pre_task.count(bin_size).as_dataframe().rolling(window=wind_cov,
                                                                                        win_type='gaussian', center=True,
                                                                                        min_periods=1).mean(std=std_conv).values)
                pre_task_time = np.array(temp_pre_task.count(bin_size).index)
                pre_task_data = temp_data[:, cells2keep].copy()

                # normalizing and smoothing data - post task periods
                temp_data = zscore(temp_post_task.count(bin_size).as_dataframe().rolling(window=wind_cov,
                                                                                         win_type='gaussian', center=True,
                                                                                         min_periods=1).mean(std=std_conv).values)
                post_task_time = np.array(temp_post_task.count(bin_size).index)
                post_task_data = temp_data[:, cells2keep].copy()

                run_pos = nap_pos.restrict(run_intervals).as_series()
                run_spd = nap_spd.restrict(run_intervals).as_series()

                # extracting left and right runs trials
                smooth_pos = nap.Tsd(np.array(run_pos.index), savgol_filter(np.array(run_pos.values), 300, 3))
                # left_runs_int2, right_runs_int2 = extract_trials(nap_pos.restrict(run_intervals), smooth_pos)
                left_runs_int2, right_runs_int2 = extract_trials(smooth_pos)
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
                N_post = zscore(post_task_data)
                N_post = N_post / np.linalg.norm(N_post, axis=0)
                N_pre = zscore(pre_task_data)
                N_pre = N_pre / np.linalg.norm(N_pre, axis=0)

                # normalizing run data
                N_task = zscore(run_data)
                N_task = N_task / np.linalg.norm(N_task, axis=0)

                # gcPCA fitting
                gcpca_mdl = gcPCA(method='v4')
                gcpca_mdl.fit(N_post, N_pre)

                # gcPCA projection run data
                run_gcpca = nap.TsdFrame(run_time, d=N_task.dot(gcpca_mdl.loadings_[:, :2]))

                # PCA fitting
                _, _, VtPre = np.linalg.svd(N_pre, full_matrices=False)
                _, _, VtTask = np.linalg.svd(N_task, full_matrices=False)
                _, _, VtPost = np.linalg.svd(N_post, full_matrices=False)

                # PCA projection run data
                run_PCA_pre = nap.TsdFrame(run_time, d=N_task.dot(VtPre[:2].T))
                run_PCA_task = nap.TsdFrame(run_time, d=N_task.dot(VtTask[:2].T))
                run_PCA_post = nap.TsdFrame(run_time, d=N_task.dot(VtPost[:2].T))


                # cPCA gitting
                np.random.seed(0)  # for reproducibility
                # cPCA all components
                cPCA_mdl = CPCA(n_components=post_task_data.shape[1])
                cPCA_mdl.fit(N_post, N_pre)
                # cPCA 2 components
                cPCA_mdl2 = CPCA(n_components=2)
                cPCA_mdl2.fit(N_post, N_pre)


                # automated alpha finding for the different cPCA models
                _, best_alphas_allPCs = cPCA_mdl.automated_cpca(N_post, n_alphas=40, max_log_alpha=4,
                                                                n_alphas_to_return=3)
                _, best_alphas_2PCs = cPCA_mdl2.automated_cpca(N_post, n_alphas=40, max_log_alpha=4,
                                                               n_alphas_to_return=3)
                # cPCA projection run data
                projected_data_allPCs = []
                for bbb in best_alphas_allPCs[1:]:
                    projected_data_allPCs.append(
                        cPCA_mdl.transform(N_task, alpha_selection='manual', alpha_value=bbb))

                projected_data_2PCs = []
                for bbb in best_alphas_2PCs[1:]:
                    projected_data_2PCs.append(
                        cPCA_mdl2.transform(N_task, alpha_selection='manual', alpha_value=bbb))

                # passing to TsdFrame
                run_CPCA_allpcs = []
                for bbb in range(len(projected_data_allPCs)):
                    run_CPCA_allpcs.append(nap.TsdFrame(run_time, d=projected_data_allPCs[bbb]))

                run_CPCA_2pcs = []
                for bbb in range(len(projected_data_2PCs)):
                    run_CPCA_2pcs.append(nap.TsdFrame(run_time, d=projected_data_2PCs[bbb]))

                # setting plot color according to whether run is safe or dange
                if runs_type['left'] == 'safe':
                    color_l = 'k'
                    color_r = 'k'
                    color_ap_edge_r = ['firebrick', 'k']
                    color_ap_edge_l = ['orchid', 'k']
                    color_r_dots = ['whitesmoke', 'dimgray']
                    color_l_dots = ['whitesmoke', 'dimgray']

                else:
                    color_l = 'k'
                    color_r = 'k'
                    color_ap_edge_l = ['firebrick', 'k']
                    color_ap_edge_r = ['orchid', 'k']
                    color_l_dots = ['whitesmoke', 'dimgray']
                    color_r_dots = ['whitesmoke', 'dimgray']

                # find the airpuff in the position
                temp_ap = nap.IntervalSet(start=nap_air_puff.index - 0.05, end=nap_air_puff.index + 0.05)
                air_puff_loc = nap_pos.restrict(temp_ap).as_series().median()

                # interpolating PCA data to match the position
                new_run_PCA_pre = interpolate_data(run_pos, run_PCA_pre)
                new_run_PCA_task = interpolate_data(run_pos, run_PCA_task)
                new_run_PCA_post = interpolate_data(run_pos, run_PCA_post)

                # interpolating the run gcPCA data to match the position
                # first dimensions
                new_r_gcpca = interpolate_data(run_pos, run_gcpca)

                # interpolating the CPCA multiple alpha to match the position
                new_run_CPCA_allPCs_first = []
                new_run_CPCA_2PCs_first = []
                new_run_CPCA_allPCs_last = []
                for bbb in range(len(run_CPCA_allpcs)):
                    new_run_CPCA_allPCs_first.append(interpolate_data(run_pos, run_CPCA_allpcs[bbb][:, [0, 1]]))

                    new_run_CPCA_allPCs_last.append(interpolate_data(run_pos, run_CPCA_allpcs[bbb][:, [-1, -2]]))

                    new_run_CPCA_2PCs_first.append(interpolate_data(run_pos, run_CPCA_2pcs[bbb]*[-1,1])) # the multiplication is to flip the sign and match gcPCA


                # setting parameters rc to allow better pdf fonttype handling
                plt.rcParams['pdf.fonttype'] = 42
                plt.rcParams['ps.fonttype'] = 42
                plt.rcParams['font.size'] = 16

                ###
                # plot gcPCA first dimensions
                plot_init_left = True
                plot_init_right = True
                plot_ap_left = True
                plot_ap_right = True
                # cmap_str_danger = 'seismic'
                cmap_str_danger = custom_gradient_danger
                cmap_str_safe = custom_gradient
                min_max_t_left = [nap_pos.restrict(left_runs_int[:20]).values.min(),nap_pos.restrict(left_runs_int[:20]).values.max()]
                fig, (axs) = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(3.1, 5))
                for a in left_runs_int.values[:20, :]:  # plotting only the first 20 trials for better visualization
                    prepare_and_plot(axs[0], a, new_r_gcpca,min_max_t_left, roll_wind, nap_pos, air_puff_loc, color_ap_edge_l,
                                     color_l_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)

                min_max_t_right = [nap_pos.restrict(right_runs_int[:20]).values.min(),nap_pos.restrict(right_runs_int[:20]).values.max()]
                for a in right_runs_int.values[:20, :]:
                    prepare_and_plot(axs[1], a, new_r_gcpca, min_max_t_right, roll_wind, nap_pos, air_puff_loc, color_ap_edge_r,
                                     color_r_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)
                axs[0].set_title('gcPCA')
                fig.subplots_adjust(left=0.2, top=0.90, right=0.85, bottom=0.1, hspace=0.2, wspace=0.2)
                fig.savefig(save_fig_dir + "gcPCA_space_PRE_POST_v2.pdf", transparent=True)

                #####################################################
                # making a plot specifically for different alphas CPCA
                cpc_str_xlabel = ['(Post,Pre) cPC$_1$']
                cpc_str_ylabel = ['(Post,Pre) cPC$_2$']
                plt.rcParams['font.size'] = 16
                for bbb in range(len(new_run_CPCA_allPCs_first)):
                    fig, (axs) = plt.subplots(2, 1, figsize=(3.1, 5), sharex=False, sharey=False)
                    for a in left_runs_int.values[:20, :]:
                        prepare_and_plot(axs[0], a, new_run_CPCA_allPCs_first[bbb], min_max_t_left, roll_wind, nap_pos, air_puff_loc,
                                         color_ap_edge_l, color_l_dots, cpc_str_xlabel[0], cpc_str_ylabel[0], cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)

                    for a in right_runs_int.values[:20, :]:
                        prepare_and_plot(axs[1], a, new_run_CPCA_allPCs_first[bbb], min_max_t_right, roll_wind, nap_pos, air_puff_loc,
                                         color_ap_edge_r, color_r_dots, cpc_str_xlabel[0], cpc_str_ylabel[0], cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)

                    fig.subplots_adjust(left=0.2, top=0.90, right=0.85, bottom=0.1, hspace=0.2, wspace=0.2)
                    fig.suptitle('k=' + str(post_task_data.shape[1]) + ', ' + r'$\alpha$ = ' + "{:.1f}".format(
                        best_alphas_allPCs[bbb + 1]), x=0.53, y=0.97,
                                 fontweight='bold')
                    fig.savefig(save_fig_dir + "PRE_POST_all_cPCs_RUN_v2_alpha_" + str(best_alphas_allPCs[bbb + 1]) + ".pdf", transparent=True)

                ###
                # plot of the 2 cPCs model
                for bbb in range(len(new_run_CPCA_2PCs_first)):
                    fig, (axs) = plt.subplots(2, 1, figsize=(3.1, 5), sharex=False, sharey=False)
                    for a in left_runs_int.values[:20, :]:
                        prepare_and_plot(axs[0], a, new_run_CPCA_2PCs_first[bbb], min_max_t_left, roll_wind, nap_pos, air_puff_loc,
                                         color_ap_edge_l, color_l_dots, cpc_str_xlabel[0], cpc_str_ylabel[0], cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)

                    for a in right_runs_int.values[:20, :]:
                        prepare_and_plot(axs[1], a, new_run_CPCA_2PCs_first[bbb],min_max_t_right, roll_wind, nap_pos, air_puff_loc,
                                            color_ap_edge_r, color_r_dots, cpc_str_xlabel[0], cpc_str_ylabel[0], cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)

                    fig.subplots_adjust(left=0.2, top=0.90, right=0.85, bottom=0.1, hspace=0.2, wspace=0.2)
                    fig.suptitle('k=2, ' + r'$\alpha$ = ' + "{:.1f}".format(best_alphas_2PCs[bbb + 1]), x=0.53, y=0.97,
                                 fontweight='bold')
                    fig.savefig(save_fig_dir + "PRE_POST_2_cPCs_RUN_v2_alpha_" + str(
                        best_alphas_2PCs[bbb + 1]) + ".pdf", transparent=True)

                ###
                # make plots of the PCA projections

                # pre-task PCA
                fig, (axs) = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(3.1, 5))
                for a in left_runs_int.values[:20, :]:
                    # pre task PCA
                    prepare_and_plot(axs[0], a, new_run_PCA_pre, min_max_t_left, roll_wind, nap_pos, air_puff_loc, color_ap_edge_l,
                                     color_l_dots, 'Pre PC$_{1}$', 'Pre PC$_{2}$', cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)

                for a in right_runs_int.values[:20, :]:
                    # pre task PCA
                    prepare_and_plot(axs[1], a, new_run_PCA_pre, min_max_t_right, roll_wind, nap_pos, air_puff_loc,
                                     color_ap_edge_r,color_r_dots, 'Pre PC$_{1}$', 'Pre PC$_{2}$', cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)
                axs[0].set_title('Pre-task PCA')
                fig.subplots_adjust(left=0.2, top=0.90, right=0.85, bottom=0.1, hspace=0.2, wspace=0.2)
                fig.savefig(save_fig_dir + "PCA_space_PRE_v1.pdf", transparent=True)


                # Task PCA
                fig, (axs) = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(3.1, 5))
                for a in left_runs_int.values[:20, :]:
                    # task PCA
                    prepare_and_plot(axs[0], a, new_run_PCA_task, min_max_t_left, roll_wind, nap_pos, air_puff_loc, color_ap_edge_l,
                                     color_l_dots, 'Task PC$_{1}$', 'Task PC$_{2}$', cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)
                for a in right_runs_int.values[:20, :]:
                    # task PCA
                    prepare_and_plot(axs[1], a, new_run_PCA_task, min_max_t_right, roll_wind, nap_pos, air_puff_loc,
                                     color_ap_edge_r, color_r_dots, 'Task PC$_{1}$', 'Task PC$_{2}$', cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)
                axs[0].set_title('Task PCA')
                fig.subplots_adjust(left=0.2, top=0.90, right=0.85, bottom=0.1, hspace=0.2, wspace=0.2)
                fig.savefig(save_fig_dir + "PCA_space_TASK_v1.pdf", transparent=True)


                fig, (axs) = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(3.1, 5))
                for a in left_runs_int.values[:20, :]:
                    # pre task PCA
                    prepare_and_plot(axs[0], a, new_run_PCA_post, min_max_t_left, roll_wind, nap_pos, air_puff_loc, color_ap_edge_l,
                                        color_l_dots, 'Post PC$_{1}$', 'Post PC$_{2}$', cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)
                for a in right_runs_int.values[:20, :]:
                    # post task PCA
                    prepare_and_plot(axs[1], a, new_run_PCA_post, min_max_t_right, roll_wind, nap_pos, air_puff_loc, color_ap_edge_r,
                                     color_r_dots, 'Post PC$_{1}$', 'Post PC$_{2}$', cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)
                axs[0].set_title('Post PCA')
                fig.subplots_adjust(left=0.2, top=0.90, right=0.85, bottom=0.1, hspace=0.2, wspace=0.2)
                fig.savefig(save_fig_dir + "PCA_space_POST_v1.pdf", transparent=True)

                ###
                # make raster plots of each condition and run
                plt.rcParams['font.size'] = 16
                pre_task_interval_plot = nap.IntervalSet(start=pre_task_temp[0], end=pre_task_temp[0] + 50)
                post_task_interval_plot = nap.IntervalSet(start=post_task_temp[0], end=post_task_temp[0] + 50)
                task_interval_plot = nap.IntervalSet(start=run_temp[0], end=run_temp[0] + 50)
                temp_spks_pre = spikes_times.restrict(pre_task_interval_plot)
                temp_spks_post = spikes_times.restrict(post_task_interval_plot)
                temp_spks_task = spikes_times.restrict(task_interval_plot)

                # pre-task raster plot
                fig, axs = plt.subplots(1, 1, figsize=(4, 3))
                for a in range(len(cells_hpc[cells2keep])):
                    axs.eventplot(temp_spks_pre[a].index, lineoffsets=a, color='k', linewidths=0.5, linelengths=0.5)
                axs.set_title('Pre-task Recordings')
                # axs.set_xticks([])
                axs.set_ylabel('Neurons')
                plt.tight_layout()
                plt.savefig(save_fig_dir + "_pre_task_raster.pdf", transparent=True)

                # post-task raster plot
                fig, axs = plt.subplots(1, 1, figsize=(4, 3))
                for a in range(len(cells_hpc[cells2keep])):
                    axs.eventplot(temp_spks_post[a].index, lineoffsets=a, color='k', linewidths=0.5, linelengths=0.5)
                axs.set_title('Post-task Recordings')
                # axs.set_xticks([])
                axs.set_ylabel('Neurons')
                plt.tight_layout()
                plt.savefig(save_fig_dir + "_post_task_raster.pdf", transparent=True)

                # task plot
                fig, axs = plt.subplots(1, 1, figsize=(4, 3))
                for a in range(len(cells_hpc[cells2keep])):
                    axs.eventplot(temp_spks_task[a].index, lineoffsets=a, color='k', linewidths=0.5, linelengths=0.5)
                axs.set_title('Task')
                # axs.set_xticks([])
                axs.set_ylabel('Neurons')
                plt.tight_layout()
                plt.savefig(save_fig_dir + "_task_raster.pdf", transparent=True)
plt.show()
