""" This script is used to show the difference between non orthogonal, orthogonal and sparse gcPCA

I'm going to show the covariance matrix of the gcPCs, the time used to calculate, and the loadings of the gcPCs

I'll be using the spike activity data from hippocampus for that
"""
import os
# importing packages
import sys
import numpy as np
import pynapple as nap
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# import custom modules
# change repo_dir to your gcPCA directory
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import sparse_gcPCA, gcPCA

# parameters for analysis
mrkr_size = 20
min_n_cell = 30  # min number of cells in the brain area to be used
min_fr = 0.01  # minimum firing rate to be included in the analysis
bin_size = 0.01  # 10 ms
std_conv = 2  # standard deviation for convolution (in num bins units)
wind_cov = 5  # window size for convolution (in num bins units)
roll_wind = 20  # window size for rolling mean

# parameters for plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'


###
# loading HPC data
#

# other directories
# directory with the HPC data
base_dir = '/home/eliezyer/Documents/github/deOliveira_gcPCA_2024/Figure3/'

# directory where to save the figures
save_fig_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/Supplemental_figures/gcPCA_versions_comparison/'


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
# normalizing and smoothing data - run periods
temp_data = zscore(temp_r.count(bin_size).as_dataframe().rolling(window=wind_cov,
                                                                 win_type='gaussian', center=True,
                                                                 min_periods=1).mean(std=std_conv).values)
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
standard_gcpca_mdl = gcPCA(method='v4', normalize_flag=False)
standard_gcpca_mdl.fit(N_post, N_pre)
standard_time_elapsed = standard_gcpca_mdl.elapsed_time_

#orthogonal gcPCA fitting
orthogonal_gcpca_mdl = gcPCA(method='v4.1',normalize_flag=False)
orthogonal_gcpca_mdl.fit(N_post, N_pre)
orthogonal_time_elapsed = orthogonal_gcpca_mdl.elapsed_time_

#sparse gcPCA fitting
lambdas_use = np.exp(np.linspace(np.log(1.2e-20), np.log(150), num=2))
sparse_gcpca_mdl = sparse_gcPCA(method='v4', normalize_flag=False, lasso_penalty=lambdas_use, ridge_penalty=1e-5, Nsparse=N_post.shape[1], max_steps=1000, tol=1e-5)
sparse_gcpca_mdl.fit(N_post, N_pre)
sparse_time_elapsed = sparse_gcpca_mdl.elapsed_time_

# preparing data for plotting
# gcPCA projection run data
run_standard_gcpca = nap.TsdFrame(run_time, d=N_task.dot(standard_gcpca_mdl.loadings_[:, :2]))
run_orthogonal_gcpca = nap.TsdFrame(run_time, d=N_task.dot(orthogonal_gcpca_mdl.loadings_[:, :2]))
sparse_loadings_use = sparse_gcpca_mdl.sparse_loadings_[-1]
run_sparse_gcpca = nap.TsdFrame(run_time, d=N_task.dot(sparse_loadings_use[:, :2]*[1,1]))

# setting plot color according to whether run is safe or danger
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

# interpolating the run gcPCA data to match the position
# first dimensions
interp_run_standard_gcpca = interpolate_data(run_pos, run_standard_gcpca)
interp_run_orthogonal_gcpca = interpolate_data(run_pos, run_orthogonal_gcpca)
interp_run_sparse_gcpca = interpolate_data(run_pos, run_sparse_gcpca)

# setting parameters rc to allow better pdf fonttype handling
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 16

# extra plot parameters
plot_init_left = True
plot_init_right = True
plot_ap_left = True
plot_ap_right = True
cmap_str_danger = custom_gradient_danger
cmap_str_safe = custom_gradient
min_max_t_left = [nap_pos.restrict(left_runs_int[:20]).values.min(),nap_pos.restrict(left_runs_int[:20]).values.max()]
min_max_t_right = [nap_pos.restrict(right_runs_int[:20]).values.min(),nap_pos.restrict(right_runs_int[:20]).values.max()]

###
# plot gcPCA first dimensions for left and right runs for standard, orthogonal and sparse gcPCA
fig, (axs) = plt.subplots(3, 4, sharex=False, sharey=False, figsize=(12, 9))
for a in left_runs_int.values[:20, :]:  # plotting only the first 20 trials for better visualization
    # standard gcPCA plots
    prepare_and_plot(axs[0,2], a, interp_run_standard_gcpca, min_max_t_left, roll_wind, nap_pos, air_puff_loc, color_ap_edge_l,
                     color_l_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)
    # orthogonal gcPCA plots
    prepare_and_plot(axs[1,2], a, interp_run_orthogonal_gcpca, min_max_t_left, roll_wind, nap_pos, air_puff_loc, color_ap_edge_l,
                        color_l_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)
    # sparse gcPCA plots
    prepare_and_plot(axs[2,2], a, interp_run_sparse_gcpca, min_max_t_left, roll_wind, nap_pos, air_puff_loc, color_ap_edge_l,
                        color_l_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_danger, plot_init=plot_init_left, plot_ap=plot_ap_left)


for a in right_runs_int.values[:20, :]:
    # standard gcPCA plots
    prepare_and_plot(axs[0,3], a, interp_run_standard_gcpca, min_max_t_right, roll_wind, nap_pos, air_puff_loc, color_ap_edge_r,
                     color_r_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)
    # orthogonal gcPCA plots
    prepare_and_plot(axs[1,3], a, interp_run_orthogonal_gcpca, min_max_t_right, roll_wind, nap_pos, air_puff_loc, color_ap_edge_r,
                        color_r_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)
    # sparse gcPCA plots
    prepare_and_plot(axs[2,3], a, interp_run_sparse_gcpca, min_max_t_right, roll_wind, nap_pos, air_puff_loc, color_ap_edge_r,
                        color_r_dots, '(Post,Pre) gcPC$_{1}$', '(Post,Pre) gcPC$_{2}$', cmap_str_safe, plot_init=plot_init_right, plot_ap=plot_ap_right)

# make the plots square
for a in range(3):
    axs[a, 2].set_aspect(1.0 / axs[a, 2].get_data_ratio(), adjustable='box')
    axs[a, 3].set_aspect(1.0 / axs[a, 3].get_data_ratio(), adjustable='box')
axs[0,0].set_title('standard gcPCA')
axs[1,0].set_title('orthogonal gcPCA')
axs[2,0].set_title('sparse gcPCA')

###
# orthogonality and loadings plots

# standard_eigenvalues = (standard_gcpca_mdl.Ra_values_-standard_gcpca_mdl.Rb_values_) / (standard_gcpca_mdl.Ra_values_+standard_gcpca_mdl.Rb_values_)
loads_ = standard_gcpca_mdl.loadings_
maxl = np.max(np.abs(loads_))
axs[0,0].imshow(np.abs(loads_.T.dot(loads_))/maxl, cmap='viridis',clim=(0,maxl))
# title with time run and 2 floating points
axs[0,0].set_title(f'standard gcPCA \n time: {standard_time_elapsed:.3f}s',rotation='vertical',x=-0.42,y=0.47,va='center')
axs[0,0].set_xlabel('gcPCs')
axs[0,0].set_ylabel('gcPCs')
axs[0,0].set_xticks([0,9,19,29,39],['1','10','20','30','40'])
axs[0,0].set_yticks([0,9,19,29,39],['1','10','20','30','40'])

loads_ = orthogonal_gcpca_mdl.loadings_
axs[1,0].imshow(np.abs(loads_.T.dot(loads_))/maxl, cmap='viridis',clim=(0,maxl))
axs[1,0].set_title(f'orthogonal gcPCA \n time: {orthogonal_time_elapsed:.3f}s',rotation='vertical',x=-0.42,y=0.47,va='center')
axs[1,0].set_xlabel('gcPCs')
axs[1,0].set_ylabel('gcPCs')
axs[1,0].set_xticks([0,9,19,29,39],['1','10','20','30','40'])
axs[1,0].set_yticks([0,9,19,29,39],['1','10','20','30','40'])


loads_ = sparse_loadings_use
im = axs[2,0].imshow(np.abs(loads_.T.dot(loads_))/maxl, cmap='viridis',clim=(0,maxl))
axs[2,0].set_title(f'sparse gcPCA \n time: {sparse_time_elapsed:.3f}s',rotation='vertical',x=-0.42,y=0.47,va='center')
axs[2,0].set_xlabel('gcPCs')
axs[2,0].set_ylabel('gcPCs')
axs[2,0].set_xticks([0,9,19,29,39],['1','10','20','30','40'])
axs[2,0].set_yticks([0,9,19,29,39],['1','10','20','30','40'])



loads_ = standard_gcpca_mdl.loadings_
max_l = np.max(np.abs(loads_))
axs[0,1].imshow(np.abs(loads_), cmap='viridis',clim=(0,max_l))
axs[0,1].set_xlabel('gcPCs')
axs[0,1].set_ylabel('Features')
axs[0,1].set_xticks([0,9,19,29,39],['1','10','20','30','40'])
axs[0,1].set_yticks([0,9,19,29,39],['1','10','20','30','40'])

loads_ = orthogonal_gcpca_mdl.loadings_
axs[1,1].imshow(np.abs(loads_), cmap='viridis',clim=(0,max_l))
axs[1,1].set_xlabel('gcPCs')
axs[1,1].set_ylabel('Features')
axs[1,1].set_xticks([0,9,19,29,39],['1','10','20','30','40'])
axs[1,1].set_yticks([0,9,19,29,39],['1','10','20','30','40'])

loads_ = sparse_loadings_use
im = axs[2,1].imshow(np.abs(loads_), cmap='viridis',clim=(0,max_l))
axs[2,1].set_xlabel('gcPCs')
axs[2,1].set_ylabel('Features')
axs[2,1].set_xticks([0,9,19,29,39],['1','10','20','30','40'])
axs[2,1].set_yticks([0,9,19,29,39],['1','10','20','30','40'])


fig.subplots_adjust(left=0.1, top=0.99, right=0.99, bottom=0.05, hspace=0.2, wspace=0.35)

fig.savefig(save_fig_dir + "HPC_gcPCA_versions_comparsion.pdf", transparent=True)
fig.savefig(save_fig_dir + "HPC_gcPCA_versions_comparsion.png", transparent=True)