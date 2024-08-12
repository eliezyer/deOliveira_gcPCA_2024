# -*- coding: utf-8 -*-
"""
@author: Eliezyer Fermino de Oliveira

Script to generate synthetic data with dimensions changing variance between conditions.
"""

# importing libraries
import sys
import numpy as np
from scipy.stats import zscore
from scipy.linalg import orth
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import os

# change repo_dir to the location where you save the gcPCA package
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

plt.rcParams.update({'figure.dpi': 150, 'font.size': 15,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                    'text.latex.preamble': r'\usepackage{amsfonts}'})

# change it to the location where to save the figures
save_fig_dir = "/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/Figure1/source_plots/"  # path to save figures

# %% parameters
N_samples = 100000
N_features = 100

# %% ancillary functions for this script
# function to lighten a picked color
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# defining the Ca - Cb contrastive method to use in the figures
def contrastive_basic(dataA, dataB):
    """ function to find the contrastive PCs in the most basic objective function"""
    covA = dataA.T.dot(dataA)/(dataA.shape[0]-1)
    covB = dataB.T.dot(dataB)/(dataB.shape[0]-1)

    sigma = covA - covB
    e, d = np.linalg.eigh(sigma)

    contrast_loadings = d[:,np.argsort(e)[::-1]]
    return contrast_loadings

# function to plot projections of data into loadings
def plotting_projection(data_plot,loadings,col_array,ax,xlabel,ylabel):
    plt.scatter(data_plot @ loadings[:, 0], data_plot @ loadings[:, 1], c=col_array, s=7)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    plt.tight_layout()
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

# function to get cPCA loadings
def extract_cpca_loadings(data_A,data_B,alpha_value):
    """script to quickly extract the loadings of cPCA"""
    covA = data_A.T.dot(data_A)/(data_A.shape[0]-1)
    covB = data_B.T.dot(data_B)/(data_B.shape[0]-1)

    sigma = covA - alpha_value*covB
    e, d = np.linalg.eigh(sigma)
    loadings = d[:,np.argsort(e)[::-1]]
    return loadings

# %% defining function generate synthetic data
def generate_data(N_samples, N_features):
    increased_factor = 2

    # ancillary function to generate latent_factors 1 through 4
    def generate_latent_factors(N_samples):
        # factors that increase in condition A
        latent_factor1 = np.random.rand(int(N_samples), 1)
        latent_factor2 = np.random.rand(int(N_samples), 1)
        idx1 = np.logical_and(np.logical_and(latent_factor1 > 0.3, latent_factor1 < 0.7),
                              np.logical_and(latent_factor2 > 0.4, latent_factor2 < 0.6))
        latent_factor1[idx1.flatten()] = np.random.rand(sum(idx1)[0], 1) * 0.4
        latent_factor2[idx1.flatten()] = np.random.rand(sum(idx1)[0], 1) * 0.4
        latent_factor1, latent_factor2 = latent_factor1 - 0.5, latent_factor2 - 0.5
        scores_for_color = np.arctan2(latent_factor2, latent_factor1).flatten()
        Isort = np.argsort(scores_for_color)
        latent_factor1 = latent_factor1[Isort]
        latent_factor2 = latent_factor2[Isort]

        # factors increased in condition B
        temp1 = np.random.rand(int(N_samples))
        temp2 = np.random.rand(int(N_samples))
        Isort = np.argsort(temp1.flatten())
        temp1, temp2 = temp1[Isort], temp2[Isort]
        theta = np.deg2rad(45)
        latent_factor3 = temp1 * np.cos(theta) + temp2 * -np.sin(theta)
        latent_factor4 = temp1 * np.sin(theta) + temp2 * np.cos(theta)

        return latent_factor1, latent_factor2, latent_factor3, latent_factor4

    temp_factor1, temp_factor2, temp_factor3, temp_factor4 = [], [], [], []
    for N in range(100):
        # generating latent factors
        temp1, temp2, temp3, temp4 = generate_latent_factors(N_samples / 100)
        temp_factor1.append(temp1)
        temp_factor2.append(temp2)
        temp_factor3.append(temp3)
        temp_factor4.append(temp4)
    latent_factor1 = np.concatenate(temp_factor1)
    latent_factor2 = np.concatenate(temp_factor2)
    latent_factor3 = np.concatenate(temp_factor3)
    latent_factor4 = np.concatenate(temp_factor4)

    # normalizing by the l2 norm
    latent_factor1 = latent_factor1 / np.linalg.norm(latent_factor1)
    latent_factor2 = latent_factor2 / np.linalg.norm(latent_factor2)
    latent_factor3 = latent_factor3 / np.linalg.norm(latent_factor3)
    latent_factor4 = latent_factor4 / np.linalg.norm(latent_factor4)

    # other factors
    rest_factors1 = zscore(np.random.randn(N_samples, N_features))
    rest_factors1 = rest_factors1 / np.linalg.norm(rest_factors1, axis=0)

    # generating data condition A
    # eigenvalues
    Sa = (np.linspace(0, stop=10, num=N_features)[::-1] + 10 ** -4)
    Sa[70] = increased_factor * Sa[70]  # boosting the dimensions' eigenvalue
    Sa[71] = increased_factor * Sa[71]  # boosting the dimensions' eigenvalue

    # getting orthogonal weights
    W = orth(np.random.randn(N_features, N_features)).T

    # generating samples from the low variance manifold in condition A
    samples1 = np.outer(latent_factor1, Sa[70] * W[70, :])
    samples2 = np.outer(latent_factor2, Sa[71] * W[71, :])

    # generating samples from the other factors
    auxSm = np.repeat(Sa[:70, np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[:70, :])
    rest_samples1 = np.dot(rest_factors1[:, :70], newW)

    auxSm = np.repeat(Sa[72:, np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[72:, :])
    rest_samples2 = np.dot(rest_factors1[:, 72:], newW)

    # adding for final data
    data_A = samples1 + samples2 + rest_samples1 + rest_samples2

    # generating data_B
    # eigenvalues
    Sb = (np.linspace(0, stop=10, num=N_features)[::-1] + 10 ** -4)
    Sb[80] = increased_factor * Sb[80]  # boosting the dimensions' eigenvalue
    Sb[81] = increased_factor * Sb[81]  # boosting the dimensions' eigenvalue

    # generating samples from the low variance manifold in condition B
    samples3 = np.outer(latent_factor3, Sb[80] * W[80, :])
    samples4 = np.outer(latent_factor4, Sb[81] * W[81, :])

    # other factors
    rest_factors1 = zscore(np.random.randn(N_samples, N_features))
    rest_factors1 = rest_factors1 / np.linalg.norm(rest_factors1, axis=0)

    # generating samples from the other factors
    auxSm = np.repeat(Sb[:80, np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[:80, :])
    rest_samples1 = np.dot(rest_factors1[:, :80], newW)

    auxSm = np.repeat(Sb[82:, np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[82:, :])
    rest_samples2 = np.dot(rest_factors1[:, 82:], newW)

    # adding for final data
    data_B = samples3 + samples4 + rest_samples1 + rest_samples2

    return data_B, data_A, W


# %% generating the figures
N_samples2plot = 1000  # it has to be multiples of 1k for better visualization
data_B, data_A, W = generate_data(N_samples, N_features)
data_a = data_A[:N_samples2plot, :]  # subsampling the data for the finite data regime
data_b = data_B[:N_samples2plot, :]  # subsampling the data for the finite data regime

# calculating eigenspectrum
temp = data_a @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
A_eigspec_sampled = np.sum(R ** 2, axis=1)

temp = data_A @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
A_eigspec_true = np.sum(R ** 2, axis=1)

temp = data_b @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
B_eigspec_sampled = np.sum(R ** 2, axis=1)

temp = data_B @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
B_eigspec_true = np.sum(R ** 2, axis=1)

# %% plotting the first row of figure 1, example of sample plots
# preparing plots
grid1 = plt.GridSpec(2, 3, left=0.03, right=0.67, hspace=0.3)
grid2 = plt.GridSpec(2, 1, left=0.75, right=1, hspace=0.3)

#colormaps for samples
hsv = mpl.colormaps.get_cmap('hsv')
spec_cmap = mpl.colormaps.get_cmap('PRGn')
temp = np.linspace(0, 1, 1000)
hsv_col = hsv(np.tile(temp, int(N_samples2plot / 1000)))
spec_col = spec_cmap(np.tile(temp, int(N_samples2plot / 1000)))

# start plotting
fig = plt.figure(num=0, figsize=(13, 6.5))
# plot of synthetic data latent features
plot1_xlim = (-0.05, 0.05)
plot1_ylim = (-0.05, 0.05)

# plot of synthetic data latent features on condition A
# dimensions 0 and 1
ax = plt.subplot(grid1[0, 0])
plt.scatter(data_a @ W[0, :].T, data_a @ W[1, :].T,
            c=hsv_col,
            s=6)
plt.xticks([])
plt.yticks([])
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
ax.set_aspect(1.0)

# plot of dimensions 70 and 71
ax = plt.subplot(grid1[0, 1])
plt.scatter(data_a @ W[70, :].T, data_a @ W[71, :].T,
            c=hsv_col,
            s=6)
plt.xticks([])
plt.yticks([])
plt.title('Low variance\nA manifold', weight='bold', color='green')
plt.xlabel('dimension 71')
plt.ylabel('dimension 72')

# change all spines
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(4)

for spine in ax.spines.values():
    spine.set_edgecolor('green')
ax.set_xlim(plot1_xlim)
ax.set_ylim(plot1_ylim)
ax.set_aspect(1.0)

# plot of dimensions 80 and 81
ax = plt.subplot(grid1[0, 2])
plt.scatter(data_a @ W[80, :].T, data_a @ W[81, :].T,
            c=hsv_col,
            s=6)

plt.xticks([])
plt.yticks([])
plt.title('Low variance\nB manifold', weight='bold')
plt.xlabel('dimension 81')
plt.ylabel('dimension 82')
ax.set_aspect(1.0)

# plot of synthetic data latent features on condition B
# dimensions 0 and 1
ax = plt.subplot(grid1[1, 0])
plt.scatter(data_b @ W[0, :].T, data_b @ W[1, :].T,
            c=spec_col,
            s=6)

plt.xticks([])
plt.yticks([])
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
ax.set_aspect(1.0)

# plot of dimensions 70 and 71
ax = plt.subplot(grid1[1, 1])
plt.scatter(data_b @ W[70, :].T, data_b @ W[71, :].T,
            c=spec_col,
            s=6)
plt.xticks([])
plt.yticks([])
plt.xlabel('dimension 71')
plt.ylabel('dimension 72')
ax.set_aspect(1.0)

# plot of dimensions 80 and 81
ax = plt.subplot(grid1[1, 2])
plt.scatter(data_b @ W[80, :].T, data_b @ W[81, :].T,
            c=spec_col,
            s=6)
plt.xticks([])
plt.yticks([])

# changing all spines
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(4)
# plt.title('Low variance\nmanifold')
plt.xlabel('dimension 81')
plt.ylabel('dimension 82')
ax.set_xlim(plot1_xlim)
ax.set_ylim((-0.01,0.03))
# ax.set_aspect(1.0)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

# plotting the eigenspectra condition A, finite and infinite data
ax = plt.subplot(grid2[0, 0])
x = np.arange(1, 101, step=1)
plt.plot(x, A_eigspec_sampled, color='green', linestyle='-', label='Finite Data',
         linewidth=4, zorder=2)
light_green = lighten_color('green',0.85)
plt.plot(x, A_eigspec_true/1e4, color=light_green, linestyle='-', label='Infinite Data',
         linewidth=4, zorder=3)

plt.legend(loc=(0.13, 0.70))
plt.xticks((1, 71, 81))
plt.yticks((0.0, 0.5, 1.0))
plt.ylabel('Eigenvalue')
plt.grid(color='grey', linestyle='-', linewidth=2, alpha=0.4, zorder=0)
plt.xlabel('Dimensions')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# moving ticks to left and right
# Get the current tick labels
ticks = ax.get_xticks()
tick_labels = ax.get_xticklabels()

# Offset value for moving the tick labels left or right
# Positive values move to the right, negative to the left
offset_left = mtransforms.ScaledTranslation(4/72, 0, fig.dpi_scale_trans)
offset_right = mtransforms.ScaledTranslation(5/72, 0, fig.dpi_scale_trans)

# Apply transformation to selected tick labels
for tick, label in zip(ticks, tick_labels):
    if tick == 71:
        label.set_transform(label.get_transform() - offset_left)  # Move left
    elif tick == 81:
        label.set_transform(label.get_transform() + offset_right)  # Move right

# plotting the eigenspectra condition B, finite and infinite data
ax = plt.subplot(grid2[1, 0])
x = np.arange(1, 101, step=1)
plt.plot(x, B_eigspec_sampled, color='black', linestyle='-', label='Finite Data',
         linewidth=4, zorder=2)
light_black = lighten_color('black', 0.3)
plt.plot(x, B_eigspec_true/1e4, color=light_black, linestyle='-', label='Infinite Data', linewidth=4,
         zorder=3) # normalized by the subsampling ratio

plt.legend(loc=(0.13, 0.70))
plt.xticks((1, 71, 81))
plt.yticks((0.0, 0.5, 1.0))
plt.ylabel('Eigenvalue')
plt.grid(color='black', linestyle='-', linewidth=2,  alpha=0.4, zorder=0)
plt.xlabel('Dimensions')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# moving ticks to left and right
# Get the current tick labels
ticks = ax.get_xticks()
tick_labels = ax.get_xticklabels()
# Apply transformation to selected tick labels
for tick, label in zip(ticks, tick_labels):
    if tick == 71:
        label.set_transform(label.get_transform() - offset_left)  # Move left
    elif tick == 81:
        label.set_transform(label.get_transform() + offset_right)  # Move right

plt.figtext(0.21, 0.69, '+', fontsize=30, fontweight='bold')
plt.figtext(0.21, 0.25, '+', fontsize=30, fontweight='bold')
plt.figtext(0.44, 0.69, '+', fontsize=30, fontweight='bold')
plt.figtext(0.44, 0.25, '+', fontsize=30, fontweight='bold')
plt.show()
fig.savefig(os.path.join(save_fig_dir, 'latent_features_eigenvalue_model_linear.pdf'))
fig.savefig(os.path.join(save_fig_dir, 'latent_features_eigenvalue_model_linear.png'))

# %% plot the second row of figure 1, showing all the methods solutions in a single row
fig = plt.figure(num=1, figsize=(13, 4))
grid1 = plt.GridSpec(1, 3, left=0.05, right=0.99, bottom=0.15, top=0.83,wspace=0.2)

###
# plot of A-B infinite vs finite data
ax = plt.subplot(grid1[0])
A_sampled = A_eigspec_sampled
B_sampled = B_eigspec_sampled
x = np.arange(1, 101, step=1)
light_royal_blue = lighten_color('royalblue', amount=0.5)
plt.plot(x, (A_sampled - B_sampled), color='royalblue', linestyle='-', label='Finite Data',
         linewidth=4, zorder=4)
plt.plot(x[70], (A_sampled - B_sampled)[70], color='royalblue', marker='*', markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=4)
plt.plot(x[80], (A_sampled - B_sampled)[80], color='royalblue', marker='*', markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=4)
plt.xticks((1, 71, 81))

A_true = A_eigspec_true
B_true = B_eigspec_true
x = np.arange(1, 101, step=1)
plt.plot(x, (A_true - B_true)/1e4, color=light_royal_blue, linestyle='-', label='Infinite Data', linewidth=4, zorder=5)
plt.grid(color='grey', linestyle='-', linewidth=2, alpha=0.4, zorder=0)

plt.title(r'$\mathbf{C}_A-\mathbf{C}_B$')
plt.ylabel('Value (A.U.)')
plt.yticks((-0.2, 0, 0.2))
plt.ylabel('Value (A.U.)')
plt.xlabel('Dimensions')
plt.legend(loc=(0.15, 0.009))
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# moving ticks to left and right
# Get the current tick labels
ticks = ax.get_xticks()
tick_labels = ax.get_xticklabels()
offset_left = mtransforms.ScaledTranslation(4/72, 0, fig.dpi_scale_trans)
offset_right = mtransforms.ScaledTranslation(5/72, 0, fig.dpi_scale_trans)
# Apply transformation to selected tick labels
for tick, label in zip(ticks, tick_labels):
    if tick == 81:
        label.set_transform(label.get_transform() + offset_left)  # Move right
    elif tick == 71:
        label.set_transform(label.get_transform() - offset_right)  # Move left

###
# plot of cPCA with different alphas
ax = plt.subplot(grid1[1])
x = np.arange(1, 101, step=1)
# colormap for the different resultant eigenspectra
inferno = mpl.colormaps.get_cmap('inferno')
inferno_col = inferno(np.linspace(0, 1, 6))
c = 0
alpha_list = [0.4, 0.7, 1.0, 1.3, 1.9]
for alpha in alpha_list:
    plt.plot(x, A_eigspec_sampled - alpha * B_eigspec_sampled,
             color=inferno_col[c],
             linestyle='-',
             linewidth=4,
             zorder=2,
             label='α: ' + str(alpha))
    c += 1

plt.title('$\mathbf{C}_A-α*\mathbf{C}_B$')
plt.xticks((1, 71, 81))
plt.yticks((-1, 0, 1))
plt.ylabel('Value (A.U.)')
plt.grid(color='grey',
         linestyle='-',
         linewidth=2,
         alpha=0.4,
         zorder=0)

plt.xlabel('Dimensions')
plt.tight_layout()
plt.legend(loc='lower right',fontsize=12)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# moving ticks to left and right
# Get the current tick labels
ticks = ax.get_xticks()
tick_labels = ax.get_xticklabels()
# Apply transformation to selected tick labels
for tick, label in zip(ticks, tick_labels):
    if tick == 71:
        label.set_transform(label.get_transform() - offset_left)  # Move left
    elif tick == 81:
        label.set_transform(label.get_transform() + offset_right)  # Move right

###
# plot of gcPCa objective function
ax = plt.subplot (grid1[2])
x = np.arange(1, 101, step=1)
plt.plot(x, ((A_sampled - B_sampled) / (A_sampled + B_sampled)),
         color='firebrick', linestyle='-', linewidth=4, zorder=2)
plt.xticks((1, 71, 81))
plt.yticks((-0.5, 0, 0.5))
plt.ylabel('Value (A.U.)')
plt.grid(color='grey',
         linestyle='-', linewidth=2, alpha=0.4, zorder=0)
plt.title(r'$\frac{\mathbf{C}_A-\mathbf{C}_B}{\mathbf{C}_A+\mathbf{C}_B}$')


plt.xlabel('Dimensions')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# moving ticks to left and right
# Get the current tick labels
ticks = ax.get_xticks()
tick_labels = ax.get_xticklabels()
# Apply transformation to selected tick labels
for tick, label in zip(ticks, tick_labels):
    if tick == 71:
        label.set_transform(label.get_transform() - offset_left)  # Move left
    elif tick == 81:
        label.set_transform(label.get_transform() + offset_right)  # Move right

plt.show()

fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_functions.pdf'), transparent=True)
fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_functions.png'), transparent=True)

# %% third row of figure 1: what each contrastive method (cPCA vs gcPCA) reveal in samples
fig = plt.figure(num=2, figsize=(13, 4))
# preparing grid for plot
grid2 = plt.GridSpec(2, 5, figure=fig, left=0.03, right=0.8, wspace=0.3)
grid3 = plt.GridSpec(2, 1, figure=fig, left=0.85, right=0.99)

# setting colormap for samples
temp = np.linspace(0, 1, 1000)
hsv_col = hsv(np.tile(temp, 1))
spec_col = spec_cmap(np.tile(temp, 1))

###
# cPCA different alphas
alpha_list2 = [0.1, 0.5, 1, 2, 10]
for c,alp in enumerate(alpha_list2):
    V = extract_cpca_loadings(data_a,data_b,alp)
    ax = plt.subplot(grid2[0, c])
    V_temp = V[:, :2]
    plotting_projection(data_a, V_temp, hsv_col, ax, 'cPC$_{1}$', 'cPC$_{2}$')
    # write the title and add the alpha value in the text with 2 floating decimal points precision
    if alp == 1:
        plt.title('$\mathbf{C}_A - \mathbf{C}_B$',fontsize=18)
    else:
        title = r'$\mathbf{C}_A - $' + f'{alp:.1f}' + '$\mathbf{C}_B$'
        plt.title(title,fontsize=18)  # add alpha here

    ax = plt.subplot(grid2[1, c])
    V_temp = V[:, -2:][:, [1, 0]]
    plotting_projection(data_a, V_temp, spec_col, ax, 'cPC$_{last}$', 'cPC$_{last-1}$')



###
# gcPCA plots
gcPCA_mdl = gcPCA(method='v4', normalize_flag=False)
gcPCA_mdl.fit(data_a, data_b)

ax = plt.subplot(grid3[0, 0])
V_temp = gcPCA_mdl.loadings_[:,:2]
plotting_projection(data_a,V_temp,hsv_col,ax,'gcPC$_{1}$','gcPC$_{2}$')
plt.title(r'$\frac{\mathbf{C}_A-\mathbf{C}_B}{\mathbf{C}_A+\mathbf{C}_B}$',fontsize=18)

ax = plt.subplot(grid3[1, 0])
V_temp = gcPCA_mdl.loadings_[:,-2:][:,[1,0]]
plotting_projection(data_b,V_temp,spec_col,ax,'gcPC$_{last}$','gcPC$_{last-1}$')

plt.show()
fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_projections.pdf'), transparent=True)
fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_projections.png'), transparent=True)