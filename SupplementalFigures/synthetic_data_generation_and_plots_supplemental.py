"""
@author: Eliezyer Fermino de Oliveira
Script to generate synthetic data with linear manifolds for the gcPCA paper. It creates a synthetic dataset with a
 change in variance between conditions that are uneven, i.e., from condition B to A is a certain amount of change, and
 from B to A is another amount."""


import sys
import numpy as np
from scipy.stats import zscore
from scipy.linalg import orth
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# change repo_dir to where you downloaded the gcPCA class
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"  # repository dir on laptop
sys.path.append(repo_dir)
from contrastive_methods import gcPCA

plt.rcParams.update({'figure.dpi': 150, 'font.size': 15,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

# change save fig directory to where you want to save te figure
save_fig_dir = "/mnt/extraSSD4TB/CloudStorage/Dropbox/figure_manuscripts/figures_gcPCA/toy_data/"  # path to save figures


# %% parameters
N_samples = 100000
N_features = 100
np.random.seed(5) #seed for reproducibility

# function to extract cPCA loadings
def extract_cpca_loadings(A_samples, B_samples, alpha_value):
    A_cov = A_samples.T.dot(A_samples)/ (A_samples.shape[0]-1)
    B_cov = B_samples.T.dot(B_samples)/ (B_samples.shape[0]-1)

    # eigendecomposition
    sigma = A_cov - alpha_value*B_cov
    e,w = np.linalg.eigh(sigma)
    idx = np.argsort(e)[::-1]
    e = e[idx]
    w = w[:,idx]
    return w

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plotting_projection(data_plot,loadings,col_array,ax,xlabel,ylabel):
    plt.scatter(data_plot @ loadings[:, 0], data_plot @ loadings[:, 1], c=col_array, s=7)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    plt.tight_layout()
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
# %% defining
# function to generate synthetic data
def generate_data(N_samples, N_features):
    increased_factor_A = [1.5, 1.7]
    increased_factor_B = [1.3, 2]

    dim_in_A = [19, 92]
    dim_in_B = [31, 82]

    # generating toy data

    # ancillary function to generate latent_Factors 1 through 4
    def generate_latent_factors(N_samples):
        # factors that increase in A
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

        # factors increased in B
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
    # eigenvalue
    Sa = (np.linspace(0, stop=10, num=N_features)[::-1] + 10 ** -4)
    Sa[dim_in_A[0]] = increased_factor_A[0] * Sa[dim_in_A[0]]  # boosting the dimensions' eigenvalue
    Sa[dim_in_A[1]] = increased_factor_A[1] * Sa[dim_in_A[1]]  # boosting the dimensions' eigenvalue

    # getting orthogonal weights
    W = orth(np.random.randn(N_features, N_features)).T

    # generating samples from the low variance manifold in condition A
    samples1 = np.outer(latent_factor1, Sa[dim_in_A[0]] * W[dim_in_A[0], :])
    samples2 = np.outer(latent_factor2, Sa[dim_in_A[1]] * W[dim_in_A[1], :])

    # generating samples from the other factors
    auxSm = np.repeat(Sa[:dim_in_A[0], np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[:dim_in_A[0], :])
    rest_samples1 = np.dot(rest_factors1[:, :dim_in_A[0]], newW)

    auxSm = np.repeat(Sa[dim_in_A[0]+1:dim_in_A[1], np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[dim_in_A[0]+1:dim_in_A[1], :])
    rest_samples2 = np.dot(rest_factors1[:, dim_in_A[0]+1:dim_in_A[1]], newW)

    auxSm = np.repeat(Sa[dim_in_A[1]+1:, np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[dim_in_A[1]+1:, :])
    rest_samples3 = np.dot(rest_factors1[:, dim_in_A[1]+1:], newW)

    # adding for final data
    data_A = samples1 + samples2 + rest_samples1 + rest_samples2 + rest_samples3

    # generating data_B
    # eigenvalues
    Sb = (np.linspace(0, stop=10, num=N_features)[::-1] + 10 ** -4)
    Sb[dim_in_B[0]] = increased_factor_B[0] * Sb[dim_in_B[0]]  # boosting the dimensions' eigenvalue
    Sb[dim_in_B[1]] = increased_factor_B[1] * Sb[dim_in_B[1]]  # boosting the dimensions' eigenvalue

    # generating samples from the low variance manifold in condition B
    samples3 = np.outer(latent_factor3, Sb[dim_in_B[0]] * W[dim_in_B[0], :])
    samples4 = np.outer(latent_factor4, Sb[dim_in_B[1]] * W[dim_in_B[1], :])

    # other factors
    rest_factors1 = zscore(np.random.randn(N_samples, N_features))
    rest_factors1 = rest_factors1 / np.linalg.norm(rest_factors1, axis=0)

    # generating samples from the other factors
    auxSm = np.repeat(Sb[:dim_in_B[0], np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[:dim_in_B[0], :])
    rest_samples1 = np.dot(rest_factors1[:, :dim_in_B[0]], newW)

    auxSm = np.repeat(Sb[dim_in_B[0] + 1:dim_in_B[1], np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[dim_in_B[0] + 1:dim_in_B[1], :])
    rest_samples2 = np.dot(rest_factors1[:, dim_in_B[0] + 1:dim_in_B[1]], newW)

    auxSm = np.repeat(Sb[dim_in_B[1]+1:, np.newaxis], N_features, axis=1)
    newW = np.multiply(auxSm, W[dim_in_B[1]+1:, :])
    rest_samples3 = np.dot(rest_factors1[:, dim_in_B[1]+1:], newW)

    # adding for final data
    data_B = samples3 + samples4 + rest_samples1 + rest_samples2 + rest_samples3

    return data_B, data_A, W, dim_in_B, dim_in_A


# %%parameters for generating data with 2 tracks
N_samples2plot = 1000  # it has to be multiples of 1k
data_B, data_A, W, dim_B, dim_A = generate_data(N_samples, N_features)
data_b = data_B[:N_samples2plot, :]
data_a = data_A[:N_samples2plot, :]

# calculating eigenspectrum
temp = data_a @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
fg_eigspec_sampled = np.sum(R ** 2, axis=1)

temp = data_A @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
fg_eigspec_true = np.sum(R ** 2, axis=1)

temp = data_b @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
bg_eigspec_sampled = np.sum(R ** 2, axis=1)

temp = data_B @ W.T
_, R = np.linalg.qr(temp.T.dot(temp))
bg_eigspec_true = np.sum(R ** 2, axis=1)

hsv = mpl.colormaps.get_cmap('hsv')
spec_cmap = mpl.colormaps.get_cmap('PRGn')
temp = np.linspace(0, 1, 1000)
hsv_col = hsv(np.tile(temp, int(N_samples2plot / 1000)))
spec_col = spec_cmap(np.tile(temp, int(N_samples2plot / 1000)))

# %% plot the second row of figure 1, showing all the methods solutions in a single row
fig = plt.figure(num=0, figsize=(13, 4))
grid1 = plt.GridSpec(1, 3, left=0.05, right=0.99, bottom=0.15, top=0.83,wspace=0.2)

###
# plot of A-B infinite vs finite data
ax = plt.subplot(grid1[0])
A_sampled = fg_eigspec_sampled
B_sampled = bg_eigspec_sampled
A_true = fg_eigspec_true
B_true = bg_eigspec_true

x = np.arange(1, 101, step=1)
light_royal_blue = lighten_color('royalblue', amount=0.5)
plt.plot(x, (A_sampled - B_sampled), color='royalblue', linestyle='-', label='Finite Data',
         linewidth=4, zorder=4)
plt.plot(x[dim_A[0]], (A_sampled - B_sampled)[dim_A[0]], color='royalblue', marker='*', markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=4)
plt.plot(x[dim_A[1]], (A_sampled - B_sampled)[dim_A[1]], color='royalblue', marker='*', markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=4)
plt.plot(x[dim_B[0]], (A_sampled - B_sampled)[dim_B[0]], color='royalblue', marker='*', markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=4)
plt.plot(x[dim_B[1]], (A_sampled - B_sampled)[dim_B[1]], color='royalblue', marker='*', markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=4)
plt.xticks(dim_A+dim_B)


x = np.arange(1, 101, step=1)
plt.plot(x, (A_true - B_true)/1e4, color=light_royal_blue, linestyle='-', label='Infinite Data', linewidth=4, zorder=5)
plt.grid(color='grey', linestyle='-', linewidth=2, alpha=0.4, zorder=0)

plt.title(r'$\mathbf{C}_A-\mathbf{C}_B$')
plt.ylabel('Value (A.U.)')
plt.yticks((-0.2, 0, 0.2))
plt.ylabel('Value (A.U.)')
plt.xlabel('Dimensions')
plt.legend(loc='upper right',fontsize=12)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

###
# plot of cPCA
ax = plt.subplot(grid1[1])
x = np.arange(1, 101, step=1)
inferno = mpl.colormaps.get_cmap('inferno')
inferno_col = inferno(np.linspace(0, 1, 6))
c = 0
alpha_list = [0.4, 0.7, 1.0, 1.3, 1.9]
for alpha in alpha_list:
    plt.plot(x, fg_eigspec_sampled - alpha * bg_eigspec_sampled,
             color=inferno_col[c],
             linestyle='-',
             linewidth=4,
             zorder=2,
             label='α: ' + str(alpha))
    c += 1

plt.title('$\mathbf{C}_A-α*\mathbf{C}_B$')
plt.xticks(dim_A+dim_B)
plt.yticks((-1, 0, 1))
plt.ylim([-2,2])
plt.ylabel('Value (A.U.)')
plt.grid(color='grey',
         linestyle='-',
         linewidth=2,
         alpha=0.4,
         zorder=0)

plt.xlabel('Dimensions')
plt.tight_layout()
plt.legend(loc=[0.55,0.01],fontsize=12)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
###
# plot of gcPCa objective function
ax = plt.subplot (grid1[2])
x = np.arange(1, 101, step=1)
plt.plot(x, ((A_sampled - B_sampled) / (A_sampled + B_sampled)),
         color='firebrick', linestyle='-', linewidth=4, zorder=2)
plt.xticks(dim_A+dim_B)
plt.yticks((-0.5, 0, 0.5))
plt.ylabel('Value (A.U.)')
plt.grid(color='grey',
         linestyle='-', linewidth=2, alpha=0.4, zorder=0)
plt.title(r'$\frac{\mathbf{C}_A-\mathbf{C}_B}{\mathbf{C}_A+\mathbf{C}_B}$')

plt.xlabel('Dimensions')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()

fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_functions_s1.pdf'), transparent=True)
fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_functions_s1.png'), transparent=True)

# %% third row of figure 1: what each contrastive method reveal
fig = plt.figure(num=1, figsize=(13, 4))
grid2 = plt.GridSpec(2, 5, figure=fig, left=0.03, right=0.8, wspace=0.3)
grid3 = plt.GridSpec(2, 1, figure=fig, left=0.85, right=0.99)


###
# cPCA different alphas
alpha_list2 = [2.2,2.3,2.6,2.8,3.0]
for c,alp in enumerate(alpha_list2):
    V = extract_cpca_loadings(data_a, data_b, alp)
    ax = plt.subplot(grid2[0, c])
    V_temp = V[:, :2]
    plotting_projection(data_a, V_temp, hsv_col, ax, 'cPC$_{1}$', 'cPC$_{2}$')
    # write the title and add the alpha value in the text with 2 floating decimal points precision
    title = r'$\mathbf{C}_A - $' + f'{alp:.1f}' + '$\mathbf{C}_B$'
    plt.title(title,fontsize=18)  # add alpha here

    ax = plt.subplot(grid2[1, c])
    V_temp = V[:, -2:][:, [1, 0]]
    plotting_projection(data_b, V_temp, spec_col, ax, 'cPC$_{last}$', 'cPC$_{last-1}$')

###
# gcPCA plots
gcPCA_mdl = gcPCA(method='v4', normalize_flag=False)
gcPCA_mdl.fit(data_a, data_b)

ax = plt.subplot(grid3[0, 0])
V_temp = gcPCA_mdl.loadings_[:, :2]
plotting_projection(data_a, V_temp, hsv_col, ax, 'gcPC$_{1}$', 'gcPC$_{2}$')
plt.title(r'$\frac{\mathbf{C}_A-\mathbf{C}_B}{\mathbf{C}_A+\mathbf{C}_B}$',fontsize=18)

ax = plt.subplot(grid3[1, 0])
V_temp = gcPCA_mdl.loadings_[:, -2:][:, [1, 0]]
plotting_projection(data_b, V_temp, spec_col, ax, 'gcPC$_{last}$', 'gcPC$_{last-1}$')

plt.show()
fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_projections_s1.pdf'), transparent=True)
fig.savefig(os.path.join(save_fig_dir, 'contrastive_methods_projections_s1.png'), transparent=True)