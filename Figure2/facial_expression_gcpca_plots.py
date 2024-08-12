# -*- coding: utf-8 -*-
"""
@author: Eliezyer Fermino de Oliveira

Script to run the gcPCA and cPCA analysis in the face with emotional vs neutral expressions

This script requires you to run the matlab script first to select and preprocess the Chicago Face Dataset images.
"""

# libraries
import numpy as np
from scipy.linalg import norm
from scipy.stats import zscore
from scipy.io import loadmat

import sys
import os

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 150, 'font.size': 24})
import seaborn as sns

# repository dir
# change repo_dir to your directory location
repo_dir = '/home/eliezyer/Documents/github/generalized_contrastive_PCA/'
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
from contrastive import CPCA

# change save_fig_path to the directory to save the figures
save_fig_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/Figure2/source_plots/'


###
# defining ancillary functions
# A function for flooring a number with decimal precision
def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10 ** precision), 10 ** precision)


# function for plotting the dimensions found in contrastive methods
def face_cPCs_plot(face_cpc1, face_cpc2, Mask, cmap, numfig, gcPC_flag=False):
    fig = plt.figure(num=numfig, figsize=(7, 5))
    m_lim = np.max(np.abs(face_cpc1 * Mask)) / 1.0

    grid1 = plt.GridSpec(1, 2, left=0, right=0.79, wspace=0.05)

    plt.subplot(grid1[0])
    plt.imshow(face_cpc1 * Mask, cmap=cmap, aspect='auto')
    if gcPC_flag:
        plt.title('gcPC$_1$', fontsize=24)
    else:
        plt.title('cPC$_1$', fontsize=24)
    plt.xlim((35, 145))
    plt.ylim((205, 45))
    plt.clim(-1 * m_lim, m_lim)
    plt.axis('off')

    ax = plt.subplot(grid1[1])
    auxp = plt.imshow(face_cpc2 * Mask, cmap=cmap, aspect='auto')
    if gcPC_flag:
        plt.title('gcPC$_2$', fontsize=24)
    else:
        plt.title('cPC$_2$', fontsize=24)
    plt.xlim((35, 145))
    plt.ylim((205, 45))
    plt.clim(-1 * m_lim, m_lim)

    cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])

    # legend colorbar
    cbar = fig.colorbar(auxp, cax=cax, ticks=[-1 * my_floor(m_lim, precision=2), 0, my_floor(m_lim, precision=2)])
    cbar.set_label('Magnitude', rotation=270, labelpad=18, fontsize=22)
    plt.axis('off')
    plt.tight_layout()


# function to extract the face cPCs loadings
def extract_face_cPCs(A_samples, B_samples, PCA_mdl, best_alphas, image_dimensions=(246, 186)):
    """get the normalized and preprocessed faces samples
    and apply the PCA model to extract the cPCs and project it back, this is the same process as cPCA"""
    Acov = A_samples.T.dot(A_samples) / (A_samples.shape[0] - 1)
    Bcov = B_samples.T.dot(B_samples) / (B_samples.shape[0] - 1)

    image_cpc1 = []
    image_cpc2 = []
    image_cpc_last = []
    for a in best_alphas:
        sigma = Acov - a * Bcov
        e, w = np.linalg.eig(sigma)
        idx = np.argsort(e)[::-1]
        e = e[idx]
        w2 = PCA_mdl.components_.T.dot(w[:, idx])
        image_cpc1.append(np.reshape(w2[:, 0], image_dimensions).copy())
        image_cpc2.append(np.reshape(w2[:, 1], image_dimensions).copy())
        image_cpc_last.append(np.reshape(w2[:, -1], image_dimensions).copy())

    return image_cpc1, image_cpc2, image_cpc_last


###
# starting analysis

# loading data
# change data_dir to the location where you saved the file face_emotions.mat
data_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/preprocessing_data/gcPCA_files/face/CFD_V3/Images/CFD'

# extract the pre-processed data
tempmat = loadmat(os.path.join(data_dir, 'face_emotions.mat'))
data_A = tempmat['face_emotions']['data_A'][0][0]
data_B = tempmat['face_emotions']['data_B'][0][0]
labels = tempmat['face_emotions']['labels'][0][0]
Mask = tempmat['face_emotions']['EllipseMask'][0][0]

# reshaping and normalizing data condition A
A = np.reshape(data_A, (data_A.shape[0] * data_A.shape[1], data_A.shape[2]))
A_zsc = zscore(A.T)
A_zsc[np.isnan(A_zsc)] = 0
temp = norm(A_zsc, axis=0)
temp[temp == 0] = 1
A_norm = A_zsc / temp

# reshaping and normalizing data condition B
B = np.reshape(data_B, (data_B.shape[0] * data_B.shape[1], data_B.shape[2]))
B_zsc = zscore(B.T)
B_zsc[np.isnan(B_zsc)] = 0
temp = norm(B_zsc, axis=0)
temp[temp == 0] = 1
B_norm = B_zsc / temp

# contrasting condition A and B with cPCA
preprocess_with_pca_dim = np.min([A_norm.shape[0], B_norm.shape[0]])
data = np.concatenate((A_norm, B_norm), axis=0)
from sklearn.decomposition import PCA

pca = PCA(n_components=preprocess_with_pca_dim)
data = pca.fit_transform(data)
newN1 = data[:A_norm.shape[0], :]
newN2 = data[A_norm.shape[0]:, :]

cpca_mdl = CPCA(standardize=False, verbose=True, n_components=2)
cpca_mdl.fit(newN1, newN2)

# running automated cpca to get best alphas
np.random.seed(0)  # setting seed for reproducibility
projected_data, best_alphas = cpca_mdl.automated_cpca(newN1, n_alphas=40, max_log_alpha=3, n_alphas_to_return=5)
best_alphas = best_alphas[1:]  # first cPCA is always alpha=0, i.e., just PCA on condition A

# extracting face cPCs from each alpha
face_cpc1, face_cpc2, _ = extract_face_cPCs(newN1, newN2, pca, best_alphas, image_dimensions=(246, 186))

# contrasting conditions with gcPCA and getting the two first gcPCs
gcPCA_mdl = gcPCA(method='v3', normalize_flag=True)
gcPCA_mdl.fit(A_norm, B_norm)
U_gcpca = gcPCA_mdl.Ra_scores_

gcpcs = gcPCA_mdl.loadings_
temp1 = gcpcs[:, 0]
image_gcpc1 = np.reshape(temp1, (data_A.shape[0], data_A.shape[1])).copy()
temp1 = gcpcs[:, 1]
image_gcpc2 = np.reshape(temp1, (data_A.shape[0], data_A.shape[1])).copy()

# running cPCA for multiple alphas
projected_data = []
for a in best_alphas:
    projected_data.append(cpca_mdl.fit_transform(newN1, newN2, alpha_value=a, alpha_selection='manual'))

# %% making figures

# plot parameters
m_size = 100  # marker size for scatter
subject = 5  # which subject to show
neutral_example = subject  # neutral expression example
hc_example = np.argwhere(labels == 0)[:, 0][subject]  # happy mouth closed expression example
angry_example = np.argwhere(labels == 1)[:, 0][subject]  # angry expression example

from matplotlib import colors as clrs

sns.set_style("ticks")
sns.set_context("talk")
# making custom colormap and adding as the default
cmap = clrs.LinearSegmentedColormap.from_list("", ["seagreen", "white", "blueviolet"])

# starting first plot: example of the conditions A and B
fig = plt.figure(num=1, figsize=(12, 5))
grid1 = plt.GridSpec(1, 2, left=0.0, right=0.53, wspace=0.05)
grid2 = plt.GridSpec(1, 1, left=0.70, right=0.96, wspace=0.05)

# plotting examples from dataset
# happy mouth close
plt.subplot(grid1[0])
temp = data_A[:, :, hc_example].astype(float) * Mask
temp2 = temp.copy()
happy_face = temp.copy()
happy_face[(temp2 == 0.0)] = np.nan
plt.imshow(happy_face, cmap='gray', aspect='auto')
plt.xlim((35, 145))
plt.ylim((205, 45))
plt.axis('off')
plt.title('Happy', fontsize=30, weight='bold')

# angry
plt.subplot(grid1[1])
temp = data_A[:, :, angry_example].astype(float) * Mask
temp2 = temp.copy()
angry_face = temp.copy()
angry_face[(temp2 == 0.0)] = np.nan
plt.imshow(angry_face, cmap='gray', aspect='auto')
plt.xlim((35, 145))
plt.ylim((205, 45))
plt.axis('off')
plt.title('Angry', fontsize=30, weight='bold')

# neutral
plt.subplot(grid2[0])
temp = data_B[:, :, neutral_example].astype(float) * Mask
temp2 = temp.copy()
happy_face = temp.copy()
happy_face[(temp2 == 0.0)] = np.nan
plt.imshow(happy_face, cmap='gray', aspect='auto')
plt.xlim((35, 145))
plt.ylim((205, 45))
plt.axis('off')
plt.title('Neutral', fontsize=30, weight='bold')

plt.savefig(os.path.join(save_fig_dir, "face_expression_dataset_conditions_panels.pdf"), format="pdf",
            transparent=True)

###
# plotting the dimensions as images

# cPCA first alpha
face_cPCs_plot(face_cpc1[0], face_cpc2[0], Mask, cmap, numfig=2)
plt.savefig(os.path.join(save_fig_dir, "face_expression_dataset_cPCs_a1_panels.pdf"), format="pdf",
            transparent=True)

# cPCA second alpha
face_cPCs_plot(face_cpc1[1], face_cpc2[1], Mask, cmap, numfig=3)
plt.savefig(os.path.join(save_fig_dir, "face_expression_dataset_cPCs_a2_panels_2.pdf"), format="pdf",
            transparent=True)

# gcPCA
face_cPCs_plot(image_gcpc1, image_gcpc2, Mask, cmap, numfig=5, gcPC_flag=True)
plt.savefig(os.path.join(save_fig_dir, "face_expression_dataset_gcPCs_panels.pdf"), format="pdf", transparent=True)

###
# plots of the projections

# labels of angry and happy
lbl_hc = (labels == 0).flatten()
lbl_a = (labels == 1).flatten()

# cPCA first alpha
fig = plt.figure(num=6, figsize=(5, 5))
ax = plt.subplot()
U = projected_data[0]
ax.scatter(U[lbl_hc, 0], U[lbl_hc, 1], c='blue', alpha=0.5, label='Happy', s=m_size)
ax.scatter(U[lbl_a, 0], U[lbl_a, 1], c='red', alpha=0.5, label='Angry', s=m_size)
ax.set_xlabel('cPC$_1$', fontsize=30)
ax.set_ylabel('cPC$_2$', fontsize=30)
plt.legend(loc='best', fontsize=20)
plt.yticks(my_floor(np.linspace(np.min(U[:, 1]), np.max(U[:, 1]), 3), 2))

fig.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(os.path.join(save_fig_dir, "face_expression_cPCA_projection_a1.pdf"), format="pdf", transparent=True)

# cPCA second alpha
fig = plt.figure(num=7, figsize=(5, 5))
ax = plt.subplot()
U = projected_data[1]
ax.scatter(U[lbl_hc, 0], U[lbl_hc, 1], c='blue', alpha=0.5, label='Happy', s=m_size)
ax.scatter(U[lbl_a, 0], U[lbl_a, 1], c='red', alpha=0.5, label='Angry', s=m_size)
ax.set_xlabel('cPC$_1$', fontsize=30)
ax.set_ylabel('cPC$_2$', fontsize=30)
plt.yticks(my_floor(np.linspace(np.min(U[:, 1]), np.max(U[:, 1]), 3), 2))

fig.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(os.path.join(save_fig_dir, "face_expression_cPCA_projection_a2.pdf"), format="pdf", transparent=True)

# gcPCA
fig = plt.figure(num=9, figsize=(5, 5))
ax = plt.subplot()
ax.scatter(U_gcpca[lbl_hc, 0], U_gcpca[lbl_hc, 1], c='blue', alpha=0.5, label='Happy', s=m_size)
ax.scatter(U_gcpca[lbl_a, 0], U_gcpca[lbl_a, 1], c='red', alpha=0.5, label='Angry', s=m_size)
ax.set_xlabel('gcPC$_1$', fontsize=30)
ax.set_ylabel('gcPC$_2$', fontsize=30)
plt.yticks(my_floor(np.linspace(np.min(U_gcpca[:, 1]), np.max(U_gcpca[:, 1]), 3), 2))

fig.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(os.path.join(save_fig_dir, "face_expression_gcPCA_projection.pdf"), format="pdf", transparent=True)
