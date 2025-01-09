# -*- coding: utf-8 -*-
"""
@author: Eliezyer Fermino de Oliveira

Script to run LDA on condition A vs B and show what it does reveal the gcPCA and cPCA analysis in the face with emotional vs neutral expressions

This script requires you to run the matlab script first to select and preprocess the Chicago Face Dataset images.
"""

# libraries
import numpy as np
from scipy.linalg import norm
from scipy.stats import zscore
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import sys
import os

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 150, 'font.size': 24, 'font.family': 'Arial'})

# repository dir
# change repo_dir to your directory location
repo_dir = '/home/eliezyer/Documents/github/generalized_contrastive_PCA/'
sys.path.append(repo_dir)
from contrastive_methods import gcPCA
from contrastive import CPCA


# change save_fig_path to the directory to save the figures
save_fig_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/figure_manuscripts/figures_gcPCA/Supplemental_figures/Supplemental_LDA_fig/source_plots/'

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

# function to plot scatter with marginal distribution
def scatter_marginal_plot(data_happy_angry,data_neutral,labels,custom_colors,m_size):
    fig, ax = plt.subplots()
    ax.scatter(data_happy_angry[labels == 0, 0], data_happy_angry[labels == 0, 1], s=m_size, c=custom_colors[0], label='Happy', alpha=0.8,
               edgecolors='w')
    ax.scatter(data_happy_angry[labels == 1, 0], data_happy_angry[labels == 1, 1], s=m_size, c=custom_colors[1], label='Angry', alpha=0.8,
               edgecolors='w')
    ax.scatter(data_neutral[:, 0], data_neutral[:, 1], s=m_size, c=custom_colors[2], label='Neutral', alpha=0.8,
               edgecolors='w')

    # create marginal axes
    ax_histx = ax.inset_axes([0, 1.01, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.01, 0, 0.25, 1], sharey=ax)
    # remove ticks
    ax_histx.set_xticks([])
    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    ax_histy.set_yticks([])

    # turn specific spines from ax white
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax_histx.spines['top'].set_color('white')
    ax_histx.spines['right'].set_color('white')
    ax_histx.spines['left'].set_color('white')
    ax_histy.spines['top'].set_color('white')
    ax_histy.spines['right'].set_color('white')
    ax_histy.spines['bottom'].set_color('white')

    # marginal histograms
    temp_x_hist = np.min([data_neutral[:, 0].min(), data_happy_angry[:, 0].min()])
    min_x_hist = temp_x_hist - 0.1*np.abs(temp_x_hist)
    temp_x_hist = np.max([data_neutral[:, 0].max(), data_happy_angry[:, 0].max()])
    max_x_hist = temp_x_hist + 0.1*np.abs(temp_x_hist)

    bins = np.linspace(min_x_hist, max_x_hist, 31)

    x = bins[:-1]+np.diff(bins)/2
    histc,_ = np.histogram(data_neutral[:,0], bins=bins)
    ax_histx.bar(x,histc/histc.max(), width=np.diff(bins), color=custom_colors[2], alpha=0.7, edgecolor=custom_colors[2], linewidth=0.5)
    histc,_ = np.histogram(data_happy_angry[labels == 0,0], bins=bins)
    ax_histx.bar(x,histc/histc.max(), width=np.diff(bins), color=custom_colors[0], alpha=0.7, edgecolor=custom_colors[0], linewidth=0.5)
    histc,_ = np.histogram(data_happy_angry[labels == 1,0], bins=bins)
    ax_histx.bar(x,histc/histc.max(), width=np.diff(bins), color=custom_colors[1], alpha=0.7, edgecolor=custom_colors[1], linewidth=0.5)

    # setting scatter plot xlim to the same as the histogram
    ax.set_xlim((min_x_hist, max_x_hist))

    temp_y_hist = np.min([data_neutral[:, 1].min(), data_happy_angry[:, 1].min()])
    min_y_hist = temp_y_hist - 0.1*np.abs(temp_y_hist)
    temp_y_hist = np.max([data_neutral[:, 1].max(), data_happy_angry[:, 1].max()])
    max_y_hist = temp_y_hist + 0.1*np.abs(temp_y_hist)

    bins = np.linspace(min_y_hist, max_y_hist, 31)
    x = bins[:-1]+np.diff(bins)/2

    histc,_ = np.histogram(data_neutral[:,1], bins=bins)
    ax_histy.barh(x, width=histc/histc.max(), height=np.diff(bins), color=custom_colors[2], alpha=0.7, edgecolor=custom_colors[2], linewidth=0.5)
    histc,_ = np.histogram(data_happy_angry[labels == 0,1], bins=bins)
    ax_histy.barh(x, width=histc/histc.max(), height=np.diff(bins), color=custom_colors[0], alpha=0.7, edgecolor=custom_colors[0], linewidth=0.5)
    histc,_ = np.histogram(data_happy_angry[labels == 1,1], bins=bins)
    ax_histy.barh(x, width=histc/histc.max(), height=np.diff(bins), color=custom_colors[1], alpha=0.7, edgecolor=custom_colors[1], linewidth=0.5)
    ax_histy.set_ylim((min_y_hist, max_y_hist))

    # setting scatter plot xlim to the same as the histogram
    ax.set_ylim((min_y_hist, max_y_hist))

    fig.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
    fig.set_size_inches(5,5)
    plt.legend(fontsize=14)
    return fig, ax

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
U_gcpca = gcPCA_mdl.Ra_scores_*gcPCA_mdl.Ra_values_
U_gcpca_neutral = gcPCA_mdl.Rb_scores_*gcPCA_mdl.Rb_values_

gcpcs = gcPCA_mdl.loadings_
temp1 = gcpcs[:, 0]
image_gcpc1 = np.reshape(temp1, (data_A.shape[0], data_A.shape[1])).copy()
temp1 = gcpcs[:, 1]
image_gcpc2 = np.reshape(temp1, (data_A.shape[0], data_A.shape[1])).copy()

# running cPCA for multiple alphas
projected_data = []
for a in best_alphas:
    projected_data.append(cpca_mdl.fit_transform(newN1, newN2, alpha_value=a, alpha_selection='manual'))

# LDA
# running LDA in A vs B (using the PCA reduced data to help LDA)
lda_mdl = LDA()
data_lda = np.concatenate((newN1, newN2), axis=0)
labels_lda = np.concatenate((np.zeros(A_norm.shape[0]), np.ones(B_norm.shape[0])), axis=0)
lda_mdl.fit(data_lda, labels_lda)

# plot parameters
m_size = 100  # marker size for scatter

###
# making joint plot of scatter with marginal histograms
#

# plotting gcPCA
custom_colors = ["#1833FA", "#FA0700", "#808080"]
labels = labels.flatten()
fig, ax = scatter_marginal_plot(U_gcpca, U_gcpca_neutral, labels, custom_colors, m_size)
plt.xlabel('gcPC$_1$', fontsize=24)
plt.ylabel('gcPC$_2$', fontsize=24)

plt.savefig(os.path.join(save_fig_dir, "face_expression_gcPCA_projection.pdf"), format="pdf", transparent=True)


### LDA plots
lda_mdl = LDA()
lbls_happy = np.argwhere(labels == 0)[:,0]
lbls_angry = np.argwhere(labels == 1)[:,0]

# training LDA on condition A vs B
data_lda = np.concatenate((newN1[lbls_happy, :], newN1[lbls_angry, :], newN2), axis=0)
labels_lda = np.concatenate((np.zeros(newN1.shape[0]), np.ones(newN2.shape[0])), axis=0)
labels_classes = np.concatenate((labels[:,np.newaxis], 2*np.ones((newN2.shape[0],1))), axis=0)
string_labels = ['Happy' if i==0 else 'Angry' if i==1 else 'Neutral' for i in labels_classes.flatten()]
cond_A_labels = ['Happy' if i==0 else 'Angry' for i in labels.flatten()]
lda_mdl.fit(data_lda, labels_lda)
data_projected_condA_vs_condB = data_lda.dot(lda_mdl.scalings_)

# training LDA on happy vs angry
data_lda_2 = np.concatenate((newN1[lbls_happy, :], newN1[lbls_angry, :]), axis=0)
labels_lda = np.concatenate((np.zeros(len(lbls_happy)), np.ones(len(lbls_angry))), axis=0)
lda_mdl.fit(data_lda_2, labels_lda)
data_projected_happy_vs_angry = data_lda.dot(lda_mdl.scalings_)  # projecting also the neutral faces

data_lda_happy_angry = np.concatenate((data_projected_happy_vs_angry[:newN1.shape[0]], data_projected_condA_vs_condB[:newN1.shape[0]]), axis=1)
data_lda_neutral = np.concatenate((data_projected_happy_vs_angry[newN1.shape[0]:], data_projected_condA_vs_condB[newN1.shape[0]:]), axis=1)

# plotting
fig, ax = scatter_marginal_plot(data_lda_happy_angry, data_lda_neutral, labels, custom_colors, m_size)
plt.xlabel('LDA(Happy, Angry)', fontsize=24)
plt.ylabel('LDA(cond. A, cond. B)', fontsize=24)

plt.savefig(os.path.join(save_fig_dir, "face_expression_LDA_projection_happy_angry.pdf"), format="pdf", transparent=True)