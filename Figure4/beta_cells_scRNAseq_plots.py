""" plot to make the exploratory analysis of gcPCA on beta cells from jens dataset"""

# importing essentials
import os
import numpy as np
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')

# change the repo_dir to the folder where you downloaded the gcPCA
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/" #repository dir in data.einsteinmed

sys.path.append(repo_dir)
from contrastive_methods import gcPCA

# defining ancillary functions
# change the data path to where you downloaded the data
data_path = "/mnt/extraSSD4TB/CloudStorage/Dropbox/preprocessing_data/gcPCA_files/Jens_data/scRNA_seq/"
save_path = data_path
os.chdir(data_path)
save_fig_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/figures_gcPCA/Figure4/source_plots/'

# loadings negative and positive rank of beta cells
# FIX THIS
neg_rank = pd.read_csv(os.path.join(data_path, 'NegRank_BetaCells_dataset1.csv'), delimiter=' ', index_col=None, header=None)
neg_rank.columns = ['gene', 'rank']
pos_rank = pd.read_csv(os.path.join(data_path, 'PosRank_BetaCells_dataset1.csv'), delimiter=' ', index_col=None, header=None)
pos_rank.columns = ['gene', 'rank']

# loading data in as pandas dataframe
data = pd.read_table(os.path.join(data_path, 'GSE153855_Expression_RPKM_HQ_allsamples.txt'), sep='\t',
                     index_col=0, header=None).T

annotation = pd.read_csv(os.path.join(data_path,'GSE153855_Cell_annotation.txt'), sep='\t')

# picking only beta cells of T2D and normal patients
diabetes_beta_df = data[(annotation.CellType.values == 'Beta') & (annotation.Disease.values == 'type II diabetes')]
normal_beta_df = data[(annotation.CellType.values == 'Beta') & (annotation.Disease.values == 'normal')]

# finding donors identities for T2D and normal
temp = annotation.Donor[(annotation.CellType.values == 'Beta') & (annotation.Disease.values == 'type II diabetes')].values
donors_diabetes = np.zeros(temp.shape)
for c,a in enumerate(np.unique(temp)):
    donors_diabetes[np.isin(temp, a)] = c

temp = annotation.Donor[(annotation.CellType.values == 'Beta') & (annotation.Disease.values == 'normal')].values
donors_normal = np.zeros(temp.shape)
for c,a in enumerate(np.unique(temp)):
    donors_normal[np.isin(temp, a)] = c

donors_diabetes = donors_diabetes.astype(int)
donors_normal = donors_normal.astype(int)

# getting all genes names
all_gene_names = data.columns

# log transforming the data
tempN1 = np.log2(diabetes_beta_df.values+1)
tempN2 = np.log2(normal_beta_df.values+1)

###
# keeping only genes that were used in previous analysis (Martinez-Lopez et al)
temp_whole_data = np.concatenate((diabetes_beta_df.values, normal_beta_df.values), axis=0)
temp_donors = np.concatenate((donors_diabetes, donors_normal+donors_diabetes.max()+1), axis=0)

genes_to_keep_final = np.isin(all_gene_names, neg_rank.gene.values)

N1_red = tempN1[:, genes_to_keep_final]
N2_red = tempN2[:, genes_to_keep_final]
genes_used = all_gene_names[genes_to_keep_final]
# centering tthe data
N1 = N1_red - np.mean(N1_red, axis=0)
N2 = N2_red - np.mean(N2_red, axis=0)

# fitting gcPCA
gcpca_mdl = gcPCA(method='v4', normalize_flag=False)
gcpca_mdl.fit(N1, N2)  # N1 is diabetes and N2 is normal, all beta cells

# plot of the clusters
plt.rcParams.update({'figure.dpi':150, 'font.size':12})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
marker_size = 30

plt.figure(figsize=(4, 4))
plt.scatter(gcpca_mdl.Ra_scores_[:, 0],
            gcpca_mdl.Ra_scores_[:, 1], s=marker_size, c='k')
plt.xlabel('gcPC$_1$')
plt.ylabel('gcPC$_2$')
plt.title('T2D Beta Cells ')
plt.show()

plt.figure(figsize=(4, 4))
plt.scatter(gcpca_mdl.Rb_scores_[:, 0],
            gcpca_mdl.Rb_scores_[:, 1], s=marker_size, c='k')
plt.xlabel('gcPC$_1$')
plt.ylabel('gcPC$_2$')
plt.show()

###
# plotting scores where each colors is a subject, this is 4 panel figure
#
plt.rcParams.update({'figure.dpi':150, 'font.size':18})
cmap_set1 = plt.get_cmap('Set1')
cmap_set2 = plt.get_cmap('Set2')
xlim_const = [-0.23, 0.23]
ylim_const = [-0.29, 0.29]
fig, (axs) = plt.subplots(2, 2, figsize=(8, 8))
# plotting scores where each colors is a subject, for T2D
plt.subplot(221)
for subject in np.unique(donors_diabetes):
    subject_to_plot = donors_diabetes == subject
    plt.scatter(gcpca_mdl.Ra_scores_[subject_to_plot, 0],
                gcpca_mdl.Ra_scores_[subject_to_plot, 1], s=marker_size, label=subject, c=cmap_set1(subject))
# plt.legend(title='Donors', loc=(0.6, -0.2), fontsize=14)
plt.xlabel('gcPC$_1$')
plt.ylabel('gcPC$_2$')
plt.title('T2D Beta Cells')
plt.xticks([])
plt.yticks([])
plt.xlim(xlim_const)
plt.ylim(ylim_const)
plt.tight_layout()

plt.subplot(222)
for subject in np.unique(donors_diabetes):
    subject_to_plot = donors_diabetes == subject
    plt.scatter(gcpca_mdl.Ra_scores_[subject_to_plot, -1],
                gcpca_mdl.Ra_scores_[subject_to_plot, -2], s=marker_size, label=subject, c=cmap_set1(subject))
# plt.legend(title='Donors',loc=(0.85, 0.009))
plt.xlabel('gcPC$_{last}$')
plt.ylabel('gcPC$_{last-1}$')
plt.title('T2D Beta Cells')
plt.xticks([])
plt.yticks([])
plt.xlim(xlim_const)
plt.ylim(ylim_const)
plt.tight_layout()

plt.subplot(223)
for subject in np.unique(donors_normal):
    subject_to_plot = donors_normal == subject
    plt.scatter(gcpca_mdl.Rb_scores_[subject_to_plot, 0],
                gcpca_mdl.Rb_scores_[subject_to_plot, 1], s=marker_size, label=subject, c=cmap_set2(subject))
# plt.legend(title='Donors', loc=(0.7, -0.2), fontsize=14)
plt.xlabel('gcPC$_1$')
plt.ylabel('gcPC$_2$')
plt.title('Control Beta Cells')
plt.xticks([])
plt.yticks([])
plt.xlim(xlim_const)
plt.ylim(ylim_const)
plt.tight_layout()

plt.subplot(224)
for subject in np.unique(donors_normal):
    subject_to_plot = donors_normal == subject
    plt.scatter(gcpca_mdl.Rb_scores_[subject_to_plot, -1],
                gcpca_mdl.Rb_scores_[subject_to_plot, -2], s=marker_size, label=subject, c=cmap_set2(subject))
# plt.legend(title='Donors',loc=(0.85, 0.009))
plt.xlabel('gcPC$_{last}$')
plt.ylabel('gcPC$_{last-1}$')
plt.title('Control Beta Cells')
plt.xticks([])
plt.yticks([])
plt.xlim(xlim_const)
plt.ylim(ylim_const)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(save_fig_dir, 'gcPCA_scores_t2d_control.png'))
plt.savefig(os.path.join(save_fig_dir, 'gcPCA_scores_t2d_control.pdf'))

###
# make a plot of truncated gcPCs, showing the top 40 loadings
#
plt.rcParams.update({'figure.dpi':150, 'font.size':12})
plt.figure(figsize=(4, 8))
temp_idx = np.argsort((gcpca_mdl.loadings_[:, 0]))
idx_extremeties = np.concatenate((temp_idx[:20], temp_idx[-20:]))[::-1]
plt.stem(gcpca_mdl.loadings_[idx_extremeties, 0],orientation='horizontal')
# add gene names to xtick labels
plt.yticks(np.arange(len(idx_extremeties)), genes_used[idx_extremeties])
plt.xlabel('Loadings',fontsize=18)
plt.ylabel('Genes',fontsize=18)
plt.title('gcPC1')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(save_fig_dir, 'gcPCA_loadings_truncated.png'))
plt.savefig(os.path.join(save_fig_dir, 'gcPCA_loadings_truncated.pdf'))