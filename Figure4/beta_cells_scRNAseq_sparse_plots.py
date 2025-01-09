""" Script to analyze and plot the pancreatic single-cell RNA sequencing dataset"""

# importing essentials
import os
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# change the repo_dir to the folder where you downloaded the gcPCA
repo_dir = "/home/eliezyer/Documents/github/generalized_contrastive_PCA/"

sys.path.append(repo_dir)
from contrastive_methods import sparse_gcPCA, gcPCA

# defining ancillary functions
def adjust_plots(xlim_const,ylim_const):
    plt.xlim(xlim_const)
    plt.ylim(ylim_const)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


def sparse_PCA(theta_input, k=None, alpha=1e-4, beta=1e-4, max_iter=1000, tol=1e-5,
                        verbose=True):
    # solves the sparse PCA problem using variable projection
    _, S, Vt = np.linalg.svd(theta_input, full_matrices=False)
    Dmax = S[0]
    A = Vt.T[:, :k]
    B = Vt.T[:, :k]

    VD = Vt.T * S
    VD2 = Vt.T * (S ** 2)

    # Set tuning parameters
    alpha *= Dmax ** 2
    beta *= Dmax ** 2

    nu = 1.0 / (Dmax ** 2 + beta)
    kappa = nu * alpha

    obj = []
    improvement = np.inf

    # Apply Variable Projection Solver
    VD2_Vt = VD2 @ Vt
    for noi in range(1, max_iter + 1):
        # Update A: X'XB = UDV'
        Z = VD2_Vt @ B
        U_Z, _, Vt_Z = np.linalg.svd(Z, full_matrices=False)
        A = U_Z @ Vt_Z
        ######

        ######
        # update B
        #
        grad = (VD2_Vt @ (A - B)) - beta * B  # Gradient step
        B_temp = B + nu * grad  # Gradient step

        # l1 soft_threshold
        B = np.zeros_like(B_temp)
        B[B_temp > kappa] = B_temp[B_temp > kappa] - kappa
        B[B_temp <= -kappa] = B_temp[B_temp <= -kappa] + kappa

        R = VD.T - np.linalg.multi_dot((VD.T, B, A.T))  # residuals
        obj_value = 0.5 * np.sum(R ** 2) + alpha * np.sum(np.abs(B)) + 0.5 * beta * np.sum(B ** 2)

        ######

        obj.append(obj_value)

        # Break if objective is not improving
        if noi > 1:
            improvement = (obj[noi - 2] - obj[noi - 1]) / obj[noi - 1]

        if improvement < tol:
            print("Improvement is smaller than the tolerance, stopping the optimization")
            break


        # Verbose output
        if verbose and (noi % 10 == 0):
            print(f"Iteration: {noi}, Objective: {obj_value:.5e}, Relative improvement: {improvement:.5e}")

    loadings_ = B / np.linalg.norm(B, axis=0)
    return loadings_

# change the data dir to where you downloaded the data
data_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/preprocessing_data/gcPCA_files/Jens_data/scRNA_seq/'

# gene dir should point where you downloaded the gene files
gene_dir = '/home/eliezyer/Documents/github/deOliveira_gcPCA_2024/Figure4/'
save_fig_dir = '/mnt/extraSSD4TB/CloudStorage/Dropbox/figure_manuscripts/figures_gcPCA/Figure4/source_plots/'

# loading genes used for analysis
genes_used = pd.read_csv(os.path.join(gene_dir, 'genes_used.txt'), delimiter=' ', index_col=None, header=None)

# loading data in as pandas dataframe
data = pd.read_table(os.path.join(data_dir, 'GSE153855_Expression_RPKM_HQ_allsamples.txt'), sep='\t',
                     index_col=0, header=None).T

annotation = pd.read_csv(os.path.join(data_dir, 'GSE153855_Cell_annotation.txt'), sep='\t')

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
temp_t2d = np.log2(diabetes_beta_df.values+1)
temp_normal = np.log2(normal_beta_df.values+1)

###
# keeping only genes that were used in previous analysis (Martinez-Lopez et al)
temp_whole_data = np.concatenate((diabetes_beta_df.values, normal_beta_df.values), axis=0)
temp_donors = np.concatenate((donors_diabetes, donors_normal+donors_diabetes.max()+1), axis=0)

genes_to_keep_final = np.isin(all_gene_names, genes_used.values)

T2D_red = temp_t2d[:, genes_to_keep_final]
Normal_red = temp_normal[:, genes_to_keep_final]
genes_used = all_gene_names[genes_to_keep_final]

# centering the data
data_normal = Normal_red - np.mean(Normal_red, axis=0)
data_t2d = T2D_red - np.mean(T2D_red, axis=0)



# fitting gcPCA
lambdas_use = np.exp(np.linspace(np.log(1e-20), np.log(1e-3), num=2))
print('fitting gcPCA T2D')
sparse_gcpca_mdl_t2d = sparse_gcPCA(method='v4', normalize_flag=False, lasso_penalty=lambdas_use, Nsparse=2, ridge_penalty=1e-5, max_steps=1000, tol=1e-5)
sparse_gcpca_mdl_t2d.fit(data_t2d, data_normal)  # data_t2d is diabetes and data_normal is normal, all beta cells


print('fitting gcPCA CTRL')
lambdas_use = np.exp(np.linspace(np.log(1e-20), np.log(1e-3), num=2))
sparse_gcpca_mdl_ctrl = sparse_gcPCA(method='v4', normalize_flag=False, lasso_penalty=lambdas_use, Nsparse=2, ridge_penalty=1e-5, max_steps=1000, tol=1e-5)
sparse_gcpca_mdl_ctrl.fit(data_normal, data_t2d)  # data_t2d is diabetes and data_normal is normal, all beta cells

# running sparse PCA on the t2d and control data
print('fitting sparse PCA T2D')
cond_number = 10**8
cov_data_t2d = (data_t2d.T @ data_t2d) / (data_t2d.shape[0]-1)
cov_data_ctrl = (data_normal.T @ data_normal) / (data_normal.shape[0]-1)
# Getting square root matrix
w_t2d, v_t2d = np.linalg.eigh(cov_data_t2d)
alpha_pos = w_t2d.max() / cond_number  # fixing it to be positive definite
theta_t2d = v_t2d * np.sqrt(w_t2d+alpha_pos) @ v_t2d.T
top_t2d_pcs = 2
loadings_t2d = sparse_PCA(theta_t2d, k=top_t2d_pcs, alpha=2e-3, beta=1e-5, max_iter=1000, tol=1e-5, verbose=True)
v_t2d_sorted = v_t2d[:, np.argsort(w_t2d)[::-1]]

print('fitting sparse PCA CTRL')
w_ctrl,v_ctrl = np.linalg.eigh(cov_data_ctrl)
alpha_pos = w_ctrl.max() / cond_number  # fixing it to be positive definite
theta_ctrl = v_ctrl * np.sqrt(w_ctrl+alpha_pos) @ v_ctrl.T
top_ctrl_pcs = 2
loadings_ctrl = sparse_PCA(theta_ctrl, k=top_ctrl_pcs, alpha=2e-3, beta=1e-5, max_iter=1000, tol=1e-5, verbose=True)
new_v_ctrl = v_ctrl[:, np.argsort(w_ctrl)[::-1]]

# plot of the clusters
plt.rcParams.update({'figure.dpi':150, 'font.size':12})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
marker_size = 10

###
# sparse PCA plots
#
control_lims = [-70, 130, -58, 58]  # [xmin, xmax, ymin, ymax]
t2d_lims = [-60,131, -39, 73]  # [xmin, xmax, ymin, ymax]
sparse_pca_t2d_df = pd.DataFrame({"t2d_scores_dim1": data_t2d@loadings_t2d[:,0],
                                    "t2d_scores_dim2": data_t2d@loadings_t2d[:,1],
                                    "ctrl_scores_dim1": data_t2d@loadings_ctrl[:,0],
                                    "ctrl_scores_dim2": data_t2d@loadings_ctrl[:,1],
                                    "donors": donors_diabetes})

sparse_pca_ctrl_df = pd.DataFrame({"ctrl_scores_dim1": data_normal@loadings_ctrl[:,0],
                                   "ctrl_scores_dim2": data_normal@loadings_ctrl[:,1],
                                   "t2d_scores_dim1": data_normal@loadings_t2d[:,0],
                                   "t2d_scores_dim2": data_normal@loadings_t2d[:,1],
                                   "donors": donors_normal})


plt.rcParams.update({'figure.dpi':150, 'font.size':16})
fig, (axs) = plt.subplots(2, 2, figsize=(6, 7))
sns.set(font_scale=1.5)
sns.set_style('white')
plt.subplot(221)
sns.scatterplot(data=sparse_pca_t2d_df, x="t2d_scores_dim1", y="t2d_scores_dim2", hue="donors", palette='Set1',legend=False)
plt.xlabel('T2D sparse PC$_1$')
plt.ylabel('T2D sparse PC$_2$')
adjust_plots(t2d_lims[:2], t2d_lims[2:])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.subplot(222)
sns.scatterplot(data=sparse_pca_t2d_df, x="ctrl_scores_dim1", y="ctrl_scores_dim2", hue="donors", palette='Set1',legend=False)
plt.xlabel('Control sparse PC$_1$')
plt.ylabel('Control sparse PC$_2$')
adjust_plots(t2d_lims[:2], t2d_lims[2:])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.subplot(223)
sns.scatterplot(data=sparse_pca_ctrl_df, x="t2d_scores_dim1", y="t2d_scores_dim2", hue="donors", palette='Set2',legend=False)
plt.xlabel('T2D sparse PC$_1$')
plt.ylabel('T2D sparse PC$_2$')
adjust_plots(control_lims[:2], control_lims[2:])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.subplot(224)
sns.scatterplot(data=sparse_pca_ctrl_df, x="ctrl_scores_dim1", y="ctrl_scores_dim2", hue="donors", palette='Set2',legend=False)
plt.xlabel('Control sparse PC$_1$')
plt.ylabel('Control sparse PC$_2$')
adjust_plots(control_lims[:2], control_lims[2:])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
fig.subplots_adjust(wspace=0.5, hspace=0.7, left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.savefig(os.path.join(save_fig_dir, 'sparse_PCA_scores_t2d_control.png'))
plt.savefig(os.path.join(save_fig_dir, 'sparse_PCA_scores_t2d_control.pdf'))


# plot of correlation of first two PCs
plt.rcParams.update({'figure.dpi':150, 'font.size':16})

# eigenvalue of the dimensions
eig_sparse_t2d = np.diag(loadings_t2d.T@cov_data_t2d@loadings_t2d)
eig_sparse_ctrl = np.diag(loadings_ctrl.T@cov_data_ctrl@loadings_ctrl)

# applying the eigenvalue too the vector
x = eig_sparse_t2d[:2]*loadings_t2d[:,:2]
y = eig_sparse_ctrl[:2]*loadings_ctrl[:,:2]

# getting the l2-norm
sparse_pca_t2d_mag = np.sqrt(np.sum(x**2, axis=1))
sparse_pca_ctrl_mag = np.sqrt(np.sum(y**2, axis=1))

data_df = pd.DataFrame({"t2dPC1": sparse_pca_t2d_mag, "ctrlPC1": sparse_pca_ctrl_mag, "dummy": np.ones(len(sparse_pca_t2d_mag))})
sns.set(font_scale=1.2)
sns.set_style('white')
sns.jointplot(data_df, x="t2dPC1", y="ctrlPC1", hue="dummy",legend=False,height=3,alpha=0.5,ratio=3)
plt.xlabel('T2D PC$_{(1,2)}$')
plt.ylabel('Control PC$_{(1,2)}$')
plt.text(0.5, 0.9, 'r: {:.2f}'.format(np.corrcoef(sparse_pca_t2d_mag, sparse_pca_ctrl_mag)[0,1]), ha='center', va='center', transform=plt.gca().transAxes)
plt.xlim([-2, sparse_pca_t2d_mag.max()+10])
plt.ylim([-4, sparse_pca_ctrl_mag.max()+10])
plt.xticks([]),plt.yticks([])
plt.savefig(os.path.join(save_fig_dir, 'sparse_PCA_loadings_corr.png'))
plt.savefig(os.path.join(save_fig_dir, 'sparse_PCA_loadings_corr.pdf'))

# plot LDA comparison
# fitting LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_mdl = LinearDiscriminantAnalysis()
lda_labels_ = np.concatenate((np.zeros(len(donors_diabetes)), np.ones(len(donors_normal))), axis=0).flatten()
lda_labels_str_ = ['T2D']*len(donors_diabetes) + ['Control']*len(donors_normal)
lda_mdl.fit(np.concatenate((data_t2d, data_normal), axis=0), lda_labels_)
lda_scores = np.concatenate((data_t2d, data_normal),axis=0).dot(lda_mdl.scalings_)

plt.rcParams.update({'figure.dpi':150, 'font.size':16})
plt.figure(figsize=(3,3))
lda_df = pd.DataFrame({"lda_scores": lda_scores.flatten(), "condition": lda_labels_str_})
colors = ["#ED1C24", "#939598"]
customPalette = sns.set_palette(sns.color_palette(colors))
ax = sns.histplot(lda_df, x="lda_scores", hue="condition", kde=True, bins=20, stat="probability", common_norm=False, palette=customPalette, legend=True)
plt.xlabel('LDA scores')
plt.title('LDA (T2D,Control)')
plt.yticks([])
ax.set_position([0.25, 0.2, 0.7, 0.7])
plt.savefig(os.path.join(save_fig_dir, 'LDA_scores_t2d_vs_control.png'))
plt.savefig(os.path.join(save_fig_dir, 'LDA_scores_t2d_vs_control.pdf'))


# make plot of truncated  loadings sparse PCA
# T2D
eig_sparse_t2d = np.diag(loadings_t2d.T@cov_data_t2d@loadings_t2d) # revise this
temp_load_ = eig_sparse_t2d*loadings_t2d
sparse_t2d_pca_loadings_mag = np.sqrt(np.sum(temp_load_**2, axis=1))
temp_idx = np.argsort(sparse_t2d_pca_loadings_mag)
idx_extremeties = temp_idx[-20:]


plt.rcParams.update({'figure.dpi':150, 'font.size':20})
plt.figure(figsize=(4, 6.5))
plt.stem(sparse_t2d_pca_loadings_mag[idx_extremeties],orientation='horizontal')
plt.yticks(np.arange(len(idx_extremeties)), genes_used[idx_extremeties])
plt.xlabel('Loadings \n magnitude',fontsize=18)
plt.xlim([-1, sparse_t2d_pca_loadings_mag.max()+5])
plt.xticks([])
plt.title('T2D PC$_{(1,2)}$')
plt.tight_layout()
plt.savefig(os.path.join(save_fig_dir, 'sparse_T2D_PCA_loadings_truncated.png'))
plt.savefig(os.path.join(save_fig_dir, 'sparse_T2D_PCA_loadings_truncated.pdf'))


###
# sparse gcPCA plots
#

xlim_const = [-0.16, 0.16]
ylim_const = [-0.20, 0.362]
a=1
sparse_gcpca_t2d_df = pd.DataFrame({"scores dim1": sparse_gcpca_mdl_t2d.Ra_scores_[a][:,0],
                                "scores dim2": sparse_gcpca_mdl_t2d.Ra_scores_[a][:,1],
                                "scores dim last": sparse_gcpca_mdl_ctrl.Rb_scores_[a][:,0],
                                "scores dim last-1": sparse_gcpca_mdl_ctrl.Rb_scores_[a][:,1],
                                "donors": donors_diabetes})

sparse_gcpca_ctrl_df = pd.DataFrame({"scores dim1": sparse_gcpca_mdl_t2d.Rb_scores_[a][:,0],
                                    "scores dim2": sparse_gcpca_mdl_t2d.Rb_scores_[a][:,1],
                                    "scores dim last": sparse_gcpca_mdl_ctrl.Ra_scores_[a][:,0],
                                    "scores dim last-1": sparse_gcpca_mdl_ctrl.Ra_scores_[a][:,1],
                                    "donors": donors_normal})


plt.rcParams.update({'figure.dpi':150, 'font.size':16})
fig, (axs) = plt.subplots(2, 2, figsize=(6, 7))
sns.set(font_scale=1.7)
sns.set_style('white')
plt.subplot(221)
sns.scatterplot(data=sparse_gcpca_t2d_df, x="scores dim1", y="scores dim2", hue="donors", palette='Set1',legend=False)
plt.xlabel('sparse gcPC$_1$')
plt.ylabel('sparse gcPC$_2$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
adjust_plots(xlim_const,ylim_const)

plt.subplot(222)
sns.scatterplot(data=sparse_gcpca_t2d_df, x="scores dim last", y="scores dim last-1", hue="donors", palette='Set1',legend=False)
plt.xlabel('sparse gcPC$_{last}$')
plt.ylabel('sparse gcPC$_{last-1}$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
adjust_plots(xlim_const,ylim_const)

plt.subplot(223)
sns.scatterplot(data=sparse_gcpca_ctrl_df, x="scores dim1", y="scores dim2", hue="donors", palette='Set2',legend=False)
plt.xlabel('sparse gcPC$_1$')
plt.ylabel('sparse gcPC$_2$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
adjust_plots(xlim_const,ylim_const)

plt.subplot(224)
sns.scatterplot(data=sparse_gcpca_ctrl_df, x="scores dim last", y="scores dim last-1", hue="donors", palette='Set2',legend=False)
plt.xlabel('sparse gcPC$_{last}$')
plt.ylabel('sparse gcPC$_{last-1}$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
adjust_plots(xlim_const, ylim_const)
fig.subplots_adjust(wspace=0.5, hspace=0.7, left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_scores_t2d_control.png'))
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_scores_t2d_control.pdf'))


# plot of correlation of first two gcPCs with the latest two gcPCs
plt.rcParams.update({'figure.dpi':150, 'font.size':12})
x = sparse_gcpca_mdl_t2d.sparse_loadings_[-1][:, :2] * sparse_gcpca_mdl_t2d.Ra_values_[-1][:2]
y = sparse_gcpca_mdl_ctrl.sparse_loadings_[-1][:, :2] * sparse_gcpca_mdl_ctrl.Ra_values_[-1][:2]
sparse_gcpca_t2d_mag = np.sqrt(np.sum(x**2, axis=1))
sparse_gcpca_ctrl_mag = np.sqrt(np.sum(y**2, axis=1))
data_df = pd.DataFrame({"gcPC1": sparse_gcpca_t2d_mag.flatten(), "gcPC2": sparse_gcpca_ctrl_mag.flatten(), "dummy": np.ones(len(sparse_gcpca_t2d_mag))})
sns.set(font_scale=1.2)
sns.set_style('white')
ax=sns.jointplot(data_df, x="gcPC1", y="gcPC2",hue="dummy",legend=False,height=3,alpha=0.5,ratio=3)
plt.xlabel('(T2D,Control)\ngcPC$_{(1,2)}$')
plt.ylabel('(T2D,Control)\ngcPC$_{(last,last-1)}$')
plt.text(0.5, 0.9, 'r: {:.2f}'.format(np.corrcoef(sparse_gcpca_t2d_mag, sparse_gcpca_ctrl_mag)[0,1]), ha='center', va='center', transform=plt.gca().transAxes)
plt.xlim([-1.5, sparse_gcpca_t2d_mag.max()+10])
plt.ylim([-1.8, sparse_gcpca_ctrl_mag.max()+10])
plt.xticks([]),plt.yticks([])
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_loadings_corr.png'))
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_loadings_corr.pdf'))


# plot of correlations of first two gcPCAs T2D with control PC
plt.rcParams.update({'figure.dpi':150, 'font.size':18})
eig_sparse_t2d = np.diag(loadings_t2d.T@cov_data_t2d@loadings_t2d) # revise this

x = sparse_gcpca_mdl_t2d.sparse_loadings_[-1][:, :2] * sparse_gcpca_mdl_t2d.Ra_values_[-1][:2]
y = eig_sparse_t2d[:2]*loadings_t2d[:, :2]

sparse_gcpca_t2d_mag = np.sqrt(np.sum(x**2, axis=1))
sparse_pca_t2d_mag = np.sqrt(np.sum(y**2, axis=1))
data_df = pd.DataFrame({"gcPC1": sparse_gcpca_t2d_mag.flatten(), "PC1": sparse_pca_t2d_mag.flatten(), "dummy": np.ones(len(sparse_gcpca_t2d_mag))})
sns.set(font_scale=1.2)
sns.set_style('white')
sns.jointplot(data_df, x="gcPC1", y="PC1", hue="dummy",legend=False,height=3,alpha=0.5,ratio=3)
plt.xlabel('(T2D,Control)\ngcPC$_{(1,2)}$')
plt.ylabel('T2D PC$_{(1,2)}$')
plt.text(0.5, 0.9, 'r: {:.2f}'.format(np.corrcoef(sparse_gcpca_t2d_mag, sparse_pca_t2d_mag)[0,1]), ha='center', va='center', transform=plt.gca().transAxes)
plt.xlim([-1.5, sparse_gcpca_t2d_mag.max()+10])
plt.ylim([-2, sparse_pca_t2d_mag.max()+10])
plt.xticks([]),plt.yticks([])
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_PCA_loadings_corr.png'))
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_PCA_loadings_corr.pdf'))


# make a plot of truncated gcPCs, showing the top 40 loadings
temp_load_ = sparse_gcpca_mdl_t2d.sparse_loadings_[-1][:,:10]*sparse_gcpca_mdl_t2d.Ra_values_[-1][:10]
sparse_gcpca_t2d_loadings_mag = np.sqrt(np.sum(temp_load_**2, axis=1))
temp_idx = np.argsort(sparse_gcpca_t2d_loadings_mag)
idx_extremeties = temp_idx[-20:]


plt.rcParams.update({'figure.dpi':150, 'font.size':20})
plt.figure(figsize=(4, 6.5))
plt.stem(sparse_gcpca_t2d_loadings_mag[idx_extremeties],orientation='horizontal')
# add gene names to xtick labels
plt.yticks(np.arange(len(idx_extremeties)), genes_used[idx_extremeties])
plt.xlabel('Loadings \n magnitude',fontsize=18)
plt.xlim([-1, sparse_gcpca_t2d_loadings_mag.max()+5])
plt.xticks([])
plt.title('(T2D,Control)\ngcPC$_{(1,2)}$')
plt.tight_layout()
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_T2D_loadings_truncated.png'))
plt.savefig(os.path.join(save_fig_dir, 'sparse_gcPCA_T2D_loadings_truncated.pdf'))