import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.cluster import KMeans
from config import *


def gauss_mix_model_selection(X, k_ub=7, cv_types=('spherical', 'tied', 'diag', 'full')):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, k_ub)
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type,
                                          warm_start=True)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)


def elbow_method(X, k_lb, k_ub):
    wgs = []
    pes = []
    for k in range(k_lb, k_ub):
        clstr = KMeans(n_clusters=k, random_state=0)    # the orginal labeled data has 2 classes, so it is reasaonable to set k=2 XXXX
        clstr.fit(X)
        lb = clstr.labels_

        _, cnts = np.unique(lb, return_counts=True)
        between_group_ssd = np.sum(np.sum((clstr.cluster_centers_ - clstr.cluster_centers_.mean(axis=0))**2, axis=1) * cnts)  # weighted by the number of points in each cluster
        within_group_ssd = clstr.inertia_
        pct_explained_ssd = between_group_ssd / (between_group_ssd + within_group_ssd)
        wgs.append(within_group_ssd)
        pes.append(pct_explained_ssd)
    return pes, wgs
