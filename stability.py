import pandas as pd
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure
import matplotlib.pyplot as plt
from torch import cdist
from torch.nn.functional import pairwise_distance

from robustness import agg_exp_metric



def stability_v2(X, explanations, algorithm=KMeans, min_k=2, max_k=10, **params):
    models = {}
    ticks = list(range(min_k, max_k + 1))
    #silhouettes = pd.DataFrame(columns=['SilA', 'SilB', 'Sil'])
    for k in ticks:
        m = algorithm(n_clusters=k, **params).fit(X)
        A = m.labels_
        ms = algorithm(n_clusters=k, **params).fit(explanations)
        B = ms.labels_
        models[k] = [A, B]
        # sil_a = silhouette_score(X, A)
        # sil_b = silhouette_score(shap_values, B)
        # sil = 0.5 * (sil_a + sil_b)
        # silhouettes.loc[k] = [ sil_a, sil_b, sil ]

    return compute_cvi(models, min_k, max_k)


def stability(X, explanations, algorithm=KMeans, min_k=2, max_k=10, seed=42, **kwargs):
    models = {}
    ticks = list(range(min_k, max_k + 1))
    #silhouettes = pd.DataFrame(columns=['SilA', 'SilB', 'Sil'])
    data_cluster_kwargs = {}
    explanation_cluster_kwargs = {}
    if "data_cluster_kwargs" in kwargs:
        data_cluster_kwargs = kwargs["data_cluster_kwargs"]
    if "explanation_cluster_kwargs" in kwargs:
        explanation_cluster_kwargs = kwargs["explanation_cluster_kwargs"]
    for k in ticks:
        m = algorithm(n_clusters=k, random_state=seed, **data_cluster_kwargs).fit(X)
        A = m.labels_
        ms = algorithm(n_clusters=k, random_state=seed, **explanation_cluster_kwargs).fit(explanations)
        B = ms.labels_
        models[k] = [A, B]
        # sil_a = silhouette_score(X, A)
        # sil_b = silhouette_score(shap_values, B)
        # sil = 0.5 * (sil_a + sil_b)
        # silhouettes.loc[k] = [ sil_a, sil_b, sil ]

    return compute_cvi(models, min_k, max_k)


def stability_surrogate_euclidean(X, explain_all, maxv_X=1.0, maxv=1.0):
    explanations = explain_all(X)
    distX = cdist(X, X, p=2) / maxv_X
    distE = cdist(explanations, explanations, p=2) / maxv
    return torch.square(distE - distX).mean()


def compute_cvi(models, min_k, max_k):
    ticks = list(range(min_k, max_k + 1))
    N = max_k - min_k + 1
    scores = np.zeros((3, N, N))
    for k1 in ticks:
        A = models[k1][0]
        for k2 in ticks:
            B = models[k2][1]
            scores[0, k1 - min_k, k2 - min_k] = adjusted_rand_score(A, B)
            scores[1, k1 - min_k, k2 - min_k] = adjusted_mutual_info_score(A, B)
            _, _, scores[2, k1 - min_k, k2 - min_k] = homogeneity_completeness_v_measure(A, B)

    return scores



def stability_labels(labels, explanations, algorithm=KMeans, min_k=2, max_k=10, **params):
    models = {}
    ticks = list(range(min_k, max_k + 1))
    for k in ticks:
        ms = algorithm(n_clusters=k, **params).fit(explanations)
        B = ms.labels_
        models[k] = B
        # sil_a = silhouette_score(X, A)
        # sil_b = silhouette_score(shap_values, B)
        # sil = 0.5 * (sil_a + sil_b)
        # silhouettes.loc[k] = [ sil_a, sil_b, sil ]

    N = max_k - min_k + 1
    scores = np.zeros((3, N))
    for k in ticks:
        B = models[k]
        scores[0, k - min_k] = adjusted_rand_score(labels, B)
        scores[1, k - min_k] = adjusted_mutual_info_score(labels, B)
        _, _, scores[2, k - min_k] = homogeneity_completeness_v_measure(labels, B)

    return scores




def plot_score_lines(X, explanations, algorithm=KMeans, min_k=2, max_k=10, ax=None, file=None, **params):
    scores = pd.DataFrame(columns=['ARI', 'AMI', 'V', 'SilA', 'SilB', 'Sil', 'Mix'])
    for k in range(min_k, max_k + 1):
        m = algorithm(n_clusters=k).fit(X)
        A = m.labels_
        ms = algorithm(n_clusters=k, **params).fit(explanations)
        B = ms.labels_

        ari = adjusted_rand_score(A, B)
        ami = adjusted_mutual_info_score(A, B)
        homo, compl, vmeas = homogeneity_completeness_v_measure(A,B)
        sila = silhouette_score(X, A)
        silb = silhouette_score(shap_values, B)
        sil = (sila + silb) / 2
        scores.loc[k] = {
            "ARI": ari,
            "AMI": ami,
            "V": vmeas,
            "SilA": sila,
            "SilB": silb,
            "Sil": sil,
            "Mix": 0.5 * (sil + ari),
        }

    best_k = scores['Mix'].idxmax()
    if ax is None:
        plt.axvline(x=best_k, color='b')
    else:
        ax.axvline(x=best_k, color='b')
    sns.lineplot(scores, ax=ax)
    if file:
        plt.savefig(file, bbox_inches='tight')
    
    
def plot_score_maps(X, explanations, algorithm=KMeans, min_k=2, max_k=10, file=None, **params):
    scores = stability(X, explanations, algorithm=algorithm, min_k=min_k, max_k=max_k, **params)
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    sns.heatmap(scores[0], xticklabels=ticks, yticklabels=ticks, ax=ax[0])
    sns.heatmap(scores[1], xticklabels=ticks, yticklabels=ticks, ax=ax[1])
    sns.heatmap(scores[2], xticklabels=ticks, yticklabels=ticks, ax=ax[2])
    # sns.lineplot(silhouettes, ax=ax[3])
    
    for i in range(len(ax)):
        ax[i].invert_yaxis()
        # k1, k2 = np.unravel_index(scores[i].argmax(), scores[i].shape)
        # ax[i].set_title(f"Max={np.round(scores[i].max(), decimals=2)}, ky={k1 + min_k}, kx={k2 + min_k}, Avg={np.round(scores[i].mean(), decimals=2)}")
    ax[0].set_title(f"ARI max={np.round(scores[0].max(), decimals=2)}")
    ax[1].set_title(f"AMI max={np.round(scores[1].max(), decimals=2)}")
    ax[2].set_title(f"V max={np.round(scores[2].max(), decimals=2)}")
                                    
        
    if file:
        fig.savefig(file, bbox_inches='tight')
        
        
def lime_exp_to_array(e):
    elist = e.as_map()[1]
    
    arr = np.zeros(len(elist))
    for i, val in elist:
        arr[i] = val
    return arr


def ale_to_expl(x, e):
    '''
    Transform ALE to a local explanation for an instance.
    '''
    ales = []
    for i, fi in enumerate(e.feature_names):
        maxi = np.argmax(e.feature_values[i] > x[i])
        if x[i] - e.feature_values[i][maxi-1] > e.feature_values[i][maxi-1] - x[i]:
            ales.append(e.ale_values[i][maxi-1][0])
        else:
            ales.append(e.ale_values[i][maxi][0])
    return np.array(ales)

def anchor_explain_all(X, E, **params):
    '''
    Explain all instances using Anchor tabular explainer.
    '''
    return [E.explain(x, **params) for x in X]


def transform_anchor(a):
    a_trans = []
    for x in a:
        if ' = ' in x:
            x = x.replace(' = ', ' == ')
            val = x.split(' == ')[1]
            x = x.replace(val, f"'{val}'")
        a_trans.append(x)
    return a_trans


def eval_anchor(a, X):
    '''
    Evaluate an anchor on a dataset.
    '''
    a = transform_anchor(a)
    expr = ' and '.join(a)
    return X.query(expr)


def jaccard_similarity(list1, list2):
    '''
    Jaccard similarity of two lists.
    '''
    s1 = set(list1)
    s2 = set(list2)
    inters = len(s1.intersection(s2))
    if inters == 0:
        return 0
    return float(inters / (len(s1) + len(s2) - inters))


def sorensen_dice(list1, list2):
    '''
    Sorensen-Dice index of two lists.
    '''
    s1 = set(list1)
    s2 = set(list2)
    inters = len(s1.intersection(s2))
    if inters == 0:
        return 0
    return 2 * inters / (len(s1) + len(s2))


def anchor_distance(a, b, X, f=jaccard_similarity):
    '''
    Distance between two anchors.
    '''
    A = eval_anchor(a.data['anchor'], X)
    B = eval_anchor(b.data['anchor'], X)
    return 1 - f(A.index, B.index)


