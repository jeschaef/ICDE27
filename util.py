import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_d
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import DoubleTensor, Tensor
from torch.nn.functional import cosine_similarity

from GraphXAI.graphxai.datasets import ShapeGGen
from GraphXAI.graphxai.explainers import *
from GraphXAI.graphxai.utils import Explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_classification_threshold(y_true, y_pred):
    max_t = None
    max_acc = 0
    for t in range(100):
        p = (y_pred >= t * 0.01).astype(int)
        acc = accuracy_score(y_true, p)
        if acc > max_acc:
            max_acc = acc
            max_t = t * 0.01
    return (y_pred >= max_t).astype(int), max_t


def match_columns(columns, orig_columns):
    '''
    Match the ohe features with the original column names (not ohe).
    '''
    matching = {}
    for i, c_new in enumerate(columns):
        for j, c_old in enumerate(orig_columns):
            if c_new.startswith(c_old):  # match new and old columns (same prefix)
                if j in matching:
                    matching[j].append(i)
                else:
                    matching[j] = [i]
    return matching


def reorder_columns(df, orig_columns):
    '''
    Reorder the columns of the ohe dataframe according to the previous order without ohe.
    '''
    matching = match_columns(df.columns, orig_columns)
    column_idx = [it for t in sorted(matching) for it in
                  matching[t]]  # flatten the new column indices (sorted by old column)
    reordered = df.columns[column_idx]
    return df[reordered]  # reorder the dataframe


def match_ohe_columns(columns, orig_columns):
    '''
    Get a matching of original columns to the one-hot encoded columns
    '''
    match = match_columns(columns, orig_columns)
    return {k: v for k, v in match.items() if len(v) > 1}  # return only ohe column matches (>1 value)


def get_ohe_columns(columns, orig_columns):
    '''
    Get the indices of all ohe features.
    '''
    match_ohe = match_ohe_columns(columns, orig_columns)
    return [v for vals in match_ohe.values() for v in vals]


def reorderer_pipeline(X):
    # Transform the dataset:
    # 1. One-hot encode categorical features
    ohe = OneHotEncoder(drop='if_binary',
                        handle_unknown='infrequent_if_exist',
                        min_frequency=0.01,
                        sparse_output=False)

    # 2. Normalize numeric features
    tf = make_column_transformer(
        (ohe, make_column_selector(dtype_exclude="number")),
        (StandardScaler(), make_column_selector(dtype_include="number")),
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    # Assemble transformation with reordering columns (to get initial layout with one-hot encoded features)
    return make_pipeline(tf, Reorderer(X.columns))


class Reorderer(BaseEstimator, TransformerMixin):
    '''
    Reorder the ohe dataframe according to the original columns.
    '''

    def __init__(self, orig_columns):
        self.orig_columns = orig_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return reorder_columns(X, self.orig_columns)

def mse(heatmap1, heatmap2, **kwargs):
    return ((heatmap1 - heatmap2) ** 2).mean(**kwargs)


def cosine(heatmap1, heatmap2, **kwargs):
    # Flatten the heatmaps
    heatmap1_flat = heatmap1.flatten()
    heatmap2_flat = heatmap2.flatten()
    return cosine_d(heatmap1_flat, heatmap2_flat, **kwargs)


def ssim(heatmap1, heatmap2, flatten=False, **kwargs):
    if flatten:
        heatmap1 = heatmap1.flatten()
        heatmap2 = heatmap2.flatten()
    return 1 - structural_similarity(heatmap1, heatmap2, **kwargs)


def earth_movers_distance(heatmap1, heatmap2):
    # Flatten the heatmaps
    heatmap1_flat = heatmap1.flatten()
    heatmap2_flat = heatmap2.flatten()
    # Normalize the heatmaps
    heatmap1_flat = heatmap1_flat / heatmap1_flat.sum()
    heatmap2_flat = heatmap2_flat / heatmap2_flat.sum()
    return wasserstein_distance(heatmap1_flat, heatmap2_flat)


def intersection(t1: Tensor, t2: Tensor):
    if t2.shape[0] > t1.shape[0]:
        t1, t2 = t2, t1
    idx = torch.searchsorted(t1[:-1], t2)
    return t2[t1[idx] == t2]


def union(t1: Tensor, t2: Tensor):
    return torch.cat((t1, t2)).unique()


def jacc(t1: Tensor, t2: Tensor):
    I = intersection(t1, t2)
    U = union(t1, t2)
    return 1 - len(I) / len(U)


def feat_imp_dist(exp1: Explanation, exp2: Explanation):
    exp1_feat = exp1.feature_imp.reshape(1, -1)
    exp2_feat = exp2.feature_imp.reshape(1, -1)
    return 1 - cosine_similarity(exp1_feat, exp2_feat)


def node_imp_dist(exp1: Explanation, exp2: Explanation, threshold: float = 0.0, separated: bool = False):
    nodes1 = exp1.enc_subgraph.nodes
    nodes2 = exp2.enc_subgraph.nodes
    node_imp1 = exp1.node_imp.type(DoubleTensor)
    node_imp1 = (node_imp1 - node_imp1.mean()) / node_imp1.std()
    node_imp2 = exp2.node_imp.type(DoubleTensor)
    node_imp2 = (node_imp2 - node_imp2.mean()) / node_imp2.std()
    I_above = intersection(nodes1[node_imp1 > threshold], nodes2[node_imp2 > threshold])
    I_below = intersection(nodes1[node_imp1 <= threshold], nodes2[node_imp2 <= threshold])
    U = union(nodes1, nodes2)  # or do IoU for above and IoU for below -> 50/50
    U_above = union(nodes1[node_imp1 > threshold], nodes2[node_imp2 > threshold])
    U_below = union(nodes1[node_imp1 <= threshold], nodes2[node_imp2 <= threshold])
    if separated:
        return 0.5 * (1 - len(I_above) / len(U_above)) + 0.5 * (1 - len(I_below) / len(U_below))
    # return 1 - (len(I_above) + len(I_below)) / len(U)
    if len(U_above) == 0:
        return 1
    return 1 - len(I_above) / len(U_above)


def edge_imp_dist(exp1: Explanation, exp2: Explanation, threshold: float = 0.0, separated: bool = False):
    edges1 = torch.where(exp1.enc_subgraph.edge_mask)[0]
    edges2 = torch.where(exp2.enc_subgraph.edge_mask)[0]
    edge_imp1 = exp1.edge_imp.type(DoubleTensor)
    edge_imp1 = (edge_imp1 - edge_imp1.mean()) / edge_imp1.std()
    edge_imp2 = exp2.edge_imp.type(DoubleTensor)
    edge_imp2 = (edge_imp2 - edge_imp2.mean()) / edge_imp2.std()
    I_above = intersection(edges1[edge_imp1 > threshold], edges2[edge_imp2 > threshold])
    I_below = intersection(edges1[edge_imp1 <= threshold], edges2[edge_imp2 <= threshold])
    U = union(edges1, edges2)
    U_above = union(edges1[edge_imp1 > threshold], edges2[edge_imp2 > threshold])
    U_below = union(edges1[edge_imp1 <= threshold], edges2[edge_imp2 <= threshold])
    if separated:
        return 0.5 * (1 - len(I_above) / len(U_above)) + 0.5 * (1 - len(I_below) / len(U_below))
    # return 1 - (len(I_above) + len(I_below)) / len(U)
    if len(U_above) == 0:
        return 1
    return 1 - len(I_above) / len(U_above)


def expl_dist(exp1: Explanation,
              exp2: Explanation,
              weights: np.array = None,
              node_threshold: float = 0.0,
              node_separated: bool = False,
              edge_threshold: float = 0.0,
              edge_separated: bool = False):
    if weights is None:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
    assert weights.shape == (3,)
    assert (weights >= 0).all() == True
    f = 0
    n = 0
    e = 0
    if weights[0] > 0:
        f = feat_imp_dist(exp1, exp2)
    if weights[1] > 0:
        n = node_imp_dist(exp1, exp2, threshold=node_threshold, separated=node_separated)
    if weights[2] > 0:
        e = edge_imp_dist(exp1, exp2, threshold=edge_threshold, separated=edge_separated)
    return weights[0] * f + weights[1] * n + weights[2] * e


def graph_explain_single(dataset: ShapeGGen,
                         model: torch.nn.Module,
                         criterion: torch.nn.Module,
                         X: torch.Tensor,
                         Y: torch.Tensor,
                         EIDX: torch.Tensor,
                         exp_method: str,
                         pred: torch.Tensor,
                         node_idx: int,
                         pgex: PGExplainer = None):

    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0)


    # Get explainer
    if exp_method == 'pgex':
        explainer = pgex
        forward_kwargs = {'node_idx': node_idx,
                          'x': X.to(device),
                          'edge_index': EIDX.to(device),
                          'label': pred_class}
    else:
        explainer, forward_kwargs = get_exp_method(exp_method, model, criterion, dataset, X, Y, EIDX, node_idx,
                                                   pred_class)
    return explainer.get_explanation_node(**forward_kwargs)


def graph_explain_all(dataset: ShapeGGen,
                      model: torch.nn.Module,
                      criterion: torch.nn.Module,
                      X: torch.Tensor,
                      Y: torch.Tensor,
                      EIDX: torch.Tensor,
                      exp_method: str,
                      pred: torch.Tensor,
                      node_indices: torch.Tensor = None,
                      pgex: PGExplainer = None):
    explanations = []
    if node_indices is None:
        node_indices = torch.arange(X.shape[0])
    for node_idx in node_indices:
        node_idx = node_idx.item()
        explanations.append(
            graph_explain_single(dataset, model, criterion, X, Y, EIDX, exp_method, pred, node_idx, pgex=pgex))
    return explanations


def get_exp_method(method, model, criterion, dataset, X, Y, EIDX, node_idx, pred_class):
    if method=='gnnex':
        exp_method = GNNExplainer(model)
        forward_kwargs={'x': X.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': EIDX.to(device)}
    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': X.to(device),
                        'y': Y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': EIDX.to(device)}
    elif method == 'cam':
        act = lambda x: torch.argmax(x, dim=1)
        exp_method = CAM(model, activation=act)
        forward_kwargs={'x': X.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class,
                        'edge_index': EIDX.to(device)}
    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':X.to(device),
                        'y': Y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': EIDX.to(device),
                        'average_variant': [True]}
    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': X.to(device),
                        'y': Y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': EIDX.to(device)}
    elif method=='glime':
        exp_method = GraphLIME(model)
        forward_kwargs={'x': X.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': EIDX.to(device)}
    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': X.to(device),
                        'edge_index': EIDX.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class}
    elif method=='glrp':
        exp_method = GNN_LRP(model)
        forward_kwargs={'x': X.to(device),
                        'edge_index': EIDX.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class,
                        'edge_aggregator':torch.sum}
    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=False, p_threshold=0.1)
        forward_kwargs={'node_idx': int(node_idx),
                        'x': X.to(device),
                        'edge_index': EIDX.to(device),
                        'top_k_nodes': 10}
    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': X.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': EIDX.to(device)}
    elif method=='subx':
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = dataset.model_layers)
        forward_kwargs={'node_idx': node_idx,
                        'x': X.to(device),
                        'edge_index': EIDX.to(device),
                        'label': pred_class,
                        'max_nodes': 15}
    else:
        OSError('Invalid argument!!')
    return exp_method, forward_kwargs