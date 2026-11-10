import random as rand

import numpy as np
import torch
import tqdm
from sklearn.cluster import AgglomerativeClustering
from torch import cdist
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pairwise_distance
from torch.optim import Adam

from GraphXAI.graphxai.datasets.shape_graph import ShapeGGen
from GraphXAI.graphxai.explainers import *
from GraphXAI.graphxai.gnn_models import node_classification
from GraphXAI.graphxai.gnn_models.node_classification.testing import GIN_3layer_basic
from GraphXAI.graphxai.metrics.base import calculate_delta
from robustness import robustness_graph
from stability import stability
from util import expl_dist, graph_explain_all

seed_value = 912
rand.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

print("CUDA?", torch.cuda.is_available(), torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

dataset = ShapeGGen(
    model_layers = 2,
    num_subgraphs = 15,
    subgraph_size = 13,
    prob_connection = 0.3,
    add_sensitive_feature = False
)


data = dataset.get_graph(use_fixed_split=True).to(device)
criterion = CrossEntropyLoss()

def pretrain():
    model = GIN_3layer_basic(32, input_feat = dataset.n_features, classes = 2).to(device)
    optimizer = Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)

    # Train & test model
    for i in range(200):
        loss = node_classification.train(model, optimizer, criterion, data)
    f1, acc, prec, rec, auprc, auroc = node_classification.test(model, data, num_classes = 2, get_auc = True)
    print('Test F1 score: {:.4f}'.format(f1))
    print('Test AUROC: {:.4f}'.format(auroc))
    return model


def train():
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss_y = criterion(out[data.train_mask], data.y[data.train_mask])
    rob = measure_robustness()
    loss = loss_y * (1 - rob)
    loss.backward()
    optimizer.step()
    return loss


def train_stab():
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss_y = criterion(out[data.train_mask], data.y[data.train_mask])
    stab = measure_stability()
    loss = loss_y * (1 - stab)
    loss.backward()
    optimizer.step()
    return loss


def measure_robustness():
    if algo in ["grad", "gbp", "ig"]:
        weights=np.array([0.5, 0.5, 0])
    elif algo in ["subx", "pgex"]:
        weights=np.array([0, 0.5, 0.5])
    elif algo in ["gnnex"]:
        weights= np.array([1/3, 1/3, 1/3])
    else:
        weights = np.array([0,1,0])
    metric = lambda e1, e2: expl_dist(e1, e2, weights=weights)
    return robustness_graph(dataset, model, criterion, algo, metric, y_pred=pred, device=device, pgex=pgex)



def measure_stability():
    X_act = activation['layer']
    X_dist = cdist(X_act, X_act).cpu()

    pgex = None
    if algo == "pgex":
        pgex = PGExplainer(model, emb_layer_name='gin3', max_epochs=10, lr=0.1)
        pgex.train_explanation_model(data)
    E = graph_explain_all(dataset, model, criterion, data.x, data.y, data.edge_index, algo, pred, pgex=pgex)

    weights = np.array([0, 1, 0])
    if algo in ["grad", "gbp", "ig"]:
        weights = np.array([0.5, 0.5, 0])
    if algo in ["subx", "pgex"]:
        weights = np.array([0, 0.5, 0.5])
    if algo in ["gnnex"]:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
    E_dist = np.zeros((dataset.num_nodes, dataset.num_nodes))
    for i in range(dataset.num_nodes - 1):
        e1 = E[i]
        for j in range(i + 1, dataset.num_nodes):
            e2 = E[j]
            D = expl_dist(e1, e2, weights=weights)
            E_dist[i, j] = D
            E_dist[j, i] = D

    stab = stability(X_dist,
                     E_dist,
                     algorithm=AgglomerativeClustering,
                     min_k=2,
                     max_k=20,
                     data_cluster_kwargs=data_cluster_kwargs,
                     explanation_cluster_kwargs=explanation_cluster_kwargs)
    return stab.max()

# Explain
algos = [#'gnnex',
    'grad','cam','gcam','gbp',#'glime',
         'ig',#'glrp','pgmex','rand',
         'subx', 'pgex']

robustness = {}
robustness_after = {}
for algo in tqdm.tqdm(algos):
    model = pretrain()
    pred = model(data.x.to(device), data.edge_index.to(device))
    pgex = None
    if algo == "pgex":
        pgex = PGExplainer(model, emb_layer_name='gin3', max_epochs=10, lr=0.1)
        pgex.train_explanation_model(data)
    robustness[algo] = measure_robustness()
    print("BEFORE", algo, robustness[algo])

    for _ in range(3):
        loss = train()
    robustness_after[algo] = measure_robustness()
    print("AFTER", algo, robustness_after[algo])



delta = calculate_delta(dataset, torch.where(data.train_mask == True)[0], model, data.y)
gef_feat = {k: [] for k in algos}
gef_node = {k: [] for k in algos}
gef_edge = {k: [] for k in algos}
ges_feat = {k: [] for k in algos}
ges_node = {k: [] for k in algos}
ges_edge = {k: [] for k in algos}
gea_feat = {k: [] for k in algos}
gea_node = {k: [] for k in algos}
gea_edge = {k: [] for k in algos}

for node_idx in tqdm.tqdm((data.test_mask == True).nonzero(as_tuple=True)[0]):
    node_idx = node_idx.item()
    gt_exp = dataset.explanations[node_idx]
    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0)

    for exp_method in algos:
        # Get explanation method
        if exp_method == 'pgex':
            explainer = pgex
            forward_kwargs = {'node_idx': node_idx,
                              'x': data.x.to(device),
                              'edge_index': data.edge_index.to(device),
                              'label': pred_class}
        else:
            explainer, forward_kwargs = get_exp_method(exp_method, model, criterion, dataset, data.x, data.y,
                                                       data.edge_index, node_idx, pred_class)
        exp = explainer.get_explanation_node(**forward_kwargs)

        # Calculate metrics
        feat, node, edge = graph_exp_acc(gt_exp, exp)
        gea_feat[exp_method].append(feat)
        gea_node[exp_method].append(node)
        gea_edge[exp_method].append(edge)

        feat, node, edge = graph_exp_stability(exp, explainer, dataset, node_idx, model, delta, sens_idx=[],
                                               device=device)
        ges_feat[exp_method].append(feat)
        ges_node[exp_method].append(node)
        ges_edge[exp_method].append(edge)

        feat, node, edge = graph_exp_faith(exp, dataset, model, sens_idx=[])
        gef_feat[exp_method].append(feat)
        gef_node[exp_method].append(node)
        gef_edge[exp_method].append(edge)


df_gef_feat = pd.DataFrame(gef_feat)
df_gef_feat.to_csv("results/gef_feat.csv", index=False)
df_gef_node = pd.DataFrame(gef_node)
df_gef_node.to_csv("results/gef_node.csv", index=False)
df_gef_edge = pd.DataFrame(gef_edge)
df_gef_edge.to_csv("results/gef_edge.csv", index=False)
df_ges_feat = pd.DataFrame(ges_feat)
df_ges_feat.to_csv("results/ges_feat.csv", index=False)
df_ges_node = pd.DataFrame(ges_node)
df_ges_node.to_csv("results/ges_node.csv", index=False)
df_ges_edge = pd.DataFrame(ges_edge)
df_ges_edge.to_csv("results/ges_edge.csv", index=False)
df_gea_feat = pd.DataFrame(gea_feat)
df_gea_feat.to_csv("results/gea_feat.csv", index=False)
df_gea_node = pd.DataFrame(gea_node)
df_gea_node.to_csv("results/gea_node.csv", index=False)
df_gea_edge = pd.DataFrame(gea_edge)
df_gea_edge.to_csv("results/gea_edge.csv", index=False)

print(gef_feat)

print(pd.DataFrame(gef_feat))

df_all = [df_gef_feat,df_gef_node,df_gef_edge,df_ges_feat,df_ges_node,df_ges_edge,df_gea_feat,df_gea_node,df_gea_edge]

means = {}
vars = {}
for exp_method in algos:
    temp_means = []
    temp_vars = []
    for df_temp in df_all:
        temp_means.append(df_temp[exp_method].mean())
        temp_vars.append(df_temp[exp_method].var())
    means[exp_method] = temp_means
    vars[exp_method] = temp_vars
df_means = pd.DataFrame(means).T
df_vars = pd.DataFrame(vars).T
df_means = df_means.rename(columns={0: "gef_feat",
                                      1: "gef_node",
                                      2: "gef_edge",
                                      3: "ges_feat",
                                      4: "ges_node",
                                      5: "ges_edge",
                                      6: "gea_feat",
                                      7: "gea_node",
                                      8: "gea_edge"})
df_vars = pd.DataFrame(data=df_vars.values, index=df_vars.index, columns=df_means.columns)
df_means.to_csv("results/results.csv")
df_vars.to_csv("results/results_var.csv")

for attribute in df_means:
    if attribute.startswith("gea"):
        best_idx = df_means[attribute].argmax()
    else:
        best_idx = df_means[attribute].argmin()
    best_algo = df_means.index[best_idx]
    print(attribute, best_algo, f"{df_means[attribute][best_algo]:1.3f}", u"\u00B1", f"{df_vars[attribute][best_algo] ** 0.5:1.3f}")

data_cluster_kwargs = {"linkage": "single", "metric": "precomputed"}
explanation_cluster_kwargs = {"linkage": "single", "metric": "precomputed"}
pgex = None
for algo in tqdm.tqdm(algos):
    model = pretrain()
    layer = getattr(model, "gin2")
    activation = {}
    def get_activation():
        def hook(model, inp, out):
            activation['layer'] = out.detach()
        return hook
    layer.register_forward_hook(get_activation())
    with torch.no_grad():
        pred = model(data.x.to(device), data.edge_index.to(device))
    stab = measure_stability()
    print("BEFORE", algo, stab)
    loss = train()
    stab_after = measure_stability()
    print("AFTER", algo, stab)

