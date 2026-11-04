import time

import networkx
import numpy as np
import sklearn.cluster
import torch
from sklearn.metrics import pairwise_distances
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchviz import make_dot

import stability
import robustness
import matplotlib.pyplot as plt
from saliency_evaluation import config, loader
from saliency_evaluation.eval_infid_sen import Args, OutputLog
from saliency_evaluation.infid_sen_utils import evaluate_infid_sen, get_explanation_pdt
from saliency_evaluation.loader import Mnist_model
from util import ssim



def explanations(X, y):
    E = np.zeros(X.shape)
    for i, (img, imgy) in enumerate(zip(X, y)):
        img = img.reshape(1, 1, 28, 28).to(device)
        # expl, _ = get_explanation_pdt(img, model, imgy, exp, n_samples=1)
        expl, _ = get_explanation_pdt(img, model, imgy, "Smooth_Grad", sg_r = args.sg_r, sg_N = args.sg_N,
                                                given_expl=sg, binary_I=binary_I, n_samples=1) # sg mode
        E[i] = expl.reshape(1,28,28)
    return E


def train():
    model.train()
    total_loss = total_examples = total_y = total_stab = 0
    for X, y in train_loader:
        optimizer.zero_grad()

        # compute explanations
        E = explanations(X, y)

        # Stability
        k = len(y.unique())
        # X_d = pairwise_distances(X.reshape(-1, 28 * 28), metric=ssim, data_range=1)
        E_d = pairwise_distances(E.reshape(-1, 28 * 28), metric=ssim, data_range=1)
        # stab = stability.stability_v2(X_d, E_d, algorithm=sklearn.cluster.AgglomerativeClustering,
        #                               min_k=k, max_k=k,
        #                               metric="precomputed", linkage="single").max()
        # stab = stability.stability_labels(y, E_d, algorithm=sklearn.cluster.AgglomerativeClustering,
        #                               min_k=k, max_k=k,
        #                               metric="precomputed", linkage="single").max()

        # Forward pass
        out = model(X.to(device))
        loss_y = criterion(out, y.to(device))
        # print("Loss", loss)
        # print(out.argmax(axis=1), y)
        stab = stability.stability_labels(out.argmax(axis=1).cpu(), E_d,
                                          algorithm=sklearn.cluster.AgglomerativeClustering,
                                          min_k=2, max_k=k, metric="precomputed", linkage="single").max()
        loss = loss_y * (1 - stab)
        # print(1-stab, loss_y, loss)

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(X)
        total_y += float(loss_y.detach().cpu()) * len(X)
        total_stab += float(stab) * len(X)
        total_examples += len(X)
    return total_loss / total_examples, total_y / total_examples, total_stab / total_examples


def test():
    values = {'train_y': [], 'test_y': [], 'train_E': [], 'test_E': []}


    for X, y in train_loader:
        model.eval()
        out = model(X.to(device))
        values["train_y"].append(criterion(out, y.to(device)).detach().cpu().numpy())

        model.train()
        E = explanations(X, y)
        k = len(y.unique())
        E_d = pairwise_distances(E.reshape(-1, 28 * 28), metric=ssim, data_range=1)
        stab = stability.stability_labels(out.argmax(axis=1).cpu(), E_d,
                                          algorithm=sklearn.cluster.AgglomerativeClustering,
                                          min_k=2, max_k=k, metric="precomputed", linkage="single").max()
        values["train_E"].append(stab)

    for X, y in test_loader:
        model.eval()
        out = model(X.to(device))
        values["test_y"].append(criterion(out, y.to(device)).detach().cpu().numpy())

        model.train()
        E = explanations(X, y)
        k = len(y.unique())
        E_d = pairwise_distances(E.reshape(-1, 28 * 28), metric=ssim, data_range=1)
        stab = stability.stability_labels(out.argmax(axis=1).cpu(), E_d,
                                          algorithm=sklearn.cluster.AgglomerativeClustering,
                                          min_k=2, max_k=k, metric="precomputed", linkage="single").max()
        values["test_E"].append(stab)
    return values

print("CUDA?", torch.cuda.is_available(), torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

args = Args(config.args)
output_log = OutputLog()


# Training
batch_size = 256
N = 10000
N_test = 4000
train_loader, test_loader = loader.mnist_loaders(batch_size=batch_size, N=N, N_test=N_test)
#print(len(train_loader), len(test_loader))

# pert = 'Gaussian'
# exp = 'Grad'
for pert in ['Gaussian']: #args.perts:
    # for exp in ['Grad', 'Int_Grad', 'GBP']: #args.exps:
    for sg in ['Grad', 'GBP']: # args.sgs:
        print('Perturbation =', pert, '/ Explanation =', sg, "-SG")
        exp = "Smooth-Grad"
        # print('Perturbation =', pert, '/ Explanation =', exp)


        if pert == 'Square':
            binary_I = True
        elif pert == 'Gaussian':
            binary_I = False

        # Model
        model = loader.mnist_load_model(f"saliency_evaluation/{args.model}", state_dict=True, tf=True)
        # model = Mnist_model()
        model = model.to(device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-5)
        # optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
        # lr = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        # Evaluation before training
        stab_before = []
        for X, y in test_loader:
            out = model(X.to(device))
            E = explanations(X, y)
            k = len(y.unique())
            E_d = pairwise_distances(E.reshape(-1, 28 * 28), metric=ssim, data_range=1)
            stab = stability.stability_labels(out.argmax(axis=1).cpu(), E_d,
                                              algorithm=sklearn.cluster.AgglomerativeClustering,
                                              min_k=2, max_k=k, metric="precomputed", linkage="single").max()
            stab_before.append((1 - stab))
            # dot = make_dot(loss, show_attrs=True, show_saved=True)
            # dot.view()
            # break
        print(f"Stability before training: {np.mean(stab_before):.4f}")

        # Training
        epochs = 3
        train_hist = np.zeros(epochs)
        test_hist = np.zeros(epochs)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            train_loss, loss_y, loss_stab = train()
            #with torch.no_grad():
            test_loss = test()
            train_hist[epoch] = np.mean(test_loss["train_y"])
            test_hist[epoch] = np.mean(test_loss["test_y"])
            pbar.set_description(
                f'Epoch: {epoch + 1:04d}, '
                # f'LR: {lr.get_last_lr()[0]:.6f}, '
                f'Loss: {train_loss:.4f}, Loss_y: {loss_y:.4f}, Loss_stab: {loss_stab:.4f}, '
                f'Train: {train_hist[epoch]:.4f}, '
                f'Test: {test_hist[epoch]:.4f}')
            model.checkpoint(f"model/train_{epoch:04d}.pth")
            # lr.step(test_hist[epoch])

        # if epoch < epochs:
        #     best_epoch = test_hist[:epoch + 1].argmin()
        # else:
        #     best_epoch = test_hist.argmin()
        # print(f"best_epoch: {best_epoch}, loss: {test_hist[best_epoch]:.4f}")
        # model.resume(f"model/pretrain_{best_epoch:04d}.pth")

        # Evaluation after training
        stab_after = []
        for X, y in test_loader:
            out = model(X.to(device))
            E = explanations(X, y)
            k = len(y.unique())
            E_d = pairwise_distances(E.reshape(-1, 28 * 28), metric=ssim, data_range=1)
            stab = stability.stability_labels(out.argmax(axis=1).cpu(), E_d,
                                              algorithm=sklearn.cluster.AgglomerativeClustering,
                                              min_k=2, max_k=k, metric="precomputed", linkage="single").max()
            stab_after.append((1 - stab))
            # dot = make_dot(loss, show_attrs=True, show_saved=True)
            # dot.view()
            # break
        print(f"Stability after training: {np.mean(stab_after):.4f}")


# model.eval()
# acc_time = 0
# batches = 0
# with torch.no_grad():
#     for X, y in test_loader:  # Iterate in batches over the training/test dataset.
#         start_time = time.time_ns()
#         out = model(X.to(device))
#         acc_time += (time.time_ns() - start_time)
#         batches += 1
#         loss = cross_entropy(out, y.to(device)).detach().cpu().mean()
#         print(f"loss (batch) {loss:.2f}")
#         # break
# print(
#     f"Acc Time: {acc_time:.2f} ns, time per batch: {acc_time / batches:.2f} ns")


# exp = 'SHAP'
# infid, max_sen = evaluate_infid_sen(test_loader, model, exp, 'Square', args.sen_r, args.sen_N)
# output_log.write(exp, infid, max_sen)
# print(output_log)
# for exp in args.exps:
#     for pert in args.perts:
#         print('Perturbation =', pert, '/ Explanation =', exp)
#         infid, max_sen = evaluate_infid_sen(test_loader, model, exp, pert, args.sen_r, args.sen_N)
#         # evaluate_stab(test_loader, model, exp, pert, sen_r, sen_N)
#         output_log.write(exp, infid, max_sen)
#         print(output_log)
#
#
# for sg in args.sgs:
#     for pert in args.perts:
#         print('Perturbation =', pert, '/ Smooth-Grad =', sg)
#         infid, max_sen = evaluate_infid_sen(test_loader, model, 'Smooth_Grad', pert, args.sen_r, args.sen_N, args.sg_r,
#                                             args.sg_N,
#                                             given_expl=sg)
#         output_log.write(sg + '-SG', infid, max_sen)
#         print(output_log)



# train_loader, test_loader = loader.mnist_loaders(batch_size=1)
# N = 10
# E = {}
#
# pert = 'Gaussian' #args.perts[0]
# if pert == 'Square':
#     binary_I = True
# elif pert == 'Gaussian':
#     binary_I = False
#
# for exp in ['Square', 'Grad', 'Int_Grad', 'GBP']:
#     print('Perturbation =', pert, '/ Explanation =', exp)
#
#     Elist = []
#     for i, (X, y) in enumerate(test_loader):
#
#         if i >= N:
#             break
#
#         # fig = plt.figure()
#         # ax = fig.add_subplot(1,1,1)
#         # ax.imshow(X.reshape(28,28), cmap=plt.cm.gray, vmin=0, vmax=1)
#         # ax.axis('off')
#         # plt.show()
#
#
#         y = y.long()
#         if y.dim() == 2:
#             y = y.squeeze(1)
#
#         expl, pdt = get_explanation_pdt(X, model, y[0], exp, n_samples=10)
#         Elist.append(expl)
#     E[exp] = Elist
#
# for sg in args.sgs:
#     print('Perturbation =', pert, '/ Explanation =', sg, "-SG")
#
#     Elist = []
#     for i, (X, y) in enumerate(test_loader):
#
#         if i >= N:
#             break
#
#         y = y.long()
#         if y.dim() == 2:
#             y = y.squeeze(1)
#
#         expl, pdt = get_explanation_pdt(X, model, y[0], "Smooth_Grad", sg_r=args.sg_r, sg_N=args.sg_N,
#                                         given_expl=sg, binary_I=binary_I, n_samples=3)
#         Elist.append(expl)
#     E[f"{sg}-SG"] = Elist


