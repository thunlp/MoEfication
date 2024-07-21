import torch
import torch.distributed
import tqdm
import random
import balanced_assignment
from faiss import Kmeans
import time

def cal_intertia_(data, centroids, labels):
    dist_ = torch.cdist(data, centroids)
    dist = dist_.gather(1, labels.unsqueeze(1)).squeeze(1)
    dist = dist.pow(2)
    return dist.sum(), dist_

def update_centers_(data, labels, centroids):
    for i in range(centroids.shape[0]):
        centroids[i] = data[labels == i].mean(0)



def kmeans_single(dat, k, size, niter=100, init=None):
    hidden_size = dat.shape[1]
    kmeans = Kmeans(hidden_size, k, gpu=1, nredo=1, seed=0)
    dat_cuda = torch.tensor(dat).cuda()
    if init is not None:
        cluster_centers_ = torch.zeros(k, hidden_size).cuda()
        update_centers_(dat_cuda, init, cluster_centers_)
        cluster_centers_ = cluster_centers_.cpu().numpy()
    else:
        cluster_centers_ = None
    kmeans.train(dat, init_centroids=cluster_centers_)

    dat = dat_cuda
    cluster_centers_ = torch.tensor(kmeans.centroids).cuda()
    
    labels_raw = torch.arange(dat.shape[0]).cuda()
    
    obj = 0
    last_centers = None
    labels = None

    dists = torch.cdist(dat, cluster_centers_) * -1
    for i in range(niter):
        # E step
        sorted_id, _ = balanced_assignment.balanced_assignment(dists, size*2)

        tmp_labels = torch.ones_like(sorted_id)
        tmp_labels[sorted_id] = torch.div(labels_raw, size, rounding_mode='floor')

        # M step
        last_centers = cluster_centers_.clone()
        update_centers_(dat, tmp_labels, cluster_centers_)

        tmp_obj, dists = cal_intertia_(dat, cluster_centers_, tmp_labels)
        tmp_obj = tmp_obj.item()
        dists = dists * -1
        if obj == 0 or tmp_obj < obj:
            obj = tmp_obj
            labels = tmp_labels

    return last_centers, labels, obj

def kmeans_balanced(dat, k, size, niter=100, nredo=1):
    best_obj = 0
    cluster_centers = None
    labels = None
    for i in range(nredo):
        tmp_cluster_centers, tmp_labels, tmp_obj = kmeans_single(dat, k, size, niter)
        if best_obj == 0 or tmp_obj < best_obj:
            best_obj = tmp_obj
            cluster_centers = tmp_cluster_centers
            labels = tmp_labels
    return cluster_centers, labels, best_obj

source_in_temp_orig = 'layers.{}.ffn.ffn.w_in.w_0.weight' # 'layers.{}.ffn.ffn.w_in.w.weight'
source_out_temp_orig = 'layers.{}.ffn.ffn.w_out.weight'
target_in_temp_orig = 'layers.{}.ffn.ffn.mlp.batched_fc1_w' #'layers.{}.ffn.ffn.experts.batched_fc1_w'
target_out_temp_orig = 'layers.{}.ffn.ffn.mlp.batched_fc2_w' #'layers.{}.ffn.ffn.experts.batched_fc2_w'
target_wg_temp_orig = 'layers.{}.ffn.ffn.router.wg' #'layers.{}.ffn.ffn.gate.wg'
# layers = [i for i in range(12)]
# model_types = ['encoder.', 'decoder.']


def init_ckpt(ckpt, split_num, layers, model_types):
    for model_type in model_types:
        source_in_temp = model_type + source_in_temp_orig
        source_out_temp = model_type + source_out_temp_orig
        target_in_temp = model_type + target_in_temp_orig
        target_out_temp = model_type + target_out_temp_orig
        target_wg_temp = model_type + target_wg_temp_orig

        for layer in tqdm.tqdm(layers):
            w_in = ckpt[source_in_temp.format(layer)]
            w_out = ckpt[source_out_temp.format(layer)]

            ckpt[target_in_temp.format(layer)] = w_in
            ckpt[target_out_temp.format(layer)] = w_out

            wg = w_in.view(split_num, -1, w_in.shape[-1]).mean(dim=1)
            wg_norm = wg / torch.norm(wg, dim=-1, keepdim=True) * w_in[0, :].norm()

            ckpt[target_wg_temp.format(layer)] = wg_norm.transpose(0, 1)

            del ckpt[source_in_temp.format(layer)]
            del ckpt[source_out_temp.format(layer)]

    return ckpt

import sklearn
import numpy as np

def split_ckpt(ckpt, split_num, layers, model_types, additional=False, structures=None):

    permutes = []
    idx = 0
    for model_type in model_types:
        if not additional:
            source_in_temp = model_type + source_in_temp_orig
            source_out_temp = model_type + source_out_temp_orig
        else:
            source_in_temp = model_type + target_in_temp_orig
            source_out_temp = model_type + target_out_temp_orig
        target_in_temp = model_type + target_in_temp_orig
        target_out_temp = model_type + target_out_temp_orig
        target_wg_temp = model_type + target_wg_temp_orig

        for layer in tqdm.tqdm(layers):
            w_in = ckpt[source_in_temp.format(layer)]
            w_out = ckpt[source_out_temp.format(layer)]
            hidden_size = w_in.shape[0]
            expert_size = hidden_size // split_num
            w_in_ = sklearn.preprocessing.normalize(w_in.float().numpy())

            if structures is None:
                centers, labels, obj = kmeans_balanced(w_in_, split_num, expert_size)
                # centers, labels, obj = kmeans_balanced(w_in.float().numpy(), split_num, expert_size)
            else:
                labels = structures[idx]
                idx += 1

            labels = torch.tensor(labels).cpu()
            tmp_in = []
            tmp_in_norm = []
            for i in range(split_num):
                tmp_in.append(w_in[labels == i, :])
                tmp_in_norm.append(w_in_[labels == i, :])
            tmp_in = torch.stack(tmp_in, dim=0)
            tmp_in_norm = np.stack(tmp_in_norm, axis=0)

            tmp_out = []
            for i in range(split_num):
                tmp_out.append(w_out[:, labels == i].transpose(0, 1))
            tmp_out = torch.cat(tmp_out, dim=0)

            # wg = tmp_in.mean(1)
            wg = tmp_in_norm.mean(1)
            wg = torch.tensor(wg, dtype=tmp_in.dtype, device=tmp_in.device)
            wg_norm = wg / torch.norm(wg, dim=-1, keepdim=True) * tmp_in[0, 0, :].norm()

            ckpt[target_wg_temp.format(layer)] = wg_norm.transpose(0, 1)

            if not additional:
                tmp_in = tmp_in.view(-1, tmp_in.shape[-1])
                
                ckpt[target_in_temp.format(layer)] = tmp_in
                ckpt[target_out_temp.format(layer)] = tmp_out
                
                del ckpt[source_in_temp.format(layer)]
                del ckpt[source_out_temp.format(layer)]
            else:
                permute = []
                for i in range(split_num):
                    permute.append((labels == i).nonzero().squeeze(1))
                permute = torch.cat(permute, dim=0)
                permutes.append(permute)

    if additional:
        return ckpt, permutes
    else:
        return ckpt

def get_labels(x, k, init=None):
    centers, labels, obj = kmeans_single(x, k, x.shape[0] // k, init=init)
    labels = labels.cpu().numpy()
    return labels, obj

def cal_structure(ckpt, split_num, layers, model_types, prev_structure):
    labels = []
    for model_type in model_types:
        source_in_temp = model_type + source_in_temp_orig
        for layer in tqdm.tqdm(layers):
            w_in = ckpt[source_in_temp.format(layer)]
            # a = w_in.float().numpy()
            a = sklearn.preprocessing.normalize(w_in.float().numpy())

            if prev_structure is None:
                l, obj = get_labels(a, split_num)
            else:
                l, obj = get_labels(a, split_num, init=prev_structure[len(labels)])
                l_random, obj_ = get_labels(a, split_num)
                if obj_ < obj:
                    l = l_random
            labels.append(l)
    return labels

def merge_ckpt(ckpt, layers, model_types):
    for model_type in model_types:
        source_in_temp = model_type + source_in_temp_orig
        source_out_temp = model_type + source_out_temp_orig
        target_in_temp = model_type + target_in_temp_orig
        target_out_temp = model_type + target_out_temp_orig
        target_wg_temp = model_type + target_wg_temp_orig

        for layer in tqdm.tqdm(layers):
            w_in = ckpt[target_in_temp.format(layer)]
            w_out = ckpt[target_out_temp.format(layer)]
            # wg = ckpt[target_wg_temp.format(layer)]

            # w_in = w_in.reshape(-1, w_in.shape[-1])
            w_out = w_out.transpose(0, 1)
            # w_out = w_out.reshape(w_out.shape[0], -1)

            ckpt[source_in_temp.format(layer)] = w_in
            ckpt[source_out_temp.format(layer)] = w_out

            del ckpt[target_in_temp.format(layer)]
            del ckpt[target_out_temp.format(layer)]
            del ckpt[target_wg_temp.format(layer)]
    
    return ckpt

def split_ckpt_random(ckpt, split_num, layers, model_types, additional=False, structures=None):
    for model_type in model_types:
        source_in_temp = model_type + source_in_temp_orig
        source_out_temp = model_type + source_out_temp_orig
        target_in_temp = model_type + target_in_temp_orig
        target_out_temp = model_type + target_out_temp_orig
        target_wg_temp = model_type + target_wg_temp_orig

        for layer in tqdm.tqdm(layers):
            w_in = ckpt[source_in_temp.format(layer)]
            w_out = ckpt[source_out_temp.format(layer)]
            hidden_size = w_in.shape[0]
            expert_size = hidden_size // split_num
            # equally split
            labels = [i for i in range(split_num) for _ in range(expert_size)]
            # shuffle
            random.shuffle(labels)
            labels = torch.tensor(labels)

            tmp_in = []
            for i in range(split_num):
                tmp_in.append(w_in[labels == i, :])
            tmp_in = torch.stack(tmp_in, dim=0)

            tmp_out = []
            for i in range(split_num):
                tmp_out.append(w_out[:, labels == i])
            tmp_out = torch.stack(tmp_out, dim=0)

            ckpt[target_in_temp.format(layer)] = tmp_in
            ckpt[target_out_temp.format(layer)] = tmp_out

            wg = tmp_in.mean(1)
            wg = torch.randn_like(wg)
            ckpt[target_wg_temp.format(layer)] = wg.transpose(0, 1)

            del ckpt[source_in_temp.format(layer)]
            del ckpt[source_out_temp.format(layer)]

    return ckpt
