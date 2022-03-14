import utils
import numpy as np
import torch
import tqdm
import sys
import os

task = sys.argv[1]
model_size = sys.argv[2]


model_path = 'model_ckpt.bin' # path to the model checkpoint
res_path = 'moefication_exaple' # path to store the results of moefication
# config for T5 base
hidden_size = 768*4
num_gpu = 4
num_layer = 12
batch_size = 8
max_instance = 200000

def run(proc_id):
    proc_id, model_path, res_path, hidden_size = proc_id
    cuda_dev = torch.device('cuda:{}'.format(proc_id))
    for is_encoder in [True, False]:
        for layer in range(num_layer):
            layer_id = int(is_encoder) * num_layer + layer
            if layer_id % num_gpu != proc_id:
                continue

            ffn = torch.tensor(utils.load_ffn_weight(model_path, layer, is_encoder))
            hidden = utils.load_hidden_states(res_path, layer, is_encoder)
            hidden = torch.cat(hidden, 0)

            cnt = 0
            adj = torch.zeros(hidden_size, hidden_size, device=cuda_dev).float()
            ffn = torch.tensor(ffn).to(cuda_dev).transpose(0, 1)
            for i in tqdm.tqdm(range(hidden.shape[0]//batch_size)):
                with torch.no_grad():
                    dat = hidden[i*batch_size:(i+1)*batch_size].to(cuda_dev) 
                    res = torch.nn.functional.relu(torch.matmul(dat, ffn)).unsqueeze(-1)
                    res = torch.clamp(torch.bmm(res, res.transpose(1, 2)).sum(0), max=1)
                    adj += res
            
                cnt += batch_size
                if cnt > max_instance:
                    break
            del hidden

            adj = adj.cpu().numpy()
            sub_folder = 'encoder' if is_encoder else 'decoder'
            target = os.path.join(res_path, "{}_activations_layer_{}".format(sub_folder, layer))

            threshold = 0
            pos = 10
            while threshold == 0:
                assert pos != 110
                threshold = np.percentile(adj.reshape(-1), pos)
                pos += 10
            print("threshold", threshold, layer_id, pos, adj.max())
            threshold = threshold * 0.99
            adj /= threshold

            with open(target+'_weight', "w") as fout:
                edges = 0
                for i in range(adj.shape[0]):
                    cnt = 0
                    for j in range(adj.shape[1]):
                        if i == j or adj[i, j] < 1:
                            pass
                        else:
                            cnt += 1
                    edges += cnt
                assert edges > 0
                fout.write("{} {} {}\n".format(adj.shape[0], edges // 2, "001"))
                for i in range(adj.shape[0]):
                    vec = []
                    for j in range(adj.shape[1]):
                        if i == j or adj[i, j] < 1:
                            pass
                        else:
                            val = int(adj[i, j])
                            vec.append([j+1, val])
                    fout.write(" ".join(["{} {}".format(x[0], x[1]) for x in vec]) + "\n")

import multiprocessing
pool = multiprocessing.Pool(processes=num_gpu)
pool.map(run, [(i, model_path, res_path, hidden_size) for i in range(num_gpu)])
pool.close()
pool.join()
