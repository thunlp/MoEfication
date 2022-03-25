import utils
import numpy as np
import torch

model_path = '/data/home/scv0540/zzy/gpt-j/example/results/gpt-j-relu-new/checkpoints/ckpt-16000.pt' # path to the model checkpoint

res_path = '/data/home/scv0540/zzy/gpt-j/example/results/gpt-j-relu-new/' # path to store the results of moefication

num_layer = 28

config = utils.ModelConfig(model_path, res_path, split_num=512)

for i in range(num_layer):
    # Parameter Clustering Split
    split = utils.ParamSplit(config, "dec_layers.{}.ff.fc_in_weight", i)
    split.split()
    split.cnt()
    split.save()
