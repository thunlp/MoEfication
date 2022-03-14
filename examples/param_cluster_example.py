import utils
import numpy as np
import torch

model_path = 'model_ckpt.bin' # path to the model checkpoint

res_path = 'moefication_exaple' # path to store the results of moefication

encoder_num, decoder_num = utils.get_layer_num(model_path)

config = utils.ModelConfig(model_path, res_path, split_num=128)

for is_encoder, num_layer in zip([True, False], [encoder_num, decoder_num]):
    for i in range(num_layer):
        # Parameter Clustering Split
        split = utils.ParamSplit(config, i, is_encoder=is_encoder)
        split.split()
        split.cnt()
        split.save()
