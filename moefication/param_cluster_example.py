from ast import arg
from re import template
import os
import utils
import numpy as np
import torch
import argparse
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='results/t5-base/ckpt.bin', help='path to the model checkpoint')
parser.add_argument('--res_path', type=str, default='results/t5-base/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=12, help='number of layers')
parser.add_argument('--num-expert', type=int, default=96, help='number of experts')
parser.add_argument('--templates', type=str, default='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

args = parser.parse_args()
if not os.path.exists(args.res_path):
    os.makedirs(args.res_path)

config = utils.ModelConfig(args.model_path, args.res_path, split_num=args.num_expert)

templates = args.templates.split(',')

for template in templates:
    for i in tqdm.tqdm(range(args.num_layer)):
        split = utils.ParamSplit(config, template, i)
        split.split()
        split.cnt()
        split.save()
