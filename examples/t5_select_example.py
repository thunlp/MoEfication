from tempfile import template
import numpy as np
import torch
import sys
import argparse
from transformers import T5ForConditionalGeneration
import os

sys.path.append('moefication')
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--res_path', type=str, default='results/t5-base/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=12, help='number of layers')
parser.add_argument('--num-expert', type=int, default=96, help='number of experts')
parser.add_argument('--templates', type=str, default='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

args = parser.parse_args()

model = T5ForConditionalGeneration.from_pretrained(args.model_name)
torch.save(model.state_dict(), os.path.join(args.res_path, 'model.pt'))

config = utils.ModelConfig(os.path.join(args.res_path, 'model.pt'), args.res_path, split_num=args.num_expert)

templates = args.templates.split(',')
for template in templates:
    for i in range(args.num_layer):
        center = utils.MLPCenter(config, template, '{}/param_split/{}'.format(args.res_path, template.format(i)), i)
        center.cal_center()
