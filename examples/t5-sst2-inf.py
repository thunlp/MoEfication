import os
import types
import tqdm
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import numpy as np
from transformers.models.t5.modeling_t5 import T5DenseActDense

tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()

sst2 = load_dataset('sst2')
sst2_train = sst2['train']

pred = []

def change_forward(model):

    def _forward(ffn_self, hidden_states):
        ffn_self.res.append(hidden_states.detach().cpu())

        hidden_states = ffn_self.wi(hidden_states)
        hidden_states = ffn_self.act(hidden_states)
             
        hidden_states = ffn_self.dropout(hidden_states)
        hidden_states = ffn_self.wo(hidden_states)
        return hidden_states

    def modify_ffn(ffn, res):
        assert type(ffn) == T5DenseActDense
        ffn.res = res
        ffn.forward = types.MethodType(_forward, ffn)   

    # encoder
    res = {}
    for layer_idx, layer in enumerate(model.encoder.block):
        ffn = layer.layer[1].DenseReluDense
        name = 'encoder.block.{}.layer.1.DenseReluDense.wi.weight'.format(layer_idx)
        res[name] = []
        modify_ffn(ffn, res[name]) 

    #decoder
    for layer_idx, layer in enumerate(model.decoder.block):
        ffn = layer.layer[2].DenseReluDense
        name = 'decoder.block.{}.layer.2.DenseReluDense.wi.weight'.format(layer_idx)
        res[name] = []
        modify_ffn(ffn, res[name])   
    
    return res
        
res = change_forward(model)

# sst2 evaluation
for idx, instance in enumerate(tqdm.tqdm(sst2_train)):
    if idx == 10000:
        break

    input_ids = tokenizer("sst2 sentence: "+instance['sentence'], return_tensors="pt").input_ids.cuda()
    dec_input_ids = tokenizer("<extra_id_0>", return_tensors="pt").input_ids.cuda()[:, :1]

    output = model(input_ids=input_ids, labels=dec_input_ids)

    pred.append(int(output.logits[:, 0, 1465].item() > output.logits[:, 0, 2841].item()) == instance['label'])

print("Acc", sum(pred) * 1. / len(pred))

for k, v in res.items():
    v = [x.reshape(-1, x.shape[-1]) for x in v]
    v = torch.cat(v, dim=0)
    torch.save(v, 'results/t5-base/'+k)