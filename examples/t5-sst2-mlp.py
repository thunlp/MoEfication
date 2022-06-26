import os
import types
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import numpy as np
from transformers.models.t5.modeling_t5 import T5DenseActDense

k=20

tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()

sst2 = load_dataset('sst2')
sst2_dev = sst2['validation']

pred = []

def change_forward(model, k=20):

    def _forward(ffn_self, hidden_states):
        
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states_mlp = hidden_states.clone().detach()
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)

        hidden_states_mlp = hidden_states_mlp / torch.norm(hidden_states_mlp, dim=-1).unsqueeze(-1)
        score = ffn_self.mlp(hidden_states_mlp)

        labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
        cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
        
        hidden_states = ffn_self.wi(hidden_states)
        hidden_states = ffn_self.act(hidden_states)
        hidden_states[cur_mask == False] = 0  
        
        hidden_states = ffn_self.dropout(hidden_states)
        hidden_states = ffn_self.wo(hidden_states)
        return hidden_states

    def modify_ffn(ffn, path):
        assert type(ffn) == T5DenseActDense
        labels = torch.load(path)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda()
        ffn.k = k
        ffn.mlp = torch.load(path+'_input_compl').cuda()
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)   

    # encoder
    for layer_idx, layer in enumerate(model.encoder.block):
        ffn = layer.layer[1].DenseReluDense
        path = os.path.join('results/t5-base', 'param_split', 'encoder.block.{}.layer.1.DenseReluDense.wi.weight'.format(layer_idx))
        modify_ffn(ffn, path) 

    #decoder
    for layer_idx, layer in enumerate(model.decoder.block):
        ffn = layer.layer[2].DenseReluDense
        path = os.path.join('results/t5-base', 'param_split', 'decoder.block.{}.layer.2.DenseReluDense.wi.weight'.format(layer_idx))
        modify_ffn(ffn, path)    
        
change_forward(model, k)

# sst2 evaluation
for instance in sst2_dev:
    input_ids = tokenizer("sst2 sentence: "+instance['sentence'], return_tensors="pt").input_ids.cuda()
    dec_input_ids = tokenizer("<extra_id_0>", return_tensors="pt").input_ids.cuda()[:, :1]

    output = model(input_ids=input_ids, labels=dec_input_ids)

    pred.append(int(output.logits[:, 0, 1465].item() > output.logits[:, 0, 2841].item()) == instance['label'])

print("Acc", sum(pred) * 1. / len(pred), 'k', k)