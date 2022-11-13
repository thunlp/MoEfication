import os
import types
import tqdm
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import BertIntermediate
import numpy as np

batch_size = 8
k = 20
ckpt_path = 'bert-sst2-bsz32/epoch_1.bin'

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
config = BertConfig.from_pretrained("bert-large-uncased")

# transform BERT to relu-based BERT
config.hidden_act = 'relu'
config.num_labels = 2
config.problem_type = "single_label_classification"
model = BertForSequenceClassification(config=config)
for x in model.bert.encoder.layer:
    x.intermediate.dense.bias = None

res = model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
print(res)
model.cuda()

sst2 = load_dataset('sst2')

sst2_eval = sst2['validation']
eval_dataloaders = torch.utils.data.DataLoader(sst2_eval, batch_size=batch_size)

def change_forward(model, k=20):

    def _forward(ffn_self, hidden_states):
        hidden_states = ffn_self.forward_old(hidden_states)

        if ffn_self.patterns is not None:
            # golden
            k = ffn_self.k
            bsz, seq_len, hidden_size = hidden_states.shape
            hidden_states_relu = hidden_states.clone()
            hidden_states_relu = hidden_states_relu.view(-1, hidden_size)
            score = torch.matmul(hidden_states_relu, ffn_self.patterns.transpose(0, 1))
            labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
            cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
            hidden_states[cur_mask == False] = 0  
        
        return hidden_states

    def modify_ffn(ffn, path):
        assert type(ffn) == BertIntermediate
        labels = torch.load(path)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda()
        ffn.k = k
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)   

    # encoder
    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        ffn = layer.intermediate
        path = os.path.join('results/bert-sst2', 'param_split', 'bert.encoder.layer.{}.intermediate.dense.weight'.format(layer_idx))
        modify_ffn(ffn, path) 

change_forward(model, k)

model.eval()
correct = 0
total = 0
for batch in tqdm.tqdm(eval_dataloaders):
    inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=128)
    labels = torch.tensor(batch['label'])
    inputs = {k: v.cuda() for k, v in inputs.items()}
    labels = labels.cuda()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    logits = outputs.logits
    pred = logits.argmax(dim=-1)
    tmp_correct = (pred == labels).sum().item()
    correct += tmp_correct
    total += len(labels)
print("Acc", correct * 1. / total)
