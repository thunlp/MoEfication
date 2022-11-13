import os
import types
import tqdm
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import numpy as np

batch_size = 32
gradient_accumulation_steps = 4
mini_batch_size = batch_size // gradient_accumulation_steps
folder = "bert-sst2-bsz32"
ckpt_path = "relu-bert-large-uncased.bin"

if not os.path.exists(folder):
    os.makedirs(folder)

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
sst2_train = sst2['train']

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
all_step = len(sst2_train) // batch_size
warmup_step = all_step // 10
lr_lambda = lambda step: min(step / (warmup_step + 1e-8), 1.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
dataloaders = torch.utils.data.DataLoader(sst2_train, batch_size=mini_batch_size, shuffle=True)

sst2_eval = sst2['validation']
eval_dataloaders = torch.utils.data.DataLoader(sst2_eval, batch_size=mini_batch_size)

for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    step = 0
    for batch in tqdm.tqdm(dataloaders):
        inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        labels = torch.tensor(batch['label'])
        inputs = {k: v.cuda() for k, v in inputs.items()}
        labels = labels.cuda()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        print(loss.item())

        step += 1
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), '{}/epoch_{}.bin'.format(folder, epoch))

    model.eval()
    correct = 0
    total = 0
    for batch in tqdm.tqdm(eval_dataloaders):
        inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        labels = torch.tensor(batch['label'])
        inputs = {k: v.cuda() for k, v in inputs.items()}
        labels = labels.cuda()
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        pred = logits.argmax(dim=-1)
        tmp_correct = (pred == labels).sum().item()
        correct += tmp_correct
        total += len(labels)
    print("Acc", correct * 1. / total)
