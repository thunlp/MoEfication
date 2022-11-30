import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import time

top_k = 4
number_of_experts = 8

tokenizer = T5Tokenizer.from_pretrained('t5-large')
config = T5Config.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

from fmoe.transformer import FMoETransformerMLP

# replace FFN with MoE
for layer_idx, layer in enumerate(model.encoder.block):
    layer.layer[1].DenseReluDense = FMoETransformerMLP(
        num_expert=number_of_experts,
        d_model=config.d_model,
        d_hidden=config.d_ff // number_of_experts,
        top_k=top_k,
        activation=torch.nn.ReLU()
    )

for layer_idx, layer in enumerate(model.decoder.block):
    layer.layer[2].DenseReluDense = FMoETransformerMLP(
        num_expert=number_of_experts,
        d_model=config.d_model,
        d_hidden=config.d_ff // number_of_experts,
        top_k=top_k,
        activation=torch.nn.ReLU()
    )

model.cuda()
input_ids = torch.randint(0, config.vocab_size, (64, 64), dtype=torch.long).cuda()
dec_input_ids = torch.randint(0, config.vocab_size, (64, 64), dtype=torch.long).cuda()
# input_ids = torch.zeros((64, 64), dtype=torch.long).cuda()
# dec_input_ids = torch.zeros((64, 64), dtype=torch.long).cuda()

res = []

for _ in range(100):
    torch.cuda.synchronize()
    t_start = time.time()

    with torch.no_grad():
        output = model(input_ids=input_ids, labels=dec_input_ids)

    torch.cuda.synchronize()
    t_end = time.time()
    res.append(t_end - t_start)
    print(t_end - t_start)

res = res[10:]
print("mean: ", sum(res)/len(res))
