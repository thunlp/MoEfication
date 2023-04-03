import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import time
import math
from fmoe.linear import MOELinear
import torch.nn.functional as F
from fmoe.layers import _fmoe_general_global_forward
import tree

class CustomGate(torch.nn.Module):
    def __init__(self, dim_in, expert_num, dtype, init_mean = 0.0, init_std = 0.02, length_scale=False):
        super().__init__()
        self.wg = torch.nn.Parameter(torch.empty((dim_in, expert_num), dtype=dtype))

    def forward(self, x):
        return torch.matmul(x, self.wg)

class CustomExpert(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden_size, expert_num, dtype, init_mean = 0.0, init_std = 0.02, dropout_p = 0.0, length_scale=False):
        super().__init__()

        hidden_size = hidden_size // expert_num
        self.batched_fc1_w = torch.nn.Parameter(torch.empty((expert_num, hidden_size, dim_in), dtype=dtype))
        self.batched_fc2_w = torch.nn.Parameter(torch.empty((expert_num, dim_out, hidden_size), dtype=dtype))

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.length_scale = length_scale

    def forward(self, x, fwd_expert_count):
        y = MOELinear.apply(x, fwd_expert_count, self.batched_fc1_w)
        y = F.relu(y)
        if self.dropout is not None:
            y = self.dropout(y)
        y = MOELinear.apply(y, fwd_expert_count, self.batched_fc2_w)
        return y

class FeedForward(torch.nn.Module):
    def __init__(self,
                 dim_in : int, 
                 dim_ff : int,
                 dim_out : int = None,
                 dtype = torch.half, 
                 int8 = False,
                 init_mean = 0.0, 
                 init_std = 0.02,
                 bias = False,
                 activate_fn = "gated_gelu",
                 length_scale : bool = False,
                 dropout_p = 0,
                 num_expert = 8,
                 top_k = 4,
        ):
        super().__init__()
        self.expert_num = num_expert
        self.gate = CustomGate(dim_in, self.expert_num, dtype, init_mean, init_std, length_scale=length_scale)
        self.experts = CustomExpert(dim_in, dim_out, dim_ff, self.expert_num, dtype, init_mean=init_mean, init_std=init_std, dropout_p=dropout_p, length_scale=length_scale)
        self.length_scale = length_scale
        self.k = top_k

    def forward(self, x):
        k = self.k
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size * seq_len, dim)

        gate = self.gate(x)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        fwd = _fmoe_general_global_forward(
            x, gate_top_k_idx, self.experts,
            self.expert_num, 1,
            experts=self.experts
        )

        def view_func(tensor):
            dim = tensor.shape[-1]
            tensor = tensor.view(-1, k, dim)
            return tensor

        moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, k)
        gate_score = torch.ones_like(gate_score) + gate_score - gate_score.detach()

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)
        
        moe_outp = moe_outp.view(batch_size, seq_len, -1)

        return moe_outp

top_k = 1
number_of_experts = 8

tokenizer = T5Tokenizer.from_pretrained('t5-large')
config = T5Config.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

from fmoe.transformer import FMoETransformerMLP

# replace FFN with MoE
for layer_idx, layer in enumerate(model.encoder.block):
    layer.layer[1].DenseReluDense = FeedForward(
        dim_in=config.d_model,
        dim_ff=config.d_ff,
        dim_out=config.d_model,
        num_expert=number_of_experts,
        top_k=top_k,
        dtype=torch.float32,
    )

for layer_idx, layer in enumerate(model.decoder.block):
    layer.layer[1].DenseReluDense = FeedForward(
        dim_in=config.d_model,
        dim_ff=config.d_ff,
        dim_out=config.d_model,
        num_expert=number_of_experts,
        top_k=top_k,
        dtype=torch.float32,
    )

# old moe
# for layer_idx, layer in enumerate(model.encoder.block):
#     layer.layer[2].DenseReluDense = FMoETransformerMLP(
#         num_expert=number_of_experts,
#         d_model=config.d_model,
#         d_hidden=config.d_ff // number_of_experts,
#         top_k=top_k,
#         activation=torch.nn.ReLU()
#     )

# for layer_idx, layer in enumerate(model.decoder.block):
#     layer.layer[2].DenseReluDense = FMoETransformerMLP(
#         num_expert=number_of_experts,
#         d_model=config.d_model,
#         d_hidden=config.d_ff // number_of_experts,
#         top_k=top_k,
#         activation=torch.nn.ReLU()
#     )

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
