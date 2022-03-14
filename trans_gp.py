import sys
import torch
from collections import defaultdict
import random
import os

filename = sys.argv[1]

labels = []

with open(filename) as fin:
    d = defaultdict(list)
    for i, line in enumerate(fin):
        labels.append(int(line.strip()))
        d[labels[-1]].append(i)

need_move = []

for i in range(max(d.keys())+1):
    if i not in d:
        d[i] = []
print(len(labels), len(d.keys()))

num = len(labels) // len(d.keys())
for k, v in d.items():
    if len(v) > num:
        random.shuffle(v)
        for i in range(num, len(v)):
            need_move.append(v[i])
        d[k] = v[:num]

print("need_move", need_move)

random.shuffle(need_move)
for k, v in d.items():
    if len(v) < num:
        pos = num-len(v)
        v += need_move[:pos]
        need_move = need_move[pos:]
    for x in v:
        labels[x] = k

vec = os.path.basename(filename).split('_')
t = vec[0]
layer = vec[3]

torch.save(labels, os.path.join(os.path.dirname(filename), 'gp_split', "{}_layer_{}".format(t, layer)))

from collections import Counter

print(Counter(labels))
