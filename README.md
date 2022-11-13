
# MoEfication

Source code for "[MoEfication: Transformer Feed-forward Layers are Mixtures of Experts](https://arxiv.org/abs/2110.01786)"

## Reqirements:

* Python3.8
* torch==1.6.0
* transformers==4.20.1
* tqdm
* scikit-learn
* k_means_constrained
* datasets
* numpy
* scipy

## Expert Construction

For parameter clustering split, we use balanced K-Means. The details of the implementation can be found in `param_cluster_example.py`.

For co-activation graph split, we first construct a co-activation graph by `adj.py`. For T5, the output graphs are named as `encoder.blocks.0.ff.dense_relu_dense.wi.weight`, `encoder.blocks.1.ff.dense_relu_dense.wi.weight`, ..., `decoder.blocks.11.ff.dense_relu_dense.wi.weight`, which are the weight names.

Then, we use [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) to split the graph into subgraphs.
```
gpmetis encoder.blocks.0.ff.dense_relu_dense.wi.weight num_expert
```
where `num_expert` is the number of experts.

Finally, we balance the neurons in each expert.
```
# num_expert=128
python trans_gp.py encoder.blocks.0.ff.dense_relu_dense.wi.weight.part.128
```

## Expert Selection

For similarity selection, we average the corresponding weight columns as the expert representation. The details of the implementation can be found in `similarity_select_example.py`.

For MLP selection, We train a multi-layer perceptron (MLP), which takes the $\vx$ as input and predicts the sum of positive values in each expert. The details of the implementation can be found in `mlp_select_example.py`.

## T5 Examples

We provide an example of T5-base on SST-2 in `examples`, including groundtruth selection and MLP selection based on parameter clustering.

First, you need to construct expert by 

```
python moefication/param_cluster_example.py
```

Then, you can directly evaluate groundtruth selection by 

```
python examples/t5-sst2-gt.py
```

To use MLP selection, you need to train the MLP by 

```
python examples/t5-sst2-inf.py
python moefication/mlp_select_example.py 
```

And, you can evaluate the performance of MLP selection by 

```
python examples/t5-sst2-mlp.py
```

## BERT Examples

We also provide an example of BERT-large on SST-2 in `examples`. The checkpoint of ReLU-based BERT is available [here](https://cloud.tsinghua.edu.cn/f/cce7d1c994904f0f81bd/?dl=1). 

You need to first download it and fine-tune it on SST-2 by 

```
python examples/bert-sst2-training.py
```

Then, you need to construct expert by 

```
python moefication/param_cluster_example.py --model_path bert-sst2-bsz32/epoch_1.bin --res_path results/bert-sst2 --num-layer 24 --num-expert 128 --templates bert.encoder.layer.{}.intermediate.dense.weight
```

you can evaluate groundtruth selection by 

```
python examples/bert-sst2-gt.py
```

## Cite

If you use the code, please cite this paper:

```
@inproceedings{zhang2022moefication,
  title={{MoEfication}: Transformer Feed-forward Layers are Mixtures of Experts},
  author={Zhang, Zhengyan and Lin, Yankai and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
  booktitle={Findings of ACL 2022},
  year={2022}
}
```
