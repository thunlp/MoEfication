# Faster MoEfication

Source code for ICML 2024 paper "[Exploring the Benefit of Activation Sparsity in Pre-training](https://openreview.net/forum?id=KfXXPCcobh)"

Faster Moefication improves upon the parameter clustering method in Moefication by leveraging multi-GPU computation to significantly accelerate the model's parameter clustering process. This approach consists of two main parts:

1. Initial clustering using the k-means implementation provided by faiss-gpu.
2. Application of a balanced allocation algorithm on the k-means results to achieve a balanced cluster structure, leading to the final Moefication outcome.

This method facilitates the use of Moefication during the training process. In our ICML paper, we obtained a model that performs well under both dense and MoE sparse computations by alternating between dense and MoE sparse training. The MoE conversion in this process utilized the Faster Moefication method.

## Reqirements:

* Python3.8
* torch
* tqdm
* scikit-learn
* numpy
* faiss-gpu

Besides, users need to install our custom allocation algorithm by

```
cd balanced_assignment/
python setup.py install
```

## kmeans_balanced

The main function interface for Faster Moefication is `kmeans_balanced`. This function performs balanced k-means clustering on the input matrix.

Function signature:
```python
def kmeans_balanced(matrix, num_clusters, cluster_size, ...):
    ...
```

Main parameters:

1. matrix: The input matrix to be partitioned.
2. num_clusters: The number of clusters to create.
3. cluster_size: The size of each cluster.

This function first applies k-means clustering using `faiss-gpu`, then adjusts the clusters to ensure balanced sizes using our custom allocation algorithm.

## Usage Example

We provide a usage example in `main.py` that simulates the Moefication process for an eight-layer network. This example demonstrates the full workflow of Faster Moefication:

1. Parameter Distribution: The script starts by distributing parameters from a single GPU to eight different GPUs.
2. Layer-wise Clustering: It then performs clustering on each layer independently.
3. Result Gathering: Finally, it gathers the results back to a single GPU.

This process completes the Moefication of the model, leveraging multi-GPU parallelism for increased efficiency.

To run the example:

```bash
torchruntorchrun --nproc_per_node=8 main.py
```

The script will report the total time of this process.

## Acknowledgement

Our custom allocation algorithm is inspired by the expert allocation algorithm implemented by [Base Layers](https://arxiv.org/abs/2103.16716). We are grateful to the authors for their innovative approach, which has significantly influenced our work.

## Cite

If you use the code, please cite this paper:

```
@inproceedings{
  zhang2024exploring,
  title={Exploring the Benefit of Activation Sparsity in Pre-training},
  author={Zhengyan Zhang and Chaojun Xiao and Qiujieli Qin and Yankai Lin and Zhiyuan Zeng and Xu Han and Zhiyuan Liu and Ruobing Xie and Maosong Sun and Jie Zhou},
  booktitle={Proceedings of ICML},
  year={2024},
}

@inproceedings{zhang2022moefication,
  title={{MoEfication}: Transformer Feed-forward Layers are Mixtures of Experts},
  author={Zhang, Zhengyan and Lin, Yankai and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
  booktitle={Findings of ACL 2022},
  year={2022}
}
```