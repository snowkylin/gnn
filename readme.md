# Graph Neural Networks

TensorFlow implementation of several popular Graph Neural Network layers, wrapped with `tf.keras.layers.Layer`.

Currently, this repo contains:

- Graph Convolutional Network (GCN): `gnn.GCNLayer`
- Graph Attention Network (GAT): `gnn.GATLayer` and `gnn.MultiHeadGATLayer`

# Prerequisites

- TensorFlow 2.0

# References

- Kipf, Thomas N., and Max Welling. “Semi-Supervised Classification with Graph Convolutional Networks.” ArXiv:1609.02907 [Cs, Stat], September 9, 2016. http://arxiv.org/abs/1609.02907.
- Veličković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. “Graph Attention Networks.” ArXiv:1710.10903 [Cs, Stat], October 30, 2017. http://arxiv.org/abs/1710.10903.
