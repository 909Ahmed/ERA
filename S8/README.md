# Session 8 Assignment

Goal - Achieve 70% accuracy on CIFAR10 dataset, under 50k parameters and 20 epochs using different normalization layers.

[BatchNormalization](./batchnorm.ipynb)

### Operation: 
- Computes the mean and variance across the batch dimension and normalizes each feature dimension independently.
### Benefits:
- Mitigates the "internal covariate shift" problem during training.
- Reduces sensitivity to initialization.
- Acts as a regularizer, allowing for higher learning rates.

---

[LayerNormalization](./layernorm.ipynb)

### Operation:
- Instead of normalizing across batches, LN normalizes each example independently.### Benefits:
### Benefits
- Suitable for recurrent neural networks (RNNs) and transformers.
- Maintains performance even with small batch sizes or during inference on single examples.

---

[GroupNormalization](./groupnorm.ipynb)

### Operation: 
- Divides the channels into groups and computes normalization within each group.
- Generalization of Batch Normalization and Layer Normalization.
### Benefits:
- Less sensitive to batch size and the order of inputs.
- Suitable for scenarios where batch sizes are small or inconsistent.

![versus](https://pic1.zhimg.com/v2-27977bfda164cc78cbef54c5da70d503_1440w.jpg?source=172ae18b)

We have achieved our goal of achieving 99.4 (consistent at the end) accuracy on MNIST using [Model3](./model3.ipynb).