---
title: "Learning PyTorch"
date: 2025-01-11T21:34:35-03:00
tags: ["programming", "pytorch", "machine learning", "deep learning", "python"]
author: "Pedro Piveta Barrotti"
translationKey: pytorch
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Study notes about PyTorch and Machine Learning - A complete reference guide"
canonicalURL: ""
disableHLJS: false
disableShare: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "/images/aprendizagem-pytorch/logo.png"
    alt: "PyTorch Logo"
    caption: ""
    relative: false
    hidden: false
editPost:
    URL: "https://github.com/pedropiveta/hugo-blog/content/en"
    Text: "Suggest Changes"
    appendFilePath: true
---

# Purpose of the post

To serve as a reference guide for PyTorch and Machine Learning.

I plan to update this post as I progress with PyTorch.

# What is PyTorch?

PyTorch is a machine learning library developed by Facebook AI Research (FAIR). It is widely used for deep learning tasks and provides a flexible, dynamic interface for building and training neural networks.

## What are Tensors?

**Tensors** are fundamental data structures in _machine learning_ that generalize numbers, vectors, and matrices to multiple dimensions. They are used to represent and process numerical data efficiently, storing inputs, weights, and outputs of machine learning models. In frameworks like **TensorFlow** and **PyTorch**, tensors enable large-scale mathematical operations in an optimized manner, especially on GPUs, and allow automatic gradient computation, which is essential for training neural networks.

### Types of Tensors

**Matrix:** a table of values with rows and columns (2D)

```python
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
```

**Tensor:** a generalization of matrices to more dimensions (3D or higher)

```python
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
```

### Creating Tensors

**Random tensor:**

```python
# Creates a tensor of size (3, 4) with random values
random_tensor = torch.rand(3, 4)
```

**Tensor filled with zeros:**

```python
zeros = torch.zeros(size=(3, 4))
```

**Tensor from a range:**

```python
one_to_ten = torch.arange(start=1, end=11, step=1)
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

**Tensor-like filled with zeros:**

```python
ten_zeroes = torch.zeros_like(one_to_ten)
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

### Tensor Attributes

Extracting information about tensors (tensor attributes):

- Number of dimensions of a tensor: `tensor.ndim`
- Shape of a tensor: `tensor.shape`
- Data type of a tensor: `tensor.dtype`
- Device the tensor is on: `tensor.device`

## Tensor Manipulation

### Basic Operations

**Addition:**

```python
# Creates a tensor and adds 10 to all its elements
tensor = torch.tensor([1, 2, 3])
tensor + 10
# tensor([11, 12, 13])
```

**Subtraction:**

```python
# Subtracts 10 from all its elements
tensor - 10
# tensor([-9, -8, -7])
```

**Multiplication (element-wise):**

```python
# Multiplies all its elements by 10
tensor * 10
```

**Division:**

```python
# Divides all elements by 10
tensor / 10
```

### Matrix Multiplication

There are two main rules for performing matrix multiplication:

1. **The inner dimensions must match:**
   - `(3, 2) @ (3, 2)` doesn't work
   - `(2, 3) @ (3, 2)` works because the inner dimensions match

2. **The resulting matrix will have the shape of the outer dimensions:**
   - `(2, 3) @ (3, 2)` → `(2, 2)`
   - `(3, 2) @ (2, 3)` → `(3, 3)`

```python
# Performing matrix multiplication
torch.matmul(tensor, tensor)
```

### Matrix Transposition

**Transposition** swaps the **rows for columns** of a matrix, changing its orientation.
It's useful, for example, when we need to adjust dimensions so two matrices can be multiplied.

```python
# Transposing a matrix
tensor_t = tensor.T  # or tensor.transpose(0, 1)
```

Example:

- `tensor.shape = (2, 3)`
- `tensor_t.shape = (3, 2)`

With this, it's now possible to do:

```python
torch.matmul(tensor, tensor_t)  # (2, 3) @ (3, 2) -> (2, 2)
```

### Tensor Aggregation

**Tensor aggregation** combines values into a single result by applying functions like **minimum**, **maximum**, **mean**, or **sum** over one or more dimensions.

```python
torch.min(tensor)   # minimum value
torch.max(tensor)   # maximum value
torch.mean(tensor)  # mean of values
torch.sum(tensor)   # sum of all values
```

### Finding Positions

Finding the position of minimum and maximum:

```python
tensor.argmin()  # index of minimum value
tensor.argmax()  # index of maximum value
```

### Reshaping and Dimension Adjustment

Operations for reshaping, stacking, and adjusting tensor dimensions:

- **Reshape** – redefines the shape of a tensor to a new specified shape
- **View** – creates a _view_ of the tensor with another shape, **without copying data** in memory
- **Stack** – stacks multiple tensors, either **on top of each other** (vertically) or **side by side** (horizontally)
- **Squeeze** – removes all dimensions of size `1` from a tensor
- **Unsqueeze** – adds a dimension of size `1` at a specific position
- **Permute** – returns a _view_ of the tensor with dimensions **rearranged** in a new order

```python
# Examples will be added as learning progresses
```

---

*This post will be continuously updated with new concepts and practical examples.*
