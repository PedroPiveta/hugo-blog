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

## PyTorch Workflow

The complete PyTorch workflow involves the following steps:

1. Data (prepare and load)
2. Build model
3. Fit the model to data (training)
4. Making predictions (inference)
5. Saving and loading the model

### 1. Data (preparing and loading)

In machine learning, data can be almost anything: spreadsheets, images, videos, audio, DNA, text, etc.

Machine learning is a game of two parts:

1. Get data into a numerical representation
2. Build a model to learn patterns in that numerical representation

To showcase this, let's create some known data using a linear regression formula with **known parameters**.

```python
import torch
from torch import nn
import matplotlib.pyplot as plt

# Known parameters
weight = 0.7
bias = 0.3

# Create the data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias  # linear regression formula: y = weight * X + bias
```

#### Splitting into training and test sets

```python
# Train/test split (80% train, 20% test)
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
```

```python
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Plot predictions if provided
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
```

### 2. Building the model

```python
from torch import nn

# Creating the linear regression model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters the model will learn (randomly initialized)
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    # Forward method defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
```

#### PyTorch model building essentials

- `torch.nn` — contains all the building blocks for computational graphs
- `torch.nn.Parameter` — defines which parameters the model should try to learn
- `torch.nn.Module` — the base class for all neural network models; you must override `forward`
- `torch.optim` — where the optimizers in PyTorch live
- `def forward()` — all `nn.Module` subclasses require you to override this method, which defines the forward computation

#### Checking the contents of the model

```python
# Create a random seed for reproducibility
torch.manual_seed(42)

# Create an instance of the model
model_0 = LinearRegressionModel()

# List the model parameters
list(model_0.parameters())

# Check the state dictionary (current weights and biases)
model_0.state_dict()
```

#### Making predictions with `torch.inference_mode()`

```python
# Make predictions with the model (no gradient tracking needed)
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)
```

### 3. Training the model

The whole idea of training is to move the model from _unknown_ parameters to _known_ parameters (close to the real ones).

To train, we need:

- **Loss function:** measures how wrong the model's predictions are compared to the ideal outputs. Lower is better.
- **Optimizer:** uses the loss to adjust the model's parameters to improve predictions.
- A **training loop**
- A **testing loop**

```python
# Setup a loss function (MAE - Mean Absolute Error)
loss_fn = nn.L1Loss()

# Setup an optimizer (SGD - Stochastic Gradient Descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)  # lr = learning rate
```

#### Building a training loop

Steps of the training loop:

0. Loop through the data (epoch loop)
1. **Forward pass:** data moves through the model's `forward()` method
2. **Calculate the loss:** compare predictions to the ground truth labels
3. **Zero the optimizer gradients**
4. **Backpropagation:** calculate the gradients of each parameter with respect to the loss
5. **Optimizer step (gradient descent):** adjust the parameters to reduce the loss

```python
torch.manual_seed(42)

epochs = 200  # one epoch = one full pass through the data

# Track values over training
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    # Training mode: enables gradient tracking
    model_0.train()

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero out accumulated gradients from the previous step
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Update the parameters
    optimizer.step()

    ### Testing
    model_0.eval()  # disables gradient tracking
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())
```

```python
import numpy as np

# Plot the training and test loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
```

### 4. Device-agnostic code

To take advantage of a GPU when available, we write device-agnostic code:

```python
# Check if a GPU is available; otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

In the improved version of the model, we use `nn.Linear` instead of manually defining parameters:

```python
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear automatically creates the weight and bias parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()

# Send the model to the available device (GPU or CPU)
model_1.to(device)

# Data must be on the same device as the model
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
```

### 5. Saving and loading a model

There are three main methods for saving and loading models in PyTorch:

1. `torch.save()` — saves a PyTorch object in Python's pickle format
2. `torch.load()` — loads a saved PyTorch object
3. `torch.nn.Module.load_state_dict()` — loads a model's saved state dictionary

#### Saving the model

```python
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Define the save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save only the state_dict (parameters), not the entire model object
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)
```

#### Loading the model

```python
# To load, we must instantiate the model class again
loaded_model_0 = LinearRegressionModel()

# Load the saved state_dict into the new instance
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Evaluate the loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# Verify predictions match the original model's output
y_preds == loaded_model_preds
```

---

_This post will be continuously updated with new concepts and practical examples._

