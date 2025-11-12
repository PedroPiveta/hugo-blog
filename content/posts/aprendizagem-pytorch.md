+++
title = "Aprendizagem Pytorch"
date = "2025-11-11T21:34:35-03:00"
#dateFormat = "2006-01-02" # This value can be configured for per-post date formatting
author = "Pedro Piveta Barrotti"
authorTwitter = "PedroPiBarrotti" #do not include @
cover = "/images/aprendizagem-pytorch/logo.png"
tags = ["programação", "pytorch", "machine learning"]
keywords = ["python", "pytorch", "machine learning"]
description = "Anotação de estudo sobre Pytorch"
showFullContent = false
readingTime = true
hideComments = false
draft = false
+++

# Intuito do post:

Servir como um guia de referência para PyTorch e Machine Learning.

Conforme for evoluindo no PyTorch

## O que é o PyTorch?

Pytorch é uma biblioteca de machine learning desenvolvida pelo Facebook AI Research (FAIR). Ela é amplamente utilizada para tarefas de aprendizado profundo (deep learning) e oferece uma interface flexível e dinâmica para a construção e treinamento de redes neurais.

## O que são Tensores?

Os **tensores** são estruturas de dados fundamentais em _machine learning_ que generalizam números, vetores e matrizes para múltiplas dimensões. Eles servem para representar e processar dados numéricos de forma eficiente, sendo usados para armazenar entradas, pesos e saídas de modelos de aprendizado de máquina. Em frameworks como **TensorFlow** e **PyTorch**, os tensores permitem realizar operações matemáticas em larga escala e de maneira otimizada, especialmente em GPUs, além de possibilitar o cálculo automático de gradientes, essencial para o treinamento de redes neurais.

## Criação e tipos de tensores

- Scalar: um único valor numérico (0D).

```python
scalar = torch.tensor(7)
```

- Vector: uma sequência de valores (1D)

```python
vector = torch.tensor([7, 7])
```

- Matrix: uma tabela de valores com linhas e colunas (2D)

```python
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
```

- Tensor: uma generalização de matrizes para mais dimensões (3D ou superior)

```python
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
```

### Criação de Tensor aleatório

```python
# Cria um tensor de tamanho (3, 4) com valores aleatórios
random_tensor = torch.rand(3, 4)
```

### Criação de um Tensor cheio de zeros

```python
zeros = torch.zeros(size=(3, 4))
```

### Criação um tensor a partir de um "range"

```python
one_to_ten = torch.arange(start=1, end=11, step=1)
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

### Criação de um "tensor-like" cheio de zeros

```python
ten_zeroes = torch.zeros_like(one_to_ten)
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

### Extração de informação sobre os Tensores (tensor attributes)

- Número de dimensões de um Tensor: `tensor.ndim`
- Formato de um Tensor: `tensor.shape`
- Tipo de dado um Tensor: `tensor.dtype`
- Dispositivo que o Tensor está: `tensor.device`

## Manipulação de Tensores (tensor operations)

### Adição

```python
# Cria um tensor e adiciona 10 de todos seus elementos
tensor = torch.tensor([1, 2, 3])
tensor + 10
# tensor([11, 12, 13])
```

### Subtração

```python
# Subtrai 10 de todos seus elementos
tensor - 10
# tensor([-9, -8, -7])
```

### Multiplicação (element-wise)

```python
# Multiplica todos seus elementos por 10
tensor * 10
```

### Divisão

```python
# Divide todos os elementos por 10
tensor / 10
```

### Multiplicação de matrizes

Existem duas regras principais para executar uma multiplicação de matrizes

1. As dimensões de dentro devem ser iguais:

- `(3, 2) @ (3, 2)` não funciona
- `(2, 3) @ (3, 2)` funciona pois as dimensões de dentro são iguais

2. A matrix resultante vai ter o formato das dimensões mais para fora

- `(2, 3) @ (3, 2)` -> `(2, 2)`
- `(3, 2) @ (2, 3)` -> `(3, 3)`

```python
# Executando multiplicação de matrizes
torch.matmul(tensor, tensor)
```

### Transposição de matrizes

A **transposição** troca as **linhas pelas colunas** de uma matriz, alterando sua orientação.
Ela é útil, por exemplo, quando precisamos ajustar as dimensões para que duas matrizes possam ser multiplicadas.

```python
# Transpondo uma matriz
tensor_t = tensor.T  # ou tensor.transpose(0, 1)
```

Exemplo:

- `tensor.shape = (2, 3)`
- `tensor_t.shape = (3, 2)`

Com isso, agora é possível fazer:

```python
torch.matmul(tensor, tensor_t)  # (2, 3) @ (3, 2) -> (2, 2)
```

### Agregação de tensores (tensor aggregation)

A **agregação de tensores** combina valores em um único resultado aplicando funções como **mínimo**, **máximo**, **média** ou **soma** sobre uma ou mais dimensões.

```python
torch.min(tensor)   # menor valor
torch.max(tensor)   # maior valor
torch.mean(tensor)  # média dos valores
torch.sum(tensor)   # soma de todos os valores
```

### Encontrando a posição do mínimo e do máximo

```python
tensor.argmin()
tensor.argmax()
```

### Reformatação, empilhamento e ajuste de dimensões de tensores

- **Reshape** – redefine o formato de um tensor para uma nova forma especificada.
- **View** – cria uma _view_ (visão) do tensor com outra forma, **sem copiar os dados** na memória.
- **Stack** – empilha múltiplos tensores, seja **um sobre o outro** (verticalmente) ou **lado a lado** (horizontalmente).
- **Squeeze** – remove todas as dimensões com tamanho `1` de um tensor.
- **Unsqueeze** – adiciona uma dimensão de tamanho `1` em uma posição específica.
- **Permute** – retorna uma _view_ do tensor com as dimensões **trocadas** em uma nova ordem.

```python
# Exemplos
torch.reshape(tensor, (2, 3))
tensor.view(2, 3)
torch.stack([t1, t2], dim=0)
tensor.squeeze()
tensor.unsqueeze(0)
tensor.permute(1, 0)
```

## Reprodutibilidade (controlando a aleatoriedade)

As **seeds** (sementes aleatórias) servem para **tornar os resultados reproduzíveis** em experimentos de _machine learning_.
Ao definir uma seed, você garante que as operações aleatórias — como inicialização de pesos, embaralhamento de dados ou geração de amostras — produzam sempre os mesmos resultados em execuções diferentes.

```python
# Definindo uma seed em PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

Isso é essencial para **comparar modelos**, **depurar código** e **validar experimentos** de forma consistente.

### Executando tensores e modelos na GPU

Para acelerar cálculos, especialmente em redes neurais grandes, é comum **usar GPUs**. Em PyTorch, isso significa **mover tensores e modelos para a GPU** antes de realizar operações.

```python
import torch

# Verifica se GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mover tensor para GPU
tensor = tensor.to(device)

# Mover modelo para GPU
model = model.to(device)

# Agora todas as operações com o tensor ou modelo serão feitas na GPU
```

Usar GPU **reduz drasticamente o tempo de treinamento**, pois GPUs são otimizadas para operações matriciais e paralelização em larga escala.

