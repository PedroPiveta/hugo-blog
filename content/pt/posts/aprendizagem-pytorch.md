---
title: "Aprendizagem Pytorch"
date: "2025-11-11T21:34:35-03:00"
tags: ["programação", "pytorch", "machine learning"]
author: "Pedro Piveta Barrotti"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Anotação de estudo sobre Pytorch"
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
    alt: ""
    caption: ""
    relative: false
    hidden: false
editPost:
    URL: ""
    Text: "Suggest Changes"
    appendFilePath: true
---

# Intuito do post:

Servir como um guia de referência para PyTorch e Machine Learning.

Pretendo atualizar esse post conforme eu for evoluindo no PyTorch.

# O que é o PyTorch?

Pytorch é uma biblioteca de machine learning desenvolvida pelo Facebook AI Research (FAIR). Ela é amplamente utilizada para tarefas de aprendizado profundo (deep learning) e oferece uma interface flexível e dinâmica para a construção e treinamento de redes neurais.

<details>
<summary>
Tensores
</summary>

## O que são Tensores?

Os **tensores** são estruturas de dados fundamentais em _machine learning_ que generalizam números, vetores e matrizes para múltiplas dimensões. Eles servem para representar e processar dados numéricos de forma eficiente, sendo usados para armazenar entradas, pesos e saídas de modelos de aprendizado de máquina. Em frameworks como **TensorFlow** e **PyTorch**, os tensores permitem realizar operações matemáticas em larga escala e de maneira otimizada, especialmente em GPUs, além de possibilitar o cálculo automático de gradientes, essencial para o treinamento de redes neurais.

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
```
