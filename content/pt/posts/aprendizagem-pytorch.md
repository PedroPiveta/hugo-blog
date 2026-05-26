---
title: "Aprendizagem PyTorch"
date: 2025-01-11T21:34:35-03:00
tags: ["programação", "pytorch", "machine learning", "deep learning", "python"]
author: "Pedro Piveta Barrotti"
translationKey: pytorch
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Anotações de estudo sobre PyTorch e Machine Learning - Um guia de referência completo"
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
  alt: "Logo do PyTorch"
  caption: ""
  relative: false
  hidden: false
editPost:
  URL: "https://github.com/PedroPiveta/hugo-blog/tree/master/content/pt"
  Text: "Sugerir alterações"
  appendFilePath: true
---

# Intuito do post

Servir como um guia de referência para PyTorch e Machine Learning.

Pretendo atualizar esse post conforme eu for evoluindo no PyTorch.

# O que é o PyTorch?

PyTorch é uma biblioteca de machine learning desenvolvida pelo Facebook AI Research (FAIR). Ela é amplamente utilizada para tarefas de aprendizado profundo (deep learning) e oferece uma interface flexível e dinâmica para a construção e treinamento de redes neurais.

## O que são Tensores?

Os **tensores** são estruturas de dados fundamentais em _machine learning_ que generalizam números, vetores e matrizes para múltiplas dimensões. Eles servem para representar e processar dados numéricos de forma eficiente, sendo usados para armazenar entradas, pesos e saídas de modelos de aprendizado de máquina. Em frameworks como **TensorFlow** e **PyTorch**, os tensores permitem realizar operações matemáticas em larga escala e de maneira otimizada, especialmente em GPUs, além de possibilitar o cálculo automático de gradientes, essencial para o treinamento de redes neurais.

### Tipos de Tensores

**Matrix:** uma tabela de valores com linhas e colunas (2D)

```python
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
```

**Tensor:** uma generalização de matrizes para mais dimensões (3D ou superior)

```python
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
```

### Criação de Tensores

**Tensor aleatório:**

```python
# Cria um tensor de tamanho (3, 4) com valores aleatórios
random_tensor = torch.rand(3, 4)
```

**Tensor cheio de zeros:**

```python
zeros = torch.zeros(size=(3, 4))
```

**Tensor a partir de um range:**

```python
one_to_ten = torch.arange(start=1, end=11, step=1)
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

**Tensor-like cheio de zeros:**

```python
ten_zeroes = torch.zeros_like(one_to_ten)
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

### Atributos dos Tensores

Extraindo informação sobre os Tensores (tensor attributes):

- Número de dimensões de um Tensor: `tensor.ndim`
- Formato de um Tensor: `tensor.shape`
- Tipo de dado um Tensor: `tensor.dtype`
- Dispositivo que o Tensor está: `tensor.device`

## Manipulação de Tensores

### Operações Básicas

**Adição:**

```python
# Cria um tensor e adiciona 10 de todos seus elementos
tensor = torch.tensor([1, 2, 3])
tensor + 10
# tensor([11, 12, 13])
```

**Subtração:**

```python
# Subtrai 10 de todos seus elementos
tensor - 10
# tensor([-9, -8, -7])
```

**Multiplicação (element-wise):**

```python
# Multiplica todos seus elementos por 10
tensor * 10
```

**Divisão:**

```python
# Divide todos os elementos por 10
tensor / 10
```

### Multiplicação de Matrizes

Existem duas regras principais para executar uma multiplicação de matrizes:

1. **As dimensões de dentro devem ser iguais:**
   - `(3, 2) @ (3, 2)` não funciona
   - `(2, 3) @ (3, 2)` funciona pois as dimensões de dentro são iguais

2. **A matriz resultante terá o formato das dimensões mais externas:**
   - `(2, 3) @ (3, 2)` → `(2, 2)`
   - `(3, 2) @ (2, 3)` → `(3, 3)`

```python
# Executando multiplicação de matrizes
torch.matmul(tensor, tensor)
```

### Transposição de Matrizes

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

### Agregação de Tensores

A **agregação de tensores** combina valores em um único resultado aplicando funções como **mínimo**, **máximo**, **média** ou **soma** sobre uma ou mais dimensões.

```python
torch.min(tensor)   # menor valor
torch.max(tensor)   # maior valor
torch.mean(tensor)  # média dos valores
torch.sum(tensor)   # soma de todos os valores
```

### Encontrando Posições

Encontrando a posição do mínimo e do máximo:

```python
tensor.argmin()  # índice do valor mínimo
tensor.argmax()  # índice do valor máximo
```

### Reformatação e Ajuste de Dimensões

Operações para reformatar, empilhar e ajustar dimensões de tensores:

- **Reshape** – redefine o formato de um tensor para uma nova forma especificada
- **View** – cria uma _view_ (visão) do tensor com outra forma, **sem copiar os dados** na memória
- **Stack** – empilha múltiplos tensores, seja **um sobre o outro** (verticalmente) ou **lado a lado** (horizontalmente)
- **Squeeze** – remove todas as dimensões com tamanho `1` de um tensor
- **Unsqueeze** – adiciona uma dimensão de tamanho `1` em uma posição específica
- **Permute** – retorna uma _view_ do tensor com as dimensões **trocadas** em uma nova ordem

```python
# Exemplos serão adicionados conforme o aprendizado progride
```

---

## Workflow PyTorch

O fluxo de trabalho completo no PyTorch envolve as seguintes etapas:

1. Dados (preparar e carregar)
2. Construir o modelo
3. Ajustar o modelo aos dados (treinamento)
4. Fazer predições (inferência)
5. Salvar e carregar o modelo

### 1. Dados (preparando e carregando)

Em machine learning, dados podem ser quase qualquer coisa: planilhas, imagens, vídeos, áudio, DNA, texto, etc.

Machine learning é um jogo de duas partes:

1. Transformar dados em representação numérica
2. Construir um modelo para aprender padrões nessa representação numérica

Para demonstrar isso, vamos criar dados conhecidos na forma de regressão linear, usando parâmetros **conhecidos**.

```python
import torch
from torch import nn
import matplotlib.pyplot as plt

# Parâmetros conhecidos
weight = 0.7
bias = 0.3

# Criando os dados
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias  # fórmula de regressão linear: y = peso * X + viés
```

#### Divisão em conjuntos de treino e teste

```python
# Divisão treino/teste (80% treino, 20% teste)
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
    Plota dados de treino, dados de teste e compara com as predições.
    """
    plt.figure(figsize=(10, 7))

    # Dados de treino em azul
    plt.scatter(train_data, train_labels, c="b", s=4, label="Dados de treino")

    # Dados de teste em verde
    plt.scatter(test_data, test_labels, c="g", s=4, label="Dados de teste")

    # Predições (se existirem)
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predições")

    plt.legend(prop={"size": 14})
```

### 2. Construindo o modelo

```python
from torch import nn

# Criando a classe de modelo de regressão linear
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parâmetros que o modelo vai aprender (inicializados aleatoriamente)
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    # Método forward define o cálculo do modelo
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
```

#### Componentes essenciais do PyTorch para construção de modelos

- `torch.nn` — contém todos os blocos de construção para grafos computacionais
- `torch.nn.Parameter` — define quais parâmetros o modelo deve aprender
- `torch.nn.Module` — classe base para todos os modelos de redes neurais; é necessário sobrescrever o método `forward`
- `torch.optim` — onde vivem os otimizadores do PyTorch
- `def forward()` — todas as subclasses de `nn.Module` precisam sobrescrever este método, que define a computação do passo à frente

#### Verificando o conteúdo do modelo

```python
# Criando uma semente aleatória para reprodutibilidade
torch.manual_seed(42)

# Instanciando o modelo
model_0 = LinearRegressionModel()

# Listando os parâmetros do modelo
list(model_0.parameters())

# Verificando o dicionário de estado (pesos e vieses atuais)
model_0.state_dict()
```

#### Fazendo predições com `torch.inference_mode()`

```python
# Fazendo predições com o modelo (sem rastrear gradientes)
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)
```

### 3. Treinando o modelo

O objetivo do treinamento é mover o modelo de parâmetros _desconhecidos_ para parâmetros _conhecidos_ (próximos dos reais).

Para treinar, precisamos de:

- **Função de perda (loss function):** mede o quão erradas são as predições do modelo em relação às saídas ideais. Quanto menor, melhor.
- **Otimizador:** usa a perda para ajustar os parâmetros do modelo e melhorar as predições.
- Um **loop de treino**
- Um **loop de teste**

```python
# Configurando a função de perda (MAE - Mean Absolute Error)
loss_fn = nn.L1Loss()

# Configurando o otimizador (SGD - Descida do Gradiente Estocástico)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)  # lr = taxa de aprendizado (learning rate)
```

#### Loop de treinamento

As etapas do loop de treinamento:

0. Iterar pelos dados (loop de épocas)
1. **Passo à frente (forward pass):** os dados passam pelo método `forward()` do modelo
2. **Calcular a perda:** comparar as predições com os rótulos reais
3. **Zerar os gradientes do otimizador**
4. **Retropropagação (backpropagation):** calcular os gradientes de cada parâmetro em relação à perda
5. **Passo do otimizador (gradient descent):** ajustar os parâmetros para reduzir a perda

```python
torch.manual_seed(42)

epochs = 200  # uma época = uma passagem completa pelos dados

# Listas para acompanhar os valores de perda ao longo do treino
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    # Modo de treino: ativa o rastreamento de gradientes
    model_0.train()

    # 1. Passo à frente
    y_pred = model_0(X_train)

    # 2. Calcular a perda
    loss = loss_fn(y_pred, y_train)

    # 3. Zerar os gradientes acumulados do passo anterior
    optimizer.zero_grad()

    # 4. Retropropagação
    loss.backward()

    # 5. Atualizar os parâmetros
    optimizer.step()

    #### Avaliação no conjunto de teste
    model_0.eval()  # desativa o rastreamento de gradientes
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Época: {epoch} | Perda: {loss} | Perda de teste: {test_loss}")
        print(model_0.state_dict())
```

```python
import numpy as np

# Plotando as curvas de perda de treino e teste
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Perda de treino")
plt.plot(epoch_count, test_loss_values, label="Perda de teste")
plt.title("Curvas de perda de treino e teste")
plt.ylabel("Perda")
plt.xlabel("Épocas")
plt.legend()
```

### 4. Código agnóstico de dispositivo

Para aproveitar uma GPU quando disponível, usa-se código agnóstico de dispositivo:

```python
# Verifica se há GPU disponível; caso contrário, usa CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
```

Na versão aprimorada do modelo, usa-se `nn.Linear` em vez de definir os parâmetros manualmente:

```python
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear cria automaticamente os parâmetros de peso e viés
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()

# Enviando o modelo para o dispositivo disponível (GPU ou CPU)
model_1.to(device)

# Os dados também precisam ir para o mesmo dispositivo
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
```

### 5. Salvando e carregando um modelo

Há três métodos principais para salvar e carregar modelos no PyTorch:

1. `torch.save()` — salva um objeto PyTorch no formato pickle do Python
2. `torch.load()` — carrega um objeto PyTorch salvo
3. `torch.nn.Module.load_state_dict()` — carrega o dicionário de estado salvo de um modelo

#### Salvando o modelo

```python
from pathlib import Path

# 1. Criar diretório para os modelos
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Definir o caminho de salvamento
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Salvar apenas o state_dict (parâmetros), não o modelo inteiro
print(f"Salvando modelo em: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)
```

#### Carregando o modelo

```python
# Para carregar, é preciso instanciar o modelo novamente
loaded_model_0 = LinearRegressionModel()

# Carregar o state_dict salvo na nova instância
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Avaliar o modelo carregado
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# Verificar se as predições são iguais às do modelo original
y_preds == loaded_model_preds
```

---

_Este post será atualizado continuamente com novos conceitos e exemplos práticos._

