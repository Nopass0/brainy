# 🧠 Brainy

**Быстрый AI/ML фреймворк для Bun (TypeScript) с поддержкой GPU/CPU**

Brainy — это полнофункциональный фреймворк для глубокого обучения, вдохновлённый PyTorch, написанный на чистом TypeScript для Bun runtime.

## ✨ Возможности

- **Тензорные операции** — многомерные массивы с broadcasting
- **Автоматическое дифференцирование** — полноценный autograd
- **Нейросетевые слои** — Linear, Conv2d, LSTM, Embedding, BatchNorm, LayerNorm, Dropout
- **Активации** — ReLU, GELU, Sigmoid, Softmax, Tanh, SiLU, и другие
- **Функции потерь** — MSE, CrossEntropy, BCE, NLL, Hinge, KLDiv
- **Оптимизаторы** — SGD, Adam, AdamW, RMSprop, Adagrad
- **LR Schedulers** — StepLR, CosineAnnealing, ReduceLROnPlateau
- **Data utilities** — Dataset, DataLoader, train/test split
- **Сериализация** — сохранение и загрузка моделей

## 📦 Установка

```bash
# Клонируйте репозиторий
git clone <repo-url>
cd brainy

# Установите зависимости
bun install
```

## 🚀 Быстрый старт

```typescript
import { tensor, Linear, MSELoss, SGD } from './src';

// Создание тензора
const x = tensor([[1, 2, 3], [4, 5, 6]]);
console.log(x.shape); // [2, 3]

// Математические операции
const y = x.mul(2).add(1);
console.log(y.toArray()); // [[3, 5, 7], [9, 11, 13]]

// Нейронная сеть
const model = new Linear(10, 5);
const criterion = new MSELoss();
const optimizer = new SGD(model.parameters(), 0.01);

// Обучение
const input = tensor([...Array(10)].map(() => Math.random()));
const target = tensor([...Array(5)].map(() => Math.random()));

for (let i = 0; i < 100; i++) {
  const output = model.forward(input);
  const loss = criterion.forward(output, target);
  
  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();
}
```

## 📚 Примеры

```bash
# Базовые операции с тензорами
bun run examples/01-basic-tensors.ts

# Автоматическое дифференцирование
bun run examples/02-autograd.ts

# Линейная регрессия
bun run examples/03-linear-regression.ts

# XOR нейросеть
bun run examples/04-xor-neural-network.ts

# MNIST классификация (CNN)
bun run examples/05-mnist-classification.ts

# Кастомные слои (Attention, ResNet)
bun run examples/06-custom-layer.ts
```

## 🏗️ API Reference

### Тензоры

```typescript
// Создание
tensor([[1, 2], [3, 4]])      // Из массива
zeros([2, 3])                  // Нули
ones([2, 3])                   // Единицы
rand([2, 3])                   // Случайные [0, 1)
randn([2, 3])                  // Нормальное распределение
eye(3)                         // Единичная матрица
linspace(0, 10, 5)             // Линейное распределение
arange(0, 10, 2)               // Диапазон

// Операции
t.add(other)                   // Сложение
t.sub(other)                   // Вычитание
t.mul(other)                   // Умножение (поэлементное)
t.div(other)                   // Деление
t.matmul(other)                // Матричное умножение
t.pow(2)                       // Степень
t.exp()                        // Экспонента
t.log()                        // Логарифм

// Reshape
t.reshape(3, 4)                // Изменение формы
t.flatten()                    // Выравнивание
t.transpose(0, 1)              // Транспонирование
t.squeeze()                    // Удаление размерностей=1
t.unsqueeze(0)                 // Добавление размерности

// Редукции
t.sum()                        // Сумма
t.mean()                       // Среднее
t.max()                        // Максимум
t.min()                        // Минимум
t.argmax()                     // Индекс максимума
```

### Нейронные сети

```typescript
import { Sequential, Linear, ReLU, Conv2d, Dropout } from './src';

const model = new Sequential(
  new Linear(784, 256),
  new ReLU(),
  new Dropout(0.5),
  new Linear(256, 10)
);

const output = model.forward(input);
```

### Обучение

```typescript
import { CrossEntropyLoss, Adam } from './src';

const criterion = new CrossEntropyLoss();
const optimizer = new Adam(model.parameters(), 0.001);

for (const batch of dataLoader) {
  const output = model.forward(batch.input);
  const loss = criterion.forward(output, batch.target);
  
  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();
}
```

### Кастомные слои

```typescript
import { Module, Parameter, Linear, Tensor } from './src';

class MyLayer extends Module {
  weight: Parameter;
  linear: Linear;

  constructor(inFeatures: number, outFeatures: number) {
    super();
    this.weight = new Parameter(randn([inFeatures]), 'weight');
    this.linear = new Linear(inFeatures, outFeatures);
    
    this.registerParameter('weight', this.weight);
    this.registerModule('linear', this.linear);
  }

  forward(x: Tensor): Tensor {
    return this.linear.forward(x.mul(this.weight.data));
  }
}
```

## 📂 Структура проекта

```
brainy/
├── src/
│   ├── core/           # Тензоры и autograd
│   │   ├── tensor.ts
│   │   ├── autograd.ts
│   │   ├── dtype.ts
│   │   └── shape.ts
│   ├── nn/             # Нейросетевые модули
│   │   ├── module.ts
│   │   ├── layers.ts
│   │   ├── activations.ts
│   │   ├── loss.ts
│   │   └── init.ts
│   ├── optim/          # Оптимизаторы
│   │   └── optimizer.ts
│   ├── data/           # Data utilities
│   │   └── dataloader.ts
│   ├── functional/     # Функциональный API
│   │   └── functional.ts
│   ├── utils/          # Утилиты
│   │   ├── serialize.ts
│   │   └── random.ts
│   └── index.ts        # Главный экспорт
├── examples/           # Примеры использования
└── package.json
```

## 🔧 Сравнение с PyTorch

| PyTorch | Brainy |
|---------|--------|
| `torch.tensor([1, 2, 3])` | `tensor([1, 2, 3])` |
| `torch.zeros(2, 3)` | `zeros([2, 3])` |
| `x @ y` | `x.matmul(y)` |
| `nn.Linear(10, 5)` | `new Linear(10, 5)` |
| `nn.Sequential(...)` | `new Sequential(...)` |
| `optim.Adam(...)` | `new Adam(...)` |
| `loss.backward()` | `loss.backward()` |

## 📄 Лицензия

MIT License
