# Brainy ML

> Быстрый AI/ML фреймворк для Bun/Node.js с поддержкой GPU/CPU, автодифференцированием и PyTorch-подобным API

[![npm version](https://img.shields.io/npm/v/brainy-ml.svg)](https://www.npmjs.com/package/brainy-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)

[English Documentation](README_EN.md) | [Документация на сайте](https://nopass0.github.io/brainy)

## Возможности

- **PyTorch-подобный API** — знакомый интерфейс для пользователей PyTorch
- **Автоматическое дифференцирование** — полный autograd с backward() для любого вычислительного графа
- **Поддержка GPU** — WebGPU ускорение для максимальной производительности
- **Готовые модели** — GPT, Transformers, VAE, TRM и другие
- **Интеграция с Hugging Face** — загрузка моделей напрямую из HF Hub
- **Типобезопасность** — полная поддержка TypeScript с исчерпывающими определениями типов
- **Лёгкий вес** — без тяжёлых зависимостей, работает в Bun и Node.js

## Установка

```bash
# Используя Bun (рекомендуется)
bun add brainy-ml

# Используя npm
npm install brainy-ml

# Используя yarn
yarn add brainy-ml
```

## Быстрый старт

### Базовые операции с тензорами

```typescript
import { tensor, zeros, ones, randn } from 'brainy-ml';

// Создание тензоров
const a = tensor([[1, 2], [3, 4]]);
const b = zeros([2, 2]);
const c = randn([3, 3]);

// Операции
const sum = a.add(b);
const product = a.matmul(a.T);  // Матричное умножение
const mean = c.mean();

console.log('Форма:', sum.shape);      // [2, 2]
console.log('Произведение:', product.toArray());
console.log('Среднее:', mean.item());
```

### Автоматическое дифференцирование

```typescript
import { tensor } from 'brainy-ml';

// Создаём тензор с отслеживанием градиентов
const x = tensor([[2.0]], { requiresGrad: true });

// y = x^2 + 3x + 1
const y = x.pow(2).add(x.mul(3)).add(1);

// Вычисляем градиенты
y.backward();

// dy/dx = 2x + 3 = 2*2 + 3 = 7
console.log('x:', x.item());           // 2
console.log('y:', y.item());           // 11
console.log('dy/dx:', x.grad.item());  // 7
```

### Построение нейронных сетей

```typescript
import {
  Sequential, Linear, ReLU, Sigmoid,
  MSELoss, Adam, tensor
} from 'brainy-ml';

// Создаём модель
const model = new Sequential(
  new Linear(10, 64),
  new ReLU(),
  new Linear(64, 32),
  new ReLU(),
  new Linear(32, 1),
  new Sigmoid()
);

// Настраиваем обучение
const optimizer = new Adam(model.parameters(), 0.001);
const criterion = new MSELoss();

// Цикл обучения
for (let epoch = 0; epoch < 100; epoch++) {
  const x = tensor([[/* ваши данные */]]);
  const y = tensor([[/* ваши метки */]]);

  const pred = model.forward(x);
  const loss = criterion.forward(pred, y);

  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();

  if (epoch % 10 === 0) {
    console.log(`Эпоха ${epoch}, Loss: ${loss.item()}`);
  }
}
```

### Пример: задача XOR

```typescript
import {
  Sequential, Linear, ReLU, Sigmoid,
  MSELoss, Adam, tensor
} from 'brainy-ml';

// Данные XOR
const X = tensor([[0,0], [0,1], [1,0], [1,1]]);
const Y = tensor([[0], [1], [1], [0]]);

// Простая сеть
const model = new Sequential(
  new Linear(2, 8),
  new ReLU(),
  new Linear(8, 1),
  new Sigmoid()
);

const optimizer = new Adam(model.parameters(), 0.1);
const criterion = new MSELoss();

// Обучение
for (let i = 0; i < 1000; i++) {
  const pred = model.forward(X);
  const loss = criterion.forward(pred, Y);

  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();
}

// Тест
console.log('Предсказания:', model.forward(X).toArray());
// Ожидается: [[~0], [~1], [~1], [~0]]
```

### Загрузка моделей из Hugging Face

```typescript
import { HuggingFaceHub } from 'brainy-ml';

const hub = new HuggingFaceHub();

// Получаем информацию о модели
const info = await hub.getModelInfo('Qwen/Qwen2.5-0.5B');
console.log('Модель:', info.id);

// Скачиваем конфигурацию
const config = await hub.downloadConfig('Qwen/Qwen2.5-0.5B');
console.log('Hidden size:', config.hidden_size);

// Скачиваем веса (формат safetensors)
const weights = await hub.downloadWeights('Qwen/Qwen2.5-0.5B');
console.log('Загружено', weights.size, 'тензоров');
```

## Справочник API

### Основное

| Функция | Описание |
|---------|----------|
| `tensor(data, options?)` | Создать тензор из массива |
| `zeros(shape, options?)` | Тензор из нулей |
| `ones(shape, options?)` | Тензор из единиц |
| `rand(shape, options?)` | Равномерное распределение [0, 1) |
| `randn(shape, mean?, std?, options?)` | Нормальное распределение |
| `eye(n, options?)` | Единичная матрица |
| `linspace(start, end, steps, options?)` | Равномерно распределённые значения |
| `arange(start, end?, step?, options?)` | Диапазон значений |

### Операции с тензорами

| Метод | Описание |
|-------|----------|
| `.add(other)` | Поэлементное сложение |
| `.sub(other)` | Поэлементное вычитание |
| `.mul(other)` | Поэлементное умножение |
| `.div(other)` | Поэлементное деление |
| `.matmul(other)` | Матричное умножение |
| `.pow(exp)` | Возведение в степень |
| `.sqrt()` | Квадратный корень |
| `.exp()` | Экспонента |
| `.log()` | Натуральный логарифм |
| `.sum(dim?, keepdim?)` | Сумма |
| `.mean(dim?, keepdim?)` | Среднее |
| `.max(dim?, keepdim?)` | Максимум |
| `.min(dim?, keepdim?)` | Минимум |
| `.reshape(...shape)` | Изменить форму |
| `.transpose(dim0, dim1)` | Поменять измерения местами |
| `.T` | Сокращение для transpose(0, 1) |
| `.backward()` | Вычислить градиенты |

### Слои нейронных сетей

| Класс | Описание |
|-------|----------|
| `Linear(in, out, bias?)` | Полносвязный слой |
| `Conv2d(in, out, kernel, stride?, padding?)` | 2D свёртка |
| `MaxPool2d(kernel, stride?, padding?)` | Макс пулинг |
| `AvgPool2d(kernel, stride?, padding?)` | Средний пулинг |
| `BatchNorm1d(features)` | Batch нормализация 1D |
| `BatchNorm2d(features)` | Batch нормализация 2D |
| `LayerNorm(shape)` | Layer нормализация |
| `Dropout(p)` | Dropout регуляризация |
| `Embedding(num, dim)` | Слой Embedding |
| `Flatten(startDim?, endDim?)` | Выравнивание |

### Активации

| Класс | Описание |
|-------|----------|
| `ReLU()` | Rectified Linear Unit |
| `LeakyReLU(slope?)` | Leaky ReLU |
| `GELU()` | Gaussian Error Linear Unit |
| `SiLU()` / `Swish()` | Sigmoid Linear Unit |
| `Sigmoid()` | Сигмоида |
| `Tanh()` | Гиперболический тангенс |
| `Softmax(dim)` | Softmax |
| `LogSoftmax(dim)` | Log softmax |

### Функции потерь

| Класс | Описание |
|-------|----------|
| `MSELoss()` | Среднеквадратичная ошибка |
| `L1Loss()` | Средняя абсолютная ошибка |
| `CrossEntropyLoss()` | Кросс-энтропия для классификации |
| `BCELoss()` | Бинарная кросс-энтропия |
| `BCEWithLogitsLoss()` | BCE с сигмоидой |
| `NLLLoss()` | Negative log likelihood |
| `HingeLoss()` | Hinge loss для SVM |
| `KLDivLoss()` | KL дивергенция |

### Оптимизаторы

| Класс | Описание |
|-------|----------|
| `SGD(params, lr, momentum?)` | Стохастический градиентный спуск |
| `Adam(params, lr, betas?, eps?)` | Adam оптимизатор |
| `AdamW(params, lr, betas?, weightDecay?)` | Adam с weight decay |
| `RMSprop(params, lr, alpha?)` | RMSprop оптимизатор |
| `Adagrad(params, lr)` | Adagrad оптимизатор |

### Планировщики скорости обучения

| Класс | Описание |
|-------|----------|
| `StepLR(optimizer, stepSize, gamma?)` | Ступенчатое уменьшение |
| `ExponentialLR(optimizer, gamma)` | Экспоненциальное уменьшение |
| `CosineAnnealingLR(optimizer, tMax)` | Косинусный отжиг |
| `ReduceLROnPlateau(optimizer, mode?, factor?)` | Уменьшение на плато |

### Готовые модели

| Класс | Описание |
|-------|----------|
| `GPT(config)` | GPT-style трансформер |
| `VAE(config)` | Вариационный автокодировщик |
| `ConvVAE(config)` | Свёрточный VAE |
| `TRM*(config)` | Варианты модели TRM |
| `TransformerEncoder` | Encoder трансформера |
| `TransformerDecoder` | Decoder трансформера |

## Поддержка GPU

Brainy ML поддерживает WebGPU для GPU ускорения:

```typescript
import { isWebGPUSupported, createDevice } from 'brainy-ml';

if (await isWebGPUSupported()) {
  const device = await createDevice('gpu');
  console.log('GPU включен!');
}
```

## Примеры

Смотрите папку `examples/` для больших примеров:

- `01-basic-tensors.ts` - Операции с тензорами
- `02-autograd.ts` - Автодифференцирование
- `03-linear-regression.ts` - Линейная регрессия
- `04-xor-neural-network.ts` - Задача XOR
- `05-mnist-classification.ts` - Классификация MNIST
- `07-text-generation.ts` - Генерация текста с GPT
- `26-huggingface-example.ts` - Интеграция с Hugging Face
- `27-qwen3-example.ts` - Загрузка модели Qwen

## Вклад в проект

Вклады приветствуются! Пожалуйста, создавайте Pull Request.

## Лицензия

MIT License - см. файл [LICENSE](LICENSE) для деталей.

## Ссылки

- [Документация](https://nopass0.github.io/brainy)
- [GitHub репозиторий](https://github.com/Nopass0/brainy)
- [npm пакет](https://www.npmjs.com/package/brainy-ml)
