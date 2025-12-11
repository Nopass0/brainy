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
- **Reinforcement Learning** — DQN, Policy Gradient, Actor-Critic агенты
- **Квантизация и Fine-tuning** — LoRA, Adapters, QAT
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
import { tensor, zeros, ones, randn, eye, linspace, arange } from 'brainy-ml';

// Создание тензоров
const a = tensor([[1, 2], [3, 4]]);
const b = zeros([2, 2]);
const c = randn([3, 3]);
const identity = eye(3);
const range = linspace(0, 10, 5);  // [0, 2.5, 5, 7.5, 10]

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
import { tensor, noGrad } from 'brainy-ml';

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

// Отключение градиентов для инференса
noGrad(() => {
  const result = x.mul(2);  // Без отслеживания градиентов
});
```

### Построение нейронных сетей

```typescript
import {
  Sequential, Linear, ReLU, Sigmoid, Dropout, BatchNorm1d,
  MSELoss, Adam, tensor
} from 'brainy-ml';

// Создаём модель
const model = new Sequential(
  new Linear(10, 64),
  new BatchNorm1d(64),
  new ReLU(),
  new Dropout(0.2),
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

// Переключение режимов train/eval
model.train();  // Включает Dropout, BatchNorm в режиме обучения
model.eval();   // Отключает Dropout, BatchNorm в режиме инференса
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

### Трансформеры

```typescript
import {
  TransformerEncoder, TransformerDecoder,
  MultiHeadAttention, PositionalEncoding, RotaryEmbedding,
  createCausalMask, tensor
} from 'brainy-ml';

// Multi-Head Attention
const mha = new MultiHeadAttention(512, 8);  // d_model=512, heads=8
const query = tensor(/* [batch, seq, 512] */);
const key = tensor(/* [batch, seq, 512] */);
const value = tensor(/* [batch, seq, 512] */);
const output = mha.forward(query, key, value);

// Позиционное кодирование
const posEnc = new PositionalEncoding(512, 1000);  // d_model, max_len
const withPos = posEnc.forward(tensor(/* [batch, seq, 512] */));

// Rotary Embeddings (RoPE)
const rope = new RotaryEmbedding(64);  // head_dim
const rotated = rope.forward(query, 0);  // с offset позиции

// Causal mask для авторегрессии
const mask = createCausalMask(100);  // [100, 100] маска

// Полный Transformer Encoder
const encoder = new TransformerEncoder({
  d_model: 512,
  nhead: 8,
  num_layers: 6,
  dim_feedforward: 2048,
  dropout: 0.1
});

const encoded = encoder.forward(tensor(/* input */));
```

### Загрузка моделей из Hugging Face

```typescript
import { HuggingFaceHub, loadWeightsIntoModel } from 'brainy-ml';

const hub = new HuggingFaceHub();

// Получаем информацию о модели
const info = await hub.getModelInfo('Qwen/Qwen2.5-0.5B');
console.log('Модель:', info.id);

// Список файлов в репозитории
const files = await hub.listFiles('Qwen/Qwen2.5-0.5B');
console.log('Файлы:', files);

// Скачиваем конфигурацию
const config = await hub.downloadConfig('Qwen/Qwen2.5-0.5B');
console.log('Hidden size:', config.hidden_size);

// Скачиваем токенизатор
const tokenizer = await hub.downloadTokenizer('Qwen/Qwen2.5-0.5B');

// Скачиваем веса (формат safetensors)
const weights = await hub.downloadWeights('Qwen/Qwen2.5-0.5B');
console.log('Загружено', weights.size, 'тензоров');

// Загрузка весов в модель
const result = loadWeightsIntoModel(model, weights);
console.log('Загружено:', result.loaded);
console.log('Пропущено:', result.missing);
```

### Работа с данными

```typescript
import {
  DataLoader, TensorDataset, ArrayDataset,
  trainTestSplit, StreamingDataset, loadJson, loadCsv
} from 'brainy-ml';

// Создание датасета из тензоров
const X = tensor([[1,2], [3,4], [5,6], [7,8]]);
const Y = tensor([[0], [1], [0], [1]]);
const dataset = new TensorDataset(X, Y);

// DataLoader для батчей
const loader = new DataLoader(dataset, {
  batchSize: 2,
  shuffle: true
});

for (const batch of loader) {
  const [inputs, targets] = batch;
  console.log('Batch shape:', inputs.shape);
}

// Разделение на train/test
const [trainData, testData] = trainTestSplit(dataset, 0.8);

// Загрузка данных из файлов
const jsonData = await loadJson('data.json');
const csvData = await loadCsv('data.csv');

// Streaming для больших данных
const stream = new StreamingDataset('large_data.jsonl', {
  batchSize: 32
});
```

### Токенизация текста

```typescript
import {
  BPETokenizer, WordPieceTokenizer, CharTokenizer,
  loadTokenizer
} from 'brainy-ml';

// BPE Tokenizer
const bpe = new BPETokenizer();
await bpe.train(['Hello world', 'How are you?'], { vocabSize: 1000 });
const encoded = bpe.encode('Hello world');
const decoded = bpe.decode(encoded.ids);

// WordPiece Tokenizer
const wp = new WordPieceTokenizer();
const tokens = wp.encode('Running quickly');

// Загрузка токенизатора из HuggingFace
const tokenizer = await loadTokenizer('bert-base-uncased');
```

### Reinforcement Learning

```typescript
import {
  DQNAgent, PolicyGradientAgent, ActorCriticAgent,
  ReplayBuffer, CartPoleEnv, trainDQN
} from 'brainy-ml';

// DQN Agent
const agent = new DQNAgent({
  stateSize: 4,
  actionSize: 2,
  hiddenSize: 64,
  learningRate: 0.001,
  gamma: 0.99,
  epsilon: 1.0,
  epsilonDecay: 0.995,
  epsilonMin: 0.01
});

// Окружение CartPole
const env = new CartPoleEnv();

// Обучение
for (let episode = 0; episode < 500; episode++) {
  let state = env.reset();
  let totalReward = 0;

  while (true) {
    const action = agent.act(state);
    const { nextState, reward, done } = env.step(action);
    agent.remember(state, action, reward, nextState, done);
    agent.replay(32);  // batch size

    state = nextState;
    totalReward += reward;
    if (done) break;
  }

  console.log(`Episode ${episode}: ${totalReward}`);
}
```

### Fine-tuning с LoRA

```typescript
import {
  createLoRAModel, LoRALayer, AdapterLayer,
  FineTuneTrainer, fineTune
} from 'brainy-ml';

// Создание LoRA модели
const loraModel = createLoRAModel(baseModel, {
  rank: 8,
  alpha: 16,
  targetModules: ['query', 'value']
});

// Fine-tuning
const trainer = new FineTuneTrainer(loraModel, {
  learningRate: 1e-4,
  epochs: 3,
  warmupSteps: 100
});

await trainer.train(trainDataset, valDataset);

// Или быстрый fine-tune
await fineTune(model, dataset, {
  strategy: 'lora',
  rank: 4
});
```

### Квантизация

```typescript
import {
  Quantizer, dynamicQuantize, prepareQAT, convertQAT,
  QuantizationMode, getModelSize
} from 'brainy-ml';

// Динамическая квантизация
const quantizedModel = dynamicQuantize(model, {
  bits: 8,
  mode: QuantizationMode.DYNAMIC
});

// QAT (Quantization-Aware Training)
const qatModel = prepareQAT(model);
// ... обучение ...
const finalModel = convertQAT(qatModel);

// Проверка размера модели
const sizeBefore = getModelSize(model);
const sizeAfter = getModelSize(quantizedModel);
console.log(`Сжатие: ${sizeBefore / sizeAfter}x`);
```

### Визуализация моделей

```typescript
import { visualize, summary, printModel } from 'brainy-ml';

// ASCII визуализация архитектуры
visualize(model);

// Детальная сводка
summary(model, [1, 10]);  // input shape

// Печать параметров
printModel(model);
```

### Сохранение и загрузка

```typescript
import {
  saveModel, loadModel, saveStateDict, loadStateDict,
  saveCheckpoint, loadCheckpoint, exportModel
} from 'brainy-ml';

// Простое сохранение/загрузка
await saveModel(model, 'model.bin');
await loadModel(model, 'model.bin');

// State dict
const state = model.stateDict();
await saveStateDict(state, 'weights.bin');
const loadedState = await loadStateDict('weights.bin');
model.loadStateDict(loadedState);

// Чекпоинты с метаданными
await saveCheckpoint({
  model: model.stateDict(),
  optimizer: optimizer.stateDict(),
  epoch: 10,
  loss: 0.05
}, 'checkpoint.bin');

const checkpoint = await loadCheckpoint('checkpoint.bin');
model.loadStateDict(checkpoint.model);
optimizer.loadStateDict(checkpoint.optimizer);

// Экспорт для инференса
await exportModel(model, 'model_inference.bin', {
  compress: true
});
```

## Справочник API

### Создание тензоров

| Функция | Описание |
|---------|----------|
| `tensor(data, options?)` | Создать тензор из массива |
| `scalar(value, options?)` | Скалярный тензор |
| `zeros(shape, options?)` | Тензор из нулей |
| `ones(shape, options?)` | Тензор из единиц |
| `full(shape, value, options?)` | Тензор заполненный value |
| `rand(shape, options?)` | Равномерное распределение [0, 1) |
| `randn(shape, mean?, std?, options?)` | Нормальное распределение |
| `eye(n, options?)` | Единичная матрица |
| `linspace(start, end, steps, options?)` | Равномерно распределённые значения |
| `arange(start, end?, step?, options?)` | Диапазон значений |
| `cat(tensors, dim?)` | Конкатенация тензоров |
| `stack(tensors, dim?)` | Стек тензоров |

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
| `.abs()` | Абсолютное значение |
| `.neg()` | Отрицание |
| `.sin()`, `.cos()`, `.tan()` | Тригонометрия |
| `.sum(dim?, keepdim?)` | Сумма |
| `.mean(dim?, keepdim?)` | Среднее |
| `.max(dim?, keepdim?)` | Максимум |
| `.min(dim?, keepdim?)` | Минимум |
| `.argmax(dim?)` | Индекс максимума |
| `.argmin(dim?)` | Индекс минимума |
| `.reshape(...shape)` | Изменить форму |
| `.view(...shape)` | Изменить форму (view) |
| `.transpose(dim0, dim1)` | Поменять измерения местами |
| `.permute(...dims)` | Переставить измерения |
| `.squeeze(dim?)` | Удалить размерность 1 |
| `.unsqueeze(dim)` | Добавить размерность |
| `.T` | Сокращение для transpose(0, 1) |
| `.backward()` | Вычислить градиенты |
| `.detach()` | Отсоединить от графа градиентов |
| `.clone()` | Глубокая копия |

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
| `ELU(alpha?)` | Exponential Linear Unit |
| `SELU()` | Scaled ELU |
| `PReLU(numParams?)` | Parametric ReLU |
| `GELU()` | Gaussian Error Linear Unit |
| `SiLU()` / `Swish()` | Sigmoid Linear Unit |
| `Mish()` | Mish activation |
| `Hardswish()` | Hard Swish |
| `Sigmoid()` | Сигмоида |
| `Tanh()` | Гиперболический тангенс |
| `Softmax(dim)` | Softmax |
| `LogSoftmax(dim)` | Log softmax |

### Функции потерь

| Класс | Описание |
|-------|----------|
| `MSELoss()` | Среднеквадратичная ошибка |
| `L1Loss()` | Средняя абсолютная ошибка |
| `SmoothL1Loss(beta?)` | Smooth L1 (Huber Loss) |
| `CrossEntropyLoss()` | Кросс-энтропия для классификации |
| `BCELoss()` | Бинарная кросс-энтропия |
| `BCEWithLogitsLoss()` | BCE с сигмоидой |
| `NLLLoss()` | Negative log likelihood |
| `HingeLoss()` | Hinge loss для SVM |
| `KLDivLoss()` | KL дивергенция |
| `CosineEmbeddingLoss()` | Cosine embedding loss |

### Оптимизаторы

| Класс | Описание |
|-------|----------|
| `SGD(params, lr, momentum?, weightDecay?)` | Стохастический градиентный спуск |
| `Adam(params, lr, betas?, eps?)` | Adam оптимизатор |
| `AdamW(params, lr, betas?, weightDecay?)` | Adam с weight decay |
| `RMSprop(params, lr, alpha?, eps?)` | RMSprop оптимизатор |
| `Adagrad(params, lr, eps?)` | Adagrad оптимизатор |

### Планировщики скорости обучения

| Класс | Описание |
|-------|----------|
| `StepLR(optimizer, stepSize, gamma?)` | Ступенчатое уменьшение |
| `ExponentialLR(optimizer, gamma)` | Экспоненциальное уменьшение |
| `CosineAnnealingLR(optimizer, tMax, etaMin?)` | Косинусный отжиг |
| `ReduceLROnPlateau(optimizer, mode?, factor?, patience?)` | Уменьшение на плато |

### Инициализация весов

| Функция | Описание |
|---------|----------|
| `xavierUniform(tensor, gain?)` | Xavier uniform |
| `xavierNormal(tensor, gain?)` | Xavier normal |
| `kaimingUniform(tensor, a?, mode?)` | Kaiming uniform |
| `kaimingNormal(tensor, a?, mode?)` | Kaiming normal |
| `uniform(tensor, a, b)` | Равномерное распределение |
| `normal(tensor, mean, std)` | Нормальное распределение |
| `constant(tensor, value)` | Константа |
| `zeros_(tensor)` | Заполнить нулями |
| `ones_(tensor)` | Заполнить единицами |
| `orthogonal(tensor, gain?)` | Ортогональная инициализация |

### Трансформеры

| Класс | Описание |
|-------|----------|
| `MultiHeadAttention(d_model, nhead)` | Multi-head attention |
| `ScaledDotProductAttention()` | Scaled dot-product attention |
| `FeedForward(d_model, dim_ff)` | Feed-forward сеть |
| `TransformerEncoderBlock(config)` | Блок encoder |
| `TransformerDecoderBlock(config)` | Блок decoder |
| `TransformerEncoder(config)` | Полный encoder |
| `TransformerDecoder(config)` | Полный decoder |
| `PositionalEncoding(d_model, max_len)` | Синусоидальное позиционное кодирование |
| `LearnedPositionalEmbedding(max_len, d)` | Обучаемые позиции |
| `RotaryEmbedding(dim)` | Rotary Position Embedding (RoPE) |
| `createCausalMask(size)` | Causal mask для авторегрессии |
| `createPaddingMask(lengths, max_len)` | Padding mask |

### Готовые модели

| Класс | Описание |
|-------|----------|
| `GPT(config)` | GPT-style трансформер |
| `createSmallGPT()` | Маленькая GPT модель |
| `createMediumGPT()` | Средняя GPT модель |
| `VAE(config)` | Вариационный автокодировщик |
| `ConvVAE(config)` | Свёрточный VAE |
| `createMNISTVAE()` | VAE для MNIST |
| `TRM(config)` | TRM модель |
| `TRMv2(config)` | TRM v2 с attention и MoE |
| `TRMUltra(config)` | TRM Ultra (стабильная) |
| `TRMFinal(config)` | TRM Final (оптимизированная) |
| `TRMX(config)` | TRM-X (extreme performance) |
| `TRMLite(config)` | TRM-Lite (простая) |
| `TRMPro(config)` | TRM-Pro (98%+ accuracy) |
| `TRMSupreme(config)` | TRM-Supreme с residual scaling |
| `MultimodalFewShot(config)` | Мультимодальная модель |

### Утилиты

| Функция/Класс | Описание |
|---------------|----------|
| `saveModel(model, path)` | Сохранить модель |
| `loadModel(model, path)` | Загрузить модель |
| `saveCheckpoint(data, path)` | Сохранить чекпоинт |
| `loadCheckpoint(path)` | Загрузить чекпоинт |
| `HuggingFaceHub` | Интеграция с Hugging Face |
| `Quantizer` | Квантизация модели |
| `LoRALayer` | Low-Rank Adaptation |
| `AdapterLayer` | Adapter layers |
| `DataLoader` | Загрузка данных батчами |
| `Tokenizer` | Токенизация текста |
| `visualize(model)` | Визуализация архитектуры |
| `summary(model, inputShape)` | Сводка модели |
| `manualSeed(seed)` | Установить seed для воспроизводимости |

### Вычисления

| Функция/Класс | Описание |
|---------------|----------|
| `isWebGPUSupported()` | Проверка поддержки WebGPU |
| `createDevice(type)` | Создать устройство (cpu/gpu) |
| `DeviceManager` | Управление устройствами |
| `GPUBackend` | GPU бэкенд |
| `WorkerPool` | Пул воркеров для CPU |
| `HybridEngine` | Гибридный CPU/GPU движок |
| `asyncOps` | Асинхронные операции |

## Поддержка GPU

Brainy ML поддерживает WebGPU для GPU ускорения:

```typescript
import { isWebGPUSupported, createDevice, GPUBackend } from 'brainy-ml';

if (await isWebGPUSupported()) {
  const device = await createDevice('gpu');
  console.log('GPU включен!');

  // GPU бэкенд для тензорных операций
  const gpu = await GPUBackend.create();
  const result = await gpu.matmul(a, b);
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
- `08-transformer.ts` - Transformer архитектура
- `09-vae.ts` - Вариационный автокодировщик
- `10-reinforcement-learning.ts` - DQN агент
- `26-huggingface-example.ts` - Интеграция с Hugging Face
- `27-qwen3-example.ts` - Загрузка модели Qwen

## Запуск

```bash
# Запуск примера
bun run examples/01-basic-tensors.ts

# Запуск тестов
bun test

# Запуск конкретного теста
bun test tests/tensor.test.ts
```

## Вклад в проект

Вклады приветствуются! Пожалуйста, создавайте Pull Request.

## Лицензия

MIT License - см. файл [LICENSE](LICENSE) для деталей.

## Ссылки

- [Документация](https://nopass0.github.io/brainy)
- [GitHub репозиторий](https://github.com/Nopass0/brainy)
- [npm пакет](https://www.npmjs.com/package/brainy-ml)
