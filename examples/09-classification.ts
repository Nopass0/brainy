/**
 * @fileoverview Пример классификации с нейронной сетью
 * @description Классификация данных с использованием MLP и CNN
 */

import {
  tensor,
  randn,
  zeros,
  ones,
  rand,
  Tensor,
  Module,
  Sequential,
  Linear,
  Conv2d,
  MaxPool2d,
  Flatten,
  Dropout,
  BatchNorm1d,
  ReLU,
  Softmax,
  CrossEntropyLoss,
  Adam,
  SGD,
  DType,
  TensorDataset,
  DataLoader,
  trainTestSplit,
} from '../src';

import { saveCheckpoint, loadCheckpoint, CompressionFormat } from '../src/utils/checkpoint';
import { dynamicQuantize, getModelSize } from '../src/utils/quantization';
import { FineTuneTrainer, FineTuneStrategy } from '../src/utils/finetune';

// ============================================
// 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ
// ============================================

console.log('=== Пример: Классификация с нейронной сетью ===\n');

/**
 * Создаёт синтетический датасет для классификации
 * 3 класса: спирали с разным закручиванием
 */
function createSpiralDataset(numSamplesPerClass: number = 100): {
  X: Tensor;
  y: Tensor;
} {
  const numClasses = 3;
  const numFeatures = 2;
  const totalSamples = numSamplesPerClass * numClasses;

  const X = new Float32Array(totalSamples * numFeatures);
  const y = new Float32Array(totalSamples);

  for (let c = 0; c < numClasses; c++) {
    for (let i = 0; i < numSamplesPerClass; i++) {
      const idx = c * numSamplesPerClass + i;
      const r = i / numSamplesPerClass;
      const theta = c * 4 + r * 4 + (Math.random() * 0.2 - 0.1);

      X[idx * numFeatures + 0] = r * Math.cos(theta);
      X[idx * numFeatures + 1] = r * Math.sin(theta);
      y[idx] = c;
    }
  }

  return {
    X: new Tensor(X, [totalSamples, numFeatures]),
    y: new Tensor(y, [totalSamples], { dtype: DType.Int32 }),
  };
}

console.log('Создание датасета спиралей...');

const { X, y } = createSpiralDataset(150);
console.log(`Всего образцов: ${X.shape[0]}`);
console.log(`Признаков: ${X.shape[1]}`);
console.log(`Классов: 3\n`);

// Разделяем на train/test
const splitIdx = Math.floor(X.shape[0] * 0.8);
const trainX = new Tensor(
  X.data.slice(0, splitIdx * 2),
  [splitIdx, 2]
);
const trainY = new Tensor(
  y.data.slice(0, splitIdx),
  [splitIdx],
  { dtype: DType.Int32 }
);
const testX = new Tensor(
  X.data.slice(splitIdx * 2),
  [X.shape[0] - splitIdx, 2]
);
const testY = new Tensor(
  y.data.slice(splitIdx),
  [X.shape[0] - splitIdx],
  { dtype: DType.Int32 }
);

console.log(`Train: ${trainX.shape[0]} samples`);
console.log(`Test: ${testX.shape[0]} samples\n`);

// ============================================
// 2. СОЗДАНИЕ MLP МОДЕЛИ
// ============================================

console.log('Создание MLP модели...');

class MLPClassifier extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private fc3: Linear;
  private relu: ReLU;
  private dropout: Dropout;
  private bn1: BatchNorm1d;
  private bn2: BatchNorm1d;

  constructor(inputDim: number, hiddenDim: number, numClasses: number) {
    super();

    this.fc1 = new Linear(inputDim, hiddenDim);
    this.bn1 = new BatchNorm1d(hiddenDim);
    this.fc2 = new Linear(hiddenDim, hiddenDim);
    this.bn2 = new BatchNorm1d(hiddenDim);
    this.fc3 = new Linear(hiddenDim, numClasses);
    this.relu = new ReLU();
    this.dropout = new Dropout(0.3);

    this.registerModule('fc1', this.fc1);
    this.registerModule('bn1', this.bn1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('bn2', this.bn2);
    this.registerModule('fc3', this.fc3);
    this.registerModule('relu', this.relu);
    this.registerModule('dropout', this.dropout);
  }

  forward(x: Tensor): Tensor {
    let hidden = this.fc1.forward(x);
    hidden = this.bn1.forward(hidden);
    hidden = this.relu.forward(hidden);
    hidden = this.dropout.forward(hidden);

    hidden = this.fc2.forward(hidden);
    hidden = this.bn2.forward(hidden);
    hidden = this.relu.forward(hidden);
    hidden = this.dropout.forward(hidden);

    return this.fc3.forward(hidden);
  }
}

const model = new MLPClassifier(2, 64, 3);
const modelSize = getModelSize(model);

console.log('Архитектура MLP:');
console.log('  Input(2) -> Linear(64) -> BN -> ReLU -> Dropout');
console.log('  -> Linear(64) -> BN -> ReLU -> Dropout -> Linear(3)');
console.log(`  Total parameters: ${modelSize.totalParams}`);
console.log(`  Size: ${modelSize.sizeMB.toFixed(3)} MB\n`);

// ============================================
// 3. ОБУЧЕНИЕ МОДЕЛИ
// ============================================

console.log('Обучение модели...');

const criterion = new CrossEntropyLoss();
const optimizer = new Adam(model.parameters(), 0.01);

const numEpochs = 100;
const batchSize = 32;

model.train();

const history: { loss: number; accuracy: number }[] = [];

for (let epoch = 0; epoch < numEpochs; epoch++) {
  let totalLoss = 0;
  let correct = 0;
  let total = 0;
  let numBatches = 0;

  // Mini-batch training
  const indices = Array.from({ length: trainX.shape[0] }, (_, i) => i);
  indices.sort(() => Math.random() - 0.5);

  for (let i = 0; i < indices.length; i += batchSize) {
    const batchIndices = indices.slice(i, Math.min(i + batchSize, indices.length));
    if (batchIndices.length < batchSize / 2) continue;

    // Создаём батч
    const batchXData = new Float32Array(batchIndices.length * 2);
    const batchYData = new Float32Array(batchIndices.length);

    for (let j = 0; j < batchIndices.length; j++) {
      const idx = batchIndices[j];
      batchXData[j * 2] = trainX.data[idx * 2];
      batchXData[j * 2 + 1] = trainX.data[idx * 2 + 1];
      batchYData[j] = trainY.data[idx];
    }

    const batchX = new Tensor(batchXData, [batchIndices.length, 2]);
    const batchY = new Tensor(batchYData, [batchIndices.length], { dtype: DType.Int32 });

    // Forward
    const logits = model.forward(batchX);
    const loss = criterion.forward(logits, batchY);

    // Backward
    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    totalLoss += loss.item();

    // Accuracy
    for (let j = 0; j < batchIndices.length; j++) {
      let maxIdx = 0;
      let maxVal = logits.data[j * 3];
      for (let k = 1; k < 3; k++) {
        if (logits.data[j * 3 + k] > maxVal) {
          maxVal = logits.data[j * 3 + k];
          maxIdx = k;
        }
      }
      if (maxIdx === batchYData[j]) correct++;
      total++;
    }

    numBatches++;
  }

  const avgLoss = totalLoss / numBatches;
  const accuracy = correct / total;
  history.push({ loss: avgLoss, accuracy });

  if ((epoch + 1) % 20 === 0 || epoch === 0) {
    console.log(
      `Epoch ${epoch + 1}/${numEpochs} - ` +
      `Loss: ${avgLoss.toFixed(4)}, ` +
      `Accuracy: ${(accuracy * 100).toFixed(2)}%`
    );
  }
}

console.log('\nОбучение завершено!\n');

// ============================================
// 4. ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ
// ============================================

console.log('Оценка на тестовых данных...');

model.eval();

const testLogits = model.forward(testX);

let testCorrect = 0;
for (let i = 0; i < testX.shape[0]; i++) {
  let maxIdx = 0;
  let maxVal = testLogits.data[i * 3];
  for (let k = 1; k < 3; k++) {
    if (testLogits.data[i * 3 + k] > maxVal) {
      maxVal = testLogits.data[i * 3 + k];
      maxIdx = k;
    }
  }
  if (maxIdx === testY.data[i]) testCorrect++;
}

const testAccuracy = testCorrect / testX.shape[0];
console.log(`Test Accuracy: ${(testAccuracy * 100).toFixed(2)}%\n`);

// Confusion matrix
console.log('Confusion Matrix:');
const confMatrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

for (let i = 0; i < testX.shape[0]; i++) {
  let predIdx = 0;
  let maxVal = testLogits.data[i * 3];
  for (let k = 1; k < 3; k++) {
    if (testLogits.data[i * 3 + k] > maxVal) {
      maxVal = testLogits.data[i * 3 + k];
      predIdx = k;
    }
  }
  const trueIdx = Math.floor(testY.data[i]);
  confMatrix[trueIdx][predIdx]++;
}

console.log('            Predicted');
console.log('           0    1    2');
for (let i = 0; i < 3; i++) {
  console.log(`Actual ${i}:  ${confMatrix[i].map(v => v.toString().padStart(3)).join('  ')}`);
}
console.log('');

// ============================================
// 5. КЛАССИФИКАЦИЯ ИЗОБРАЖЕНИЙ (ПРОСТОЙ CNN)
// ============================================

console.log('Создание CNN для классификации изображений...');

class SimpleCNN extends Module {
  private conv1: Conv2d;
  private conv2: Conv2d;
  private pool: MaxPool2d;
  private flatten: Flatten;
  private fc1: Linear;
  private fc2: Linear;
  private relu: ReLU;
  private dropout: Dropout;

  constructor(numClasses: number = 10) {
    super();

    // Conv layers
    this.conv1 = new Conv2d(1, 16, 3, 1, 1);
    this.conv2 = new Conv2d(16, 32, 3, 1, 1);
    this.pool = new MaxPool2d(2);
    this.flatten = new Flatten();

    // FC layers (для 14x14 -> 7x7 после pooling)
    this.fc1 = new Linear(32 * 7 * 7, 128);
    this.fc2 = new Linear(128, numClasses);

    this.relu = new ReLU();
    this.dropout = new Dropout(0.5);

    this.registerModule('conv1', this.conv1);
    this.registerModule('conv2', this.conv2);
    this.registerModule('pool', this.pool);
    this.registerModule('flatten', this.flatten);
    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
  }

  forward(x: Tensor): Tensor {
    // Conv block 1
    let hidden = this.conv1.forward(x);
    hidden = this.relu.forward(hidden);
    hidden = this.pool.forward(hidden);

    // Conv block 2
    hidden = this.conv2.forward(hidden);
    hidden = this.relu.forward(hidden);
    hidden = this.pool.forward(hidden);

    // FC layers
    hidden = this.flatten.forward(hidden);
    hidden = this.fc1.forward(hidden);
    hidden = this.relu.forward(hidden);
    hidden = this.dropout.forward(hidden);
    hidden = this.fc2.forward(hidden);

    return hidden;
  }
}

const cnn = new SimpleCNN(10);
const cnnSize = getModelSize(cnn);

console.log('Архитектура CNN:');
console.log('  Conv(1->16, 3x3) -> ReLU -> MaxPool(2x2)');
console.log('  -> Conv(16->32, 3x3) -> ReLU -> MaxPool(2x2)');
console.log('  -> Flatten -> Linear(128) -> ReLU -> Dropout -> Linear(10)');
console.log(`  Total parameters: ${cnnSize.totalParams}`);
console.log(`  Size: ${cnnSize.sizeMB.toFixed(3)} MB\n`);

// ============================================
// 6. КВАНТИЗАЦИЯ МОДЕЛИ
// ============================================

console.log('Квантизация MLP модели...');

const quantizedModel = dynamicQuantize(model, 8);
const quantStats = quantizedModel.getQuantizationStats();

console.log('Статистика квантизации:');
console.log(`  Original size: ${(quantStats.totalOriginalSize / 1024).toFixed(2)} KB`);
console.log(`  Compressed size: ${(quantStats.totalCompressedSize / 1024).toFixed(2)} KB`);
console.log(`  Compression ratio: ${quantStats.compressionRatio.toFixed(2)}x`);
console.log(`  Quantized layers: ${quantStats.numQuantizedLayers}\n`);

// ============================================
// 7. СОХРАНЕНИЕ И ЗАГРУЗКА
// ============================================

console.log('Сохранение модели...');

await saveCheckpoint('./classifier_checkpoint.brainy', model, optimizer, {
  compression: CompressionFormat.GZIP,
  metadata: {
    epoch: numEpochs,
    metrics: {
      testAccuracy,
    },
  },
});

console.log('Модель сохранена!\n');

// Загрузка
const loadedModel = new MLPClassifier(2, 64, 3);
const metadata = await loadCheckpoint('./classifier_checkpoint.brainy', loadedModel);

console.log('Модель загружена!');
console.log(`  Created: ${metadata.createdAt}`);
console.log(`  Compression: ${metadata.compression}\n`);

// Проверка загруженной модели
loadedModel.eval();
const loadedLogits = loadedModel.forward(testX);

let loadedCorrect = 0;
for (let i = 0; i < testX.shape[0]; i++) {
  let maxIdx = 0;
  let maxVal = loadedLogits.data[i * 3];
  for (let k = 1; k < 3; k++) {
    if (loadedLogits.data[i * 3 + k] > maxVal) {
      maxVal = loadedLogits.data[i * 3 + k];
      maxIdx = k;
    }
  }
  if (maxIdx === testY.data[i]) loadedCorrect++;
}

console.log(`Loaded model accuracy: ${((loadedCorrect / testX.shape[0]) * 100).toFixed(2)}%\n`);

// ============================================
// 8. ВИЗУАЛИЗАЦИЯ ГРАНИЦ РЕШЕНИЙ (ASCII)
// ============================================

console.log('Визуализация границ решений:');

const gridSize = 20;
const gridRange = 1.2;

let visualization = '';
for (let j = gridSize - 1; j >= 0; j--) {
  let row = '';
  for (let i = 0; i < gridSize; i++) {
    const x = (i / (gridSize - 1)) * 2 * gridRange - gridRange;
    const y = (j / (gridSize - 1)) * 2 * gridRange - gridRange;

    const point = new Tensor([x, y], [1, 2]);
    const pred = model.forward(point);

    let maxIdx = 0;
    let maxVal = pred.data[0];
    for (let k = 1; k < 3; k++) {
      if (pred.data[k] > maxVal) {
        maxVal = pred.data[k];
        maxIdx = k;
      }
    }

    const symbols = ['0', '1', '2'];
    row += symbols[maxIdx] + ' ';
  }
  visualization += row + '\n';
}

console.log(visualization);

console.log('=== Пример завершён ===');
