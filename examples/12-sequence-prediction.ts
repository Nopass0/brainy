/**
 * @fileoverview Sequence Prediction - Угадывание паттернов в последовательностях
 * @description Нейросеть учится предсказывать следующий элемент в числовых последовательностях
 */

import {
  Tensor,
  tensor,
  zeros,
  Sequential,
  Linear,
  ReLU,
  Tanh,
  Adam,
  Module,
  Parameter,
  summary,
  tanh,
} from '../src';

console.log('='.repeat(60));
console.log('Sequence Prediction - Угадывание паттернов');
console.log('='.repeat(60));

// ============================================
// Простая RNN ячейка (без LSTM модуля)
// ============================================
class SimpleRNN extends Module {
  private inputSize: number;
  private hiddenSize: number;
  private Wih: Parameter;  // input -> hidden
  private Whh: Parameter;  // hidden -> hidden
  private bh: Parameter;   // hidden bias

  constructor(inputSize: number, hiddenSize: number) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;

    // Xavier initialization
    const scale1 = Math.sqrt(2 / (inputSize + hiddenSize));
    const scale2 = Math.sqrt(2 / (hiddenSize + hiddenSize));

    this.Wih = new Parameter(
      new Tensor(
        Float32Array.from({ length: inputSize * hiddenSize }, () => (Math.random() - 0.5) * 2 * scale1),
        [inputSize, hiddenSize]
      )
    );
    this.Whh = new Parameter(
      new Tensor(
        Float32Array.from({ length: hiddenSize * hiddenSize }, () => (Math.random() - 0.5) * 2 * scale2),
        [hiddenSize, hiddenSize]
      )
    );
    this.bh = new Parameter(zeros([hiddenSize]));

    this.registerParameter('Wih', this.Wih);
    this.registerParameter('Whh', this.Whh);
    this.registerParameter('bh', this.bh);
  }

  forward(x: Tensor, hidden?: Tensor): { output: Tensor; hidden: Tensor } {
    const batchSize = x.shape[0];
    const seqLen = x.shape[1];

    let h = hidden || zeros([batchSize, this.hiddenSize]);

    const outputs: Tensor[] = [];

    for (let t = 0; t < seqLen; t++) {
      // Extract x[:, t, :]
      const xtData = new Float32Array(batchSize * this.inputSize);
      for (let b = 0; b < batchSize; b++) {
        for (let i = 0; i < this.inputSize; i++) {
          xtData[b * this.inputSize + i] = x.data[b * seqLen * this.inputSize + t * this.inputSize + i];
        }
      }
      const xt = new Tensor(xtData, [batchSize, this.inputSize], { requiresGrad: x.requiresGrad });

      // h = tanh(x @ Wih + h @ Whh + bh)
      const xh = xt.matmul(this.Wih.data);
      const hh = h.matmul(this.Whh.data);
      h = tanh(xh.add(hh).add(this.bh.data));

      outputs.push(h);
    }

    // Return last hidden state
    return { output: h, hidden: h };
  }
}

// ============================================
// Модель предсказания последовательностей
// ============================================
class SequencePredictor extends Module {
  private rnn: SimpleRNN;
  private fc: Linear;

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    super();
    this.rnn = new SimpleRNN(inputSize, hiddenSize);
    this.fc = new Linear(hiddenSize, outputSize);

    this.registerModule('rnn', this.rnn);
    this.registerModule('fc', this.fc);
  }

  forward(x: Tensor): Tensor {
    const { output } = this.rnn.forward(x);
    return this.fc.forward(output);
  }
}

// ============================================
// Генерация данных
// ============================================
function generateArithmeticSequence(start: number, step: number, length: number): number[] {
  return Array.from({ length }, (_, i) => start + step * i);
}

function generateGeometricSequence(start: number, ratio: number, length: number): number[] {
  return Array.from({ length }, (_, i) => start * Math.pow(ratio, i));
}

function generateFibonacci(length: number): number[] {
  const seq = [1, 1];
  for (let i = 2; i < length; i++) {
    seq.push(seq[i - 1] + seq[i - 2]);
  }
  return seq.slice(0, length);
}

function generateSquares(length: number): number[] {
  return Array.from({ length }, (_, i) => (i + 1) * (i + 1));
}

function generateTriangular(length: number): number[] {
  return Array.from({ length }, (_, i) => ((i + 1) * (i + 2)) / 2);
}

// ============================================
// 1. Арифметическая прогрессия
// ============================================
console.log('\n[1] Арифметическая прогрессия');
console.log('-'.repeat(40));

const seqLength = 5;  // Входная длина
const hiddenSize = 32;

// Генерируем обучающие данные
const trainingData: { input: number[]; target: number }[] = [];

// Разные арифметические прогрессии
for (let step = 1; step <= 10; step++) {
  for (let start = 0; start <= 20; start += 2) {
    const seq = generateArithmeticSequence(start, step, seqLength + 1);
    trainingData.push({
      input: seq.slice(0, seqLength),
      target: seq[seqLength],
    });
  }
}

console.log(`Обучающих примеров: ${trainingData.length}`);
console.log('Примеры:');
for (let i = 0; i < 3; i++) {
  const idx = Math.floor(Math.random() * trainingData.length);
  console.log(`  [${trainingData[idx].input.join(', ')}] -> ${trainingData[idx].target}`);
}

// Нормализация
const maxVal = Math.max(...trainingData.flatMap(d => [...d.input, d.target]));
const normalize = (x: number) => x / maxVal;
const denormalize = (x: number) => x * maxVal;

// Модель
const model = new SequencePredictor(1, hiddenSize, 1);
const optimizer = new Adam(model.parameters(), 0.01);

console.log('\nОбучение модели...');
const epochs = 50;

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;

  // Shuffle
  const shuffled = [...trainingData].sort(() => Math.random() - 0.5);

  for (const sample of shuffled) {
    // Prepare input: [batch=1, seq_len, features=1]
    const inputData = new Float32Array(seqLength);
    for (let i = 0; i < seqLength; i++) {
      inputData[i] = normalize(sample.input[i]);
    }
    const input = new Tensor(inputData, [1, seqLength, 1], { requiresGrad: true });

    const targetData = new Float32Array([normalize(sample.target)]);
    const target = new Tensor(targetData, [1, 1]);

    // Forward
    const output = model.forward(input);

    // Loss (MSE)
    const loss = output.sub(target).pow(2).mean();

    // Backward
    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    totalLoss += loss.item();
  }

  if ((epoch + 1) % 10 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${(totalLoss / trainingData.length).toFixed(6)}`);
  }
}

// Тестирование
console.log('\nТестирование:');
const testCases = [
  [2, 4, 6, 8, 10],      // step=2, expected=12
  [5, 10, 15, 20, 25],   // step=5, expected=30
  [3, 6, 9, 12, 15],     // step=3, expected=18
  [1, 4, 7, 10, 13],     // step=3, expected=16
];

for (const testSeq of testCases) {
  const inputData = new Float32Array(seqLength);
  for (let i = 0; i < seqLength; i++) {
    inputData[i] = normalize(testSeq[i]);
  }
  const input = new Tensor(inputData, [1, seqLength, 1]);
  const output = model.forward(input);
  const predicted = denormalize(output.data[0]);
  const expected = testSeq[testSeq.length - 1] + (testSeq[1] - testSeq[0]);
  console.log(`  [${testSeq.join(', ')}] -> Предсказано: ${predicted.toFixed(1)}, Ожидается: ${expected}`);
}

// ============================================
// 2. Квадраты чисел
// ============================================
console.log('\n[2] Квадраты чисел');
console.log('-'.repeat(40));

const squareData: { input: number[]; target: number }[] = [];
for (let start = 1; start <= 10; start++) {
  const seq = generateSquares(start + seqLength);
  squareData.push({
    input: seq.slice(start - 1, start - 1 + seqLength),
    target: seq[start - 1 + seqLength],
  });
}

console.log('Примеры квадратов:');
for (const sample of squareData.slice(0, 3)) {
  console.log(`  [${sample.input.join(', ')}] -> ${sample.target}`);
}

// ============================================
// 3. Fibonacci
// ============================================
console.log('\n[3] Числа Фибоначчи');
console.log('-'.repeat(40));

const fibSeq = generateFibonacci(15);
console.log(`Последовательность: ${fibSeq.join(', ')}`);

// Модель для Fibonacci
const fibModel = new SequencePredictor(1, 64, 1);
const fibOptimizer = new Adam(fibModel.parameters(), 0.005);

// Данные
const fibData: { input: number[]; target: number }[] = [];
for (let i = 0; i <= fibSeq.length - seqLength - 1; i++) {
  fibData.push({
    input: fibSeq.slice(i, i + seqLength),
    target: fibSeq[i + seqLength],
  });
}

const fibMax = Math.max(...fibSeq);
const fibNorm = (x: number) => x / fibMax;
const fibDenorm = (x: number) => x * fibMax;

console.log('Обучение на Fibonacci...');
for (let epoch = 0; epoch < 100; epoch++) {
  let loss = 0;
  for (const sample of fibData) {
    const inputData = new Float32Array(seqLength);
    for (let i = 0; i < seqLength; i++) {
      inputData[i] = fibNorm(sample.input[i]);
    }
    const input = new Tensor(inputData, [1, seqLength, 1], { requiresGrad: true });
    const target = new Tensor(new Float32Array([fibNorm(sample.target)]), [1, 1]);

    const output = fibModel.forward(input);
    const l = output.sub(target).pow(2).mean();

    fibOptimizer.zeroGrad();
    l.backward();
    fibOptimizer.step();

    loss += l.item();
  }

  if ((epoch + 1) % 25 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${(loss / fibData.length).toFixed(6)}`);
  }
}

console.log('\nТест Fibonacci:');
const fibTest = fibSeq.slice(5, 5 + seqLength);  // [8, 13, 21, 34, 55]
const fibInputData = new Float32Array(seqLength);
for (let i = 0; i < seqLength; i++) {
  fibInputData[i] = fibNorm(fibTest[i]);
}
const fibInput = new Tensor(fibInputData, [1, seqLength, 1]);
const fibOutput = fibModel.forward(fibInput);
const fibPredicted = fibDenorm(fibOutput.data[0]);
console.log(`  [${fibTest.join(', ')}] -> Предсказано: ${fibPredicted.toFixed(0)}, Ожидается: ${fibSeq[5 + seqLength]}`);

// ============================================
// 4. Паттерн распознавание
// ============================================
console.log('\n[4] Распознавание типа последовательности');
console.log('-'.repeat(40));

// Создаём классификатор типа последовательности
class SequenceClassifier extends Module {
  private rnn: SimpleRNN;
  private fc: Sequential;

  constructor(inputSize: number, hiddenSize: number, numClasses: number) {
    super();
    this.rnn = new SimpleRNN(inputSize, hiddenSize);
    this.fc = new Sequential(
      new Linear(hiddenSize, 16),
      new ReLU(),
      new Linear(16, numClasses)
    );

    this.registerModule('rnn', this.rnn);
    this.registerModule('fc', this.fc);
  }

  forward(x: Tensor): Tensor {
    const { output } = this.rnn.forward(x);
    return this.fc.forward(output);
  }
}

// Типы: 0=арифметическая, 1=геометрическая, 2=квадраты
const classifierData: { input: number[]; label: number }[] = [];

// Арифметические (класс 0)
for (let i = 0; i < 30; i++) {
  const start = Math.floor(Math.random() * 10);
  const step = Math.floor(Math.random() * 5) + 1;
  classifierData.push({
    input: generateArithmeticSequence(start, step, seqLength),
    label: 0,
  });
}

// Геометрические (класс 1)
for (let i = 0; i < 30; i++) {
  const start = Math.floor(Math.random() * 5) + 1;
  const ratio = Math.floor(Math.random() * 3) + 2;
  classifierData.push({
    input: generateGeometricSequence(start, ratio, seqLength),
    label: 1,
  });
}

// Квадраты (класс 2)
for (let i = 0; i < 30; i++) {
  const offset = Math.floor(Math.random() * 5);
  classifierData.push({
    input: generateSquares(seqLength + offset).slice(offset),
    label: 2,
  });
}

// Нормализация для классификатора
const classMax = Math.max(...classifierData.flatMap(d => d.input));
const classNorm = (x: number) => x / classMax;

const classifier = new SequenceClassifier(1, 32, 3);
const classOptimizer = new Adam(classifier.parameters(), 0.01);

console.log('Обучение классификатора...');
for (let epoch = 0; epoch < 50; epoch++) {
  let correct = 0;
  const shuffled = [...classifierData].sort(() => Math.random() - 0.5);

  for (const sample of shuffled) {
    const inputData = new Float32Array(seqLength);
    for (let i = 0; i < seqLength; i++) {
      inputData[i] = classNorm(sample.input[i]);
    }
    const input = new Tensor(inputData, [1, seqLength, 1], { requiresGrad: true });

    const logits = classifier.forward(input);

    // Softmax + cross-entropy
    const maxLogit = Math.max(...logits.data);
    const expLogits = logits.data.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(x => x / sumExp);

    // Predicted class
    let predicted = 0;
    let maxProb = probs[0];
    for (let i = 1; i < 3; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        predicted = i;
      }
    }
    if (predicted === sample.label) correct++;

    // Cross-entropy loss
    const targetProbs = new Float32Array(3);
    targetProbs[sample.label] = 1;
    const loss = logits.sub(new Tensor(targetProbs, [1, 3])).pow(2).mean();

    classOptimizer.zeroGrad();
    loss.backward();
    classOptimizer.step();
  }

  if ((epoch + 1) % 10 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Accuracy = ${(correct / classifierData.length * 100).toFixed(1)}%`);
  }
}

// Тест классификатора
console.log('\nТест классификатора:');
const testSequences = [
  { seq: [2, 5, 8, 11, 14], type: 'Арифметическая' },
  { seq: [2, 4, 8, 16, 32], type: 'Геометрическая' },
  { seq: [1, 4, 9, 16, 25], type: 'Квадраты' },
];

const typeNames = ['Арифметическая', 'Геометрическая', 'Квадраты'];

for (const test of testSequences) {
  const inputData = new Float32Array(seqLength);
  for (let i = 0; i < seqLength; i++) {
    inputData[i] = classNorm(test.seq[i]);
  }
  const input = new Tensor(inputData, [1, seqLength, 1]);
  const logits = classifier.forward(input);

  let predicted = 0;
  let maxLogit = logits.data[0];
  for (let i = 1; i < 3; i++) {
    if (logits.data[i] > maxLogit) {
      maxLogit = logits.data[i];
      predicted = i;
    }
  }

  console.log(`  [${test.seq.join(', ')}] -> Предсказано: ${typeNames[predicted]}, Реальный: ${test.type}`);
}

console.log('\n' + '='.repeat(60));
console.log('Sequence Prediction демо завершено!');
console.log('='.repeat(60));
