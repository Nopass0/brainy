/**
 * @fileoverview TRM для последовательностей и паттернов
 * @description Демонстрация TRM на задачах моделирования последовательностей
 *
 * TRM использует рекурсивное уточнение для:
 * - Предсказания следующего элемента
 * - Распознавания паттернов
 * - Классификации последовательностей
 */

import {
  Tensor,
  tensor,
  zeros,
  ones,
  TRM,
  TRMSeq2Seq,
  createEnhancedTRM,
  createSequenceTRM,
  initTRMGPU,
  isTRMGPUAvailable,
  Adam,
} from '../src';

console.log('='.repeat(60));
console.log('TRM для последовательностей');
console.log('='.repeat(60));

// GPU initialization
initTRMGPU().catch(() => {});
console.log(`Устройство: ${isTRMGPUAvailable() ? 'GPU' : 'CPU'}`);

// ============================================
// 1. Предсказание следующего числа
// ============================================
console.log('\n[1] Предсказание следующего числа в последовательности');
console.log('-'.repeat(40));

function generateFibonacci(n: number): {
  x: Tensor;
  y: Tensor;
  sequences: number[][];
} {
  const seqLen = 5;
  const xData = new Float32Array(n * seqLen);
  const yData = new Float32Array(n);
  const sequences: number[][] = [];

  for (let i = 0; i < n; i++) {
    // Random start for Fibonacci-like sequence
    let a = Math.floor(Math.random() * 5) + 1;
    let b = Math.floor(Math.random() * 5) + 1;
    const seq = [a, b];

    for (let j = 0; j < seqLen - 1; j++) {
      const next = a + b;
      seq.push(next);
      a = b;
      b = next;
    }

    sequences.push(seq);

    // Normalize input (first 5 elements)
    for (let j = 0; j < seqLen; j++) {
      xData[i * seqLen + j] = seq[j] / 100;
    }
    // Target: next element
    yData[i] = seq[seqLen] / 200;
  }

  return {
    x: new Tensor(xData, [n, seqLen], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    sequences,
  };
}

const fibModel = createEnhancedTRM(5, 1, {
  hiddenDim: 64,
  numRecursions: 6,
  useSelfAttention: true,
  useGating: true,
});
const fibOpt = new Adam(fibModel.parameters(), 0.005);

const fibTrain = generateFibonacci(200);

console.log('Обучение на Fibonacci-подобных последовательностях...');
for (let epoch = 0; epoch < 60; epoch++) {
  const out = fibModel.forward(fibTrain.x);
  const loss = out.sub(fibTrain.y).pow(2).mean();

  fibOpt.zeroGrad();
  loss.backward();
  fibOpt.step();

  if ((epoch + 1) % 15 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${loss.item().toFixed(6)}`);
  }
}

// Test
console.log('\nТест предсказания:');
const fibTest = generateFibonacci(4);
const fibPred = fibModel.forward(fibTest.x);

for (let i = 0; i < 4; i++) {
  const seq = fibTest.sequences[i].slice(0, 5);
  const predicted = Math.round(fibPred.data[i] * 200);
  const actual = fibTest.sequences[i][5];
  const error = Math.abs(predicted - actual);
  const correct = error <= 5 ? 'V' : 'X';
  console.log(`  [${seq.join(', ')}] -> ${predicted} (ожидается ${actual}) [${correct}]`);
}

// ============================================
// 2. Классификация последовательностей
// ============================================
console.log('\n[2] Классификация типа последовательности');
console.log('-'.repeat(40));

type SeqType = 'increasing' | 'decreasing' | 'constant' | 'oscillating';

function generateClassification(n: number): {
  x: Tensor;
  y: Tensor;
  labels: SeqType[];
} {
  const seqLen = 6;
  const xData = new Float32Array(n * seqLen);
  const yData = new Float32Array(n * 4);  // One-hot for 4 classes
  const labels: SeqType[] = [];

  for (let i = 0; i < n; i++) {
    const type = ['increasing', 'decreasing', 'constant', 'oscillating'][Math.floor(Math.random() * 4)] as SeqType;
    labels.push(type);

    const start = Math.random() * 5 + 1;
    const seq: number[] = [];

    for (let j = 0; j < seqLen; j++) {
      switch (type) {
        case 'increasing':
          seq.push(start + j * (Math.random() * 0.5 + 0.5));
          break;
        case 'decreasing':
          seq.push(start - j * (Math.random() * 0.5 + 0.5) + 5);
          break;
        case 'constant':
          seq.push(start + (Math.random() - 0.5) * 0.2);
          break;
        case 'oscillating':
          seq.push(start + Math.sin(j * Math.PI) * 2);
          break;
      }
    }

    // Normalize
    for (let j = 0; j < seqLen; j++) {
      xData[i * seqLen + j] = seq[j] / 10;
    }

    // One-hot encoding
    const classIdx = ['increasing', 'decreasing', 'constant', 'oscillating'].indexOf(type);
    for (let c = 0; c < 4; c++) {
      yData[i * 4 + c] = c === classIdx ? 1 : 0;
    }
  }

  return {
    x: new Tensor(xData, [n, seqLen], { requiresGrad: true }),
    y: new Tensor(yData, [n, 4]),
    labels,
  };
}

const classModel = createEnhancedTRM(6, 4, {
  hiddenDim: 48,
  numRecursions: 5,
  useSelfAttention: true,
});
const classOpt = new Adam(classModel.parameters(), 0.01);

const classTrain = generateClassification(300);

console.log('Обучение классификатору последовательностей...');
for (let epoch = 0; epoch < 50; epoch++) {
  const out = classModel.forward(classTrain.x);
  const loss = out.sub(classTrain.y).pow(2).mean();

  classOpt.zeroGrad();
  loss.backward();
  classOpt.step();

  if ((epoch + 1) % 10 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${loss.item().toFixed(6)}`);
  }
}

// Test
console.log('\nТест классификации:');
const classTest = generateClassification(8);
const classPred = classModel.forward(classTest.x);

const typeNames = ['Возр.', 'Убыв.', 'Конст.', 'Осцил.'];
let classCorrect = 0;

for (let i = 0; i < 8; i++) {
  // Find predicted class (argmax)
  let maxVal = -Infinity;
  let predClass = 0;
  for (let c = 0; c < 4; c++) {
    if (classPred.data[i * 4 + c] > maxVal) {
      maxVal = classPred.data[i * 4 + c];
      predClass = c;
    }
  }

  const actualClass = ['increasing', 'decreasing', 'constant', 'oscillating'].indexOf(classTest.labels[i]);
  const correct = predClass === actualClass;
  if (correct) classCorrect++;

  console.log(`  ${classTest.labels[i].padEnd(11)} -> ${typeNames[predClass]} [${correct ? 'V' : 'X'}]`);
}
console.log(`  Точность: ${(classCorrect / 8 * 100).toFixed(1)}%`);

// ============================================
// 3. Распознавание паттернов (бинарные)
// ============================================
console.log('\n[3] Распознавание бинарных паттернов');
console.log('-'.repeat(40));

function generateBinaryPattern(n: number): {
  x: Tensor;
  y: Tensor;
  patterns: string[];
  targets: string[];
} {
  const seqLen = 8;
  const xData = new Float32Array(n * seqLen);
  const yData = new Float32Array(n);
  const patterns: string[] = [];
  const targets: string[] = [];

  for (let i = 0; i < n; i++) {
    // Generate patterns with completion
    const patternType = Math.floor(Math.random() * 4);
    let seq: number[];

    switch (patternType) {
      case 0: // Alternating: 0,1,0,1,0,1,0,?
        seq = [0, 1, 0, 1, 0, 1, 0, 1];
        break;
      case 1: // Doubles: 0,0,1,1,0,0,1,?
        seq = [0, 0, 1, 1, 0, 0, 1, 1];
        break;
      case 2: // All zeros then ones: 0,0,0,0,1,1,1,?
        seq = [0, 0, 0, 0, 1, 1, 1, 1];
        break;
      case 3: // Triples: 0,0,0,1,1,1,0,?
        seq = [0, 0, 0, 1, 1, 1, 0, 0];
        break;
      default:
        seq = [0, 1, 0, 1, 0, 1, 0, 1];
    }

    patterns.push(seq.slice(0, 7).join(''));
    targets.push(seq[7].toString());

    // Input: first 7 elements
    for (let j = 0; j < seqLen - 1; j++) {
      xData[i * seqLen + j] = seq[j];
    }
    xData[i * seqLen + 7] = 0;  // Padding

    // Target: 8th element
    yData[i] = seq[7];
  }

  return {
    x: new Tensor(xData, [n, seqLen], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    patterns,
    targets,
  };
}

const patternModel = createEnhancedTRM(8, 1, {
  hiddenDim: 32,
  numRecursions: 6,
  useGating: true,
});
const patternOpt = new Adam(patternModel.parameters(), 0.01);

const patternTrain = generateBinaryPattern(200);

console.log('Обучение на бинарных паттернах...');
for (let epoch = 0; epoch < 40; epoch++) {
  const out = patternModel.forward(patternTrain.x);
  const loss = out.sub(patternTrain.y).pow(2).mean();

  patternOpt.zeroGrad();
  loss.backward();
  patternOpt.step();
}

// Test
console.log('Тест завершения паттернов:');
const patternTest = generateBinaryPattern(6);
const patternPred = patternModel.forward(patternTest.x);

let patternCorrect = 0;
for (let i = 0; i < 6; i++) {
  const predicted = patternPred.data[i] > 0.5 ? 1 : 0;
  const actual = parseInt(patternTest.targets[i]);
  const correct = predicted === actual;
  if (correct) patternCorrect++;

  console.log(`  ${patternTest.patterns[i]}? -> ${predicted} (ожидается ${actual}) [${correct ? 'V' : 'X'}]`);
}
console.log(`  Точность: ${(patternCorrect / 6 * 100).toFixed(1)}%`);

// ============================================
// 4. Сумма скользящего окна
// ============================================
console.log('\n[4] Скользящая сумма (window sum)');
console.log('-'.repeat(40));

function generateWindowSum(n: number, windowSize: number = 3): {
  x: Tensor;
  y: Tensor;
  examples: { seq: number[]; sum: number }[];
} {
  const seqLen = 6;
  const xData = new Float32Array(n * seqLen);
  const yData = new Float32Array(n);
  const examples: { seq: number[]; sum: number }[] = [];

  for (let i = 0; i < n; i++) {
    const seq: number[] = [];
    for (let j = 0; j < seqLen; j++) {
      seq.push(Math.floor(Math.random() * 5));
    }

    // Sum of last `windowSize` elements
    let sum = 0;
    for (let j = seqLen - windowSize; j < seqLen; j++) {
      sum += seq[j];
    }

    examples.push({ seq, sum });

    for (let j = 0; j < seqLen; j++) {
      xData[i * seqLen + j] = seq[j] / 5;
    }
    yData[i] = sum / (5 * windowSize);
  }

  return {
    x: new Tensor(xData, [n, seqLen], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    examples,
  };
}

const windowModel = createEnhancedTRM(6, 1, {
  hiddenDim: 32,
  numRecursions: 4,
  useSelfAttention: true,
});
const windowOpt = new Adam(windowModel.parameters(), 0.01);

const windowTrain = generateWindowSum(200);

console.log('Обучение на скользящей сумме (последние 3 элемента)...');
for (let epoch = 0; epoch < 50; epoch++) {
  const out = windowModel.forward(windowTrain.x);
  const loss = out.sub(windowTrain.y).pow(2).mean();

  windowOpt.zeroGrad();
  loss.backward();
  windowOpt.step();
}

// Test
console.log('Тест скользящей суммы:');
const windowTest = generateWindowSum(5);
const windowPred = windowModel.forward(windowTest.x);

for (let i = 0; i < 5; i++) {
  const { seq, sum } = windowTest.examples[i];
  const predicted = Math.round(windowPred.data[i] * 15);
  const correct = Math.abs(predicted - sum) <= 1 ? 'V' : 'X';
  console.log(`  [${seq.join(', ')}] сумма последних 3: ${predicted} (ожидается ${sum}) [${correct}]`);
}

// ============================================
// 5. Обнаружение аномалий
// ============================================
console.log('\n[5] Обнаружение аномалий в последовательности');
console.log('-'.repeat(40));

function generateAnomaly(n: number): {
  x: Tensor;
  y: Tensor;
  hasAnomaly: boolean[];
  sequences: number[][];
} {
  const seqLen = 8;
  const xData = new Float32Array(n * seqLen);
  const yData = new Float32Array(n);
  const hasAnomaly: boolean[] = [];
  const sequences: number[][] = [];

  for (let i = 0; i < n; i++) {
    const anomaly = Math.random() > 0.5;
    hasAnomaly.push(anomaly);

    // Normal sequence: values around 5
    const seq: number[] = [];
    for (let j = 0; j < seqLen; j++) {
      seq.push(5 + (Math.random() - 0.5) * 2);
    }

    // Add anomaly: one value is significantly different
    if (anomaly) {
      const anomalyPos = Math.floor(Math.random() * seqLen);
      seq[anomalyPos] = Math.random() > 0.5 ? 9 + Math.random() : 1 - Math.random();
    }

    sequences.push(seq.map(x => Math.round(x * 10) / 10));

    for (let j = 0; j < seqLen; j++) {
      xData[i * seqLen + j] = seq[j] / 10;
    }
    yData[i] = anomaly ? 1 : 0;
  }

  return {
    x: new Tensor(xData, [n, seqLen], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    hasAnomaly,
    sequences,
  };
}

const anomalyModel = createEnhancedTRM(8, 1, {
  hiddenDim: 32,
  numRecursions: 5,
  useSelfAttention: true,
  useGating: true,
});
const anomalyOpt = new Adam(anomalyModel.parameters(), 0.01);

const anomalyTrain = generateAnomaly(200);

console.log('Обучение детектору аномалий...');
for (let epoch = 0; epoch < 40; epoch++) {
  const out = anomalyModel.forward(anomalyTrain.x);
  const loss = out.sub(anomalyTrain.y).pow(2).mean();

  anomalyOpt.zeroGrad();
  loss.backward();
  anomalyOpt.step();
}

// Test
console.log('Тест обнаружения аномалий:');
const anomalyTest = generateAnomaly(6);
const anomalyPred = anomalyModel.forward(anomalyTest.x);

let anomalyCorrect = 0;
for (let i = 0; i < 6; i++) {
  const predicted = anomalyPred.data[i] > 0.5;
  const actual = anomalyTest.hasAnomaly[i];
  const correct = predicted === actual;
  if (correct) anomalyCorrect++;

  const seqStr = anomalyTest.sequences[i].map(x => x.toFixed(1)).join(', ');
  console.log(`  [${seqStr}]`);
  console.log(`    Аномалия: ${predicted ? 'Да' : 'Нет'} (ожидается ${actual ? 'Да' : 'Нет'}) [${correct ? 'V' : 'X'}]`);
}
console.log(`  Точность: ${(anomalyCorrect / 6 * 100).toFixed(1)}%`);

// ============================================
// 6. Рекурсивная история для sequence
// ============================================
console.log('\n[6] Визуализация рекурсивной обработки');
console.log('-'.repeat(40));

const vizModel = createEnhancedTRM(4, 2, {
  hiddenDim: 16,
  numRecursions: 6,
  useSelfAttention: true,
});

const vizInput = new Tensor(new Float32Array([0.2, 0.4, 0.6, 0.8]), [1, 4]);
const { output: vizOut, intermediates: vizInter } = vizModel.forwardWithHistory(vizInput);

console.log('Эволюция представления через рекурсии:');
console.log('Вход: [0.2, 0.4, 0.6, 0.8]');
for (let i = 0; i < vizInter.length; i++) {
  const y = vizInter[i].y;
  const yStr = Array.from(y.data).slice(0, 4).map(v => v.toFixed(2)).join(', ');
  console.log(`  Шаг ${i}: [${yStr}]`);
}
console.log(`  Выход: [${Array.from(vizOut.data).map(v => v.toFixed(2)).join(', ')}]`);

console.log('\n' + '='.repeat(60));
console.log('TRM Sequence демо завершено!');
console.log('='.repeat(60));
