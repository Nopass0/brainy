/**
 * @fileoverview TRM (Tiny Recursion Model) - Enhanced Recursive Refinement
 * @description Демонстрация улучшенной TRM архитектуры на задачах рассуждений
 *
 * TRM использует итеративное улучшение ответов через:
 * - x: Входное представление
 * - y: Текущий ответ (улучшается итеративно)
 * - z: Латентное состояние рассуждений
 *
 * Улучшения:
 * - Self-attention для лучшего рассуждения
 * - GRU-style gating для стабильного обучения
 * - Adaptive pondering (ACT) для динамического вычисления
 * - Поддержка GPU при наличии
 */

import {
  Tensor,
  tensor,
  zeros,
  TRM,
  TRMClassifier,
  createTinyTRM,
  createReasoningTRM,
  createEnhancedTRM,
  createPonderingTRM,
  initTRMGPU,
  isTRMGPUAvailable,
  Adam,
  summary,
  Sequential,
  Linear,
  ReLU,
} from '../src';

console.log('='.repeat(60));
console.log('TRM (Tiny Recursion Model) - Enhanced Version');
console.log('='.repeat(60));

// ============================================
// GPU Initialization
// ============================================
console.log('\n[0] Инициализация устройства');
console.log('-'.repeat(40));

// Try to initialize GPU (async, but we'll continue CPU if not available)
initTRMGPU().then(gpuAvailable => {
  if (gpuAvailable) {
    console.log('GPU: Включено (WebGPU)');
  } else {
    console.log('GPU: Недоступно, используется CPU');
  }
}).catch(() => {
  console.log('GPU: Инициализация не удалась, используется CPU');
});

// For non-async context, just check availability
console.log(`GPU статус: ${isTRMGPUAvailable() ? 'Доступен' : 'Недоступен (CPU режим)'}`);
console.log('');

// ============================================
// 1. Простая задача: XOR с шумом
// ============================================
console.log('\n[1] XOR с шумом (TRM vs обычная сеть)');
console.log('-'.repeat(40));

// Генерируем данные с шумом
function generateNoisyXOR(n: number, noiseLevel: number = 0.1): { x: Tensor; y: Tensor } {
  const xData = new Float32Array(n * 2);
  const yData = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const xorResult = a !== b ? 1 : 0;

    // Add noise to inputs
    xData[i * 2] = a + (Math.random() - 0.5) * noiseLevel;
    xData[i * 2 + 1] = b + (Math.random() - 0.5) * noiseLevel;
    yData[i] = xorResult;
  }

  return {
    x: new Tensor(xData, [n, 2], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
  };
}

// TRM модель
const trmModel = createTinyTRM(2, 1, 32, 4);  // 4 рекурсии
const trmOptimizer = new Adam(trmModel.parameters(), 0.01);

// Обычная сеть (для сравнения)
const mlpModel = new Sequential(
  new Linear(2, 32),
  new ReLU(),
  new Linear(32, 32),
  new ReLU(),
  new Linear(32, 1)
);
const mlpOptimizer = new Adam(mlpModel.parameters(), 0.01);

console.log('Обучение моделей...');

const trainData = generateNoisyXOR(200, 0.3);
const epochs = 50;

let trmLoss = 0;
let mlpLoss = 0;

for (let epoch = 0; epoch < epochs; epoch++) {
  // TRM training
  const trmOut = trmModel.forward(trainData.x);
  const trmL = trmOut.sub(trainData.y).pow(2).mean();
  trmOptimizer.zeroGrad();
  trmL.backward();
  trmOptimizer.step();
  trmLoss = trmL.item();

  // MLP training
  const mlpOut = mlpModel.forward(trainData.x);
  const mlpL = mlpOut.sub(trainData.y).pow(2).mean();
  mlpOptimizer.zeroGrad();
  mlpL.backward();
  mlpOptimizer.step();
  mlpLoss = mlpL.item();

  if ((epoch + 1) % 10 === 0) {
    console.log(`  Эпоха ${epoch + 1}: TRM Loss = ${trmLoss.toFixed(4)}, MLP Loss = ${mlpLoss.toFixed(4)}`);
  }
}

// Тестирование
const testData = generateNoisyXOR(50, 0.3);
const trmPred = trmModel.forward(testData.x);
const mlpPred = mlpModel.forward(testData.x);

let trmCorrect = 0, mlpCorrect = 0;
for (let i = 0; i < 50; i++) {
  const trmAnswer = trmPred.data[i] > 0.5 ? 1 : 0;
  const mlpAnswer = mlpPred.data[i] > 0.5 ? 1 : 0;
  const actual = testData.y.data[i];

  if (trmAnswer === actual) trmCorrect++;
  if (mlpAnswer === actual) mlpCorrect++;
}

console.log(`\nТочность на тесте:`);
console.log(`  TRM: ${(trmCorrect / 50 * 100).toFixed(1)}%`);
console.log(`  MLP: ${(mlpCorrect / 50 * 100).toFixed(1)}%`);

// ============================================
// 2. Визуальная аналогия (паттерны)
// ============================================
console.log('\n[2] Визуальные аналогии (A:B :: C:?)');
console.log('-'.repeat(40));

// Простые паттерны: поворот, отражение, инверсия
// Представляем как 3x3 грид = 9 входов

function createPattern(type: 'rotate' | 'mirror' | 'invert'): {
  a: number[];
  b: number[];
  c: number[];
  d: number[];  // correct answer
} {
  // Simple L-shape
  const a = [1, 0, 0, 1, 0, 0, 1, 1, 1];

  let b: number[], d: number[];

  if (type === 'rotate') {
    // Rotate 90 degrees
    b = [1, 1, 1, 1, 0, 0, 1, 0, 0];
    // Apply same to different shape (T-shape)
    // c = T, d = rotated T
  } else if (type === 'mirror') {
    // Mirror horizontally
    b = [0, 0, 1, 0, 0, 1, 1, 1, 1];
  } else {
    // Invert
    b = a.map(x => 1 - x);
  }

  // Different starting shape for C
  const c = [0, 1, 0, 1, 1, 1, 0, 1, 0];  // Plus sign

  // Apply same transformation
  if (type === 'rotate') {
    d = [0, 1, 0, 1, 1, 1, 0, 1, 0];  // Plus is symmetric
  } else if (type === 'mirror') {
    d = [0, 1, 0, 1, 1, 1, 0, 1, 0];  // Plus is symmetric
  } else {
    d = c.map(x => 1 - x);
  }

  return { a, b: b!, c, d: d! };
}

// Генерируем данные для аналогий
const analogyData: { input: number[]; output: number[] }[] = [];

for (let i = 0; i < 30; i++) {
  const types: ('rotate' | 'mirror' | 'invert')[] = ['rotate', 'mirror', 'invert'];
  const type = types[i % 3];
  const pattern = createPattern(type);

  // Input: [a, b, c] flattened = 27 values
  analogyData.push({
    input: [...pattern.a, ...pattern.b, ...pattern.c],
    output: pattern.d,
  });
}

const analogyTRM = createTinyTRM(27, 9, 64, 6);  // 6 recursions for reasoning
const analogyOptimizer = new Adam(analogyTRM.parameters(), 0.01);

console.log('Обучение TRM на визуальных аналогиях...');

for (let epoch = 0; epoch < 30; epoch++) {
  let totalLoss = 0;

  for (const sample of analogyData) {
    const input = new Tensor(new Float32Array(sample.input), [1, 27], { requiresGrad: true });
    const target = new Tensor(new Float32Array(sample.output), [1, 9]);

    const output = analogyTRM.forward(input);
    const loss = output.sub(target).pow(2).mean();

    analogyOptimizer.zeroGrad();
    loss.backward();
    analogyOptimizer.step();

    totalLoss += loss.item();
  }

  if ((epoch + 1) % 10 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${(totalLoss / analogyData.length).toFixed(4)}`);
  }
}

// Тест
console.log('\nТест визуальной аналогии (инверсия):');
const testPattern = createPattern('invert');
const testInput = new Tensor(
  new Float32Array([...testPattern.a, ...testPattern.b, ...testPattern.c]),
  [1, 27]
);
const predicted = analogyTRM.forward(testInput);

console.log('  A (L-shape):');
for (let i = 0; i < 3; i++) {
  console.log('    ' + testPattern.a.slice(i * 3, i * 3 + 3).map(x => x ? '#' : '.').join(' '));
}
console.log('  B (inverted):');
for (let i = 0; i < 3; i++) {
  console.log('    ' + testPattern.b.slice(i * 3, i * 3 + 3).map(x => x ? '#' : '.').join(' '));
}
console.log('  C (plus):');
for (let i = 0; i < 3; i++) {
  console.log('    ' + testPattern.c.slice(i * 3, i * 3 + 3).map(x => x ? '#' : '.').join(' '));
}
console.log('  D (predicted inverted plus):');
for (let i = 0; i < 3; i++) {
  console.log('    ' + Array.from(predicted.data).slice(i * 3, i * 3 + 3).map(x => x > 0.5 ? '#' : '.').join(' '));
}
console.log('  D (expected):');
for (let i = 0; i < 3; i++) {
  console.log('    ' + testPattern.d.slice(i * 3, i * 3 + 3).map(x => x ? '#' : '.').join(' '));
}

// ============================================
// 3. Адаптивное вычисление
// ============================================
console.log('\n[3] Адаптивное количество итераций');
console.log('-'.repeat(40));

// Задачи разной сложности требуют разное число итераций
const adaptiveTRM = createTinyTRM(4, 2, 32, 8);

// Простая задача (идентичность)
const simpleInput = new Tensor(new Float32Array([1, 0, 0, 1]), [1, 4]);
const { output: simpleOut, steps: simpleSteps } = adaptiveTRM.forwardAdaptive(simpleInput, 16, 0.001);
console.log(`  Простая задача: ${simpleSteps} итераций`);

// Сложная задача (случайный вход)
const complexInput = new Tensor(new Float32Array([0.5, 0.3, 0.7, 0.2]), [1, 4]);
const { output: complexOut, steps: complexSteps } = adaptiveTRM.forwardAdaptive(complexInput, 16, 0.001);
console.log(`  Сложная задача: ${complexSteps} итераций`);

// ============================================
// 4. Сравнение разного числа рекурсий
// ============================================
console.log('\n[4] Влияние числа рекурсий на качество');
console.log('-'.repeat(40));

// Задача: арифметика (a + b * c)
function generateArithmetic(n: number): { x: Tensor; y: Tensor } {
  const xData = new Float32Array(n * 3);
  const yData = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const a = Math.random() * 10;
    const b = Math.random() * 10;
    const c = Math.random() * 10;

    xData[i * 3] = a / 10;      // normalize
    xData[i * 3 + 1] = b / 10;
    xData[i * 3 + 2] = c / 10;
    yData[i] = (a + b * c) / 110;  // normalize output
  }

  return {
    x: new Tensor(xData, [n, 3], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
  };
}

const arithmeticTrain = generateArithmetic(100);

for (const numRecursions of [1, 2, 4]) {
  const model = createTinyTRM(3, 1, 32, numRecursions);
  const opt = new Adam(model.parameters(), 0.01);

  // Quick training
  for (let epoch = 0; epoch < 20; epoch++) {
    const out = model.forward(arithmeticTrain.x);
    const loss = out.sub(arithmeticTrain.y).pow(2).mean();
    opt.zeroGrad();
    loss.backward();
    opt.step();
  }

  // Test
  const testData = generateArithmetic(30);
  const pred = model.forward(testData.x);
  const mse = pred.sub(testData.y).pow(2).mean().item();

  console.log(`  Рекурсий: ${numRecursions}, MSE: ${mse.toFixed(6)}`);
}

// ============================================
// 5. История уточнений
// ============================================
console.log('\n[5] Визуализация процесса уточнения');
console.log('-'.repeat(40));

const historyModel = createTinyTRM(2, 2, 16, 4);
const historyInput = new Tensor(new Float32Array([0.5, 0.5]), [1, 2]);

const { output, intermediates } = historyModel.forwardWithHistory(historyInput);

console.log('Эволюция ответа y через итерации:');
for (let i = 0; i < intermediates.length; i++) {
  const y = intermediates[i].y;
  const yValues = Array.from(y.data).slice(0, 2).map(v => v.toFixed(3));
  console.log(`  Итерация ${i}: y = [${yValues.join(', ')}]`);
}
console.log(`  Финальный выход: [${Array.from(output.data).map(v => v.toFixed(3)).join(', ')}]`);

// ============================================
// 6. Few-shot классификация
// ============================================
console.log('\n[6] Few-shot классификация');
console.log('-'.repeat(40));

const fewShotClassifier = new TRMClassifier(4, 32, 3, 4);

// Support set: 2 examples per class
const supportX = new Tensor(new Float32Array([
  // Class 0
  1, 0, 0, 0,
  0.9, 0.1, 0, 0,
  // Class 1
  0, 1, 0, 0,
  0, 0.9, 0.1, 0,
  // Class 2
  0, 0, 1, 0,
  0, 0, 0.9, 0.1,
]), [6, 4]);
const supportY = [0, 0, 1, 1, 2, 2];

// Query set
const queryX = new Tensor(new Float32Array([
  0.8, 0.2, 0, 0,    // Should be class 0
  0.1, 0.8, 0.1, 0,  // Should be class 1
  0, 0.1, 0.8, 0.1,  // Should be class 2
]), [3, 4]);
const expectedClasses = [0, 1, 2];

const predictions = fewShotClassifier.fewShotPredict(supportX, supportY, queryX);

console.log('Few-shot предсказания:');
for (let i = 0; i < predictions.length; i++) {
  const correct = predictions[i] === expectedClasses[i] ? 'V' : 'X';
  console.log(`  Query ${i + 1}: Предсказано ${predictions[i]}, Ожидается ${expectedClasses[i]} [${correct}]`);
}

// ============================================
// Архитектура модели
// ============================================
console.log('\n[7] Архитектура TRM');
console.log('-'.repeat(40));

console.log(summary(trmModel, [1, 2]));

// ============================================
// 8. Enhanced TRM с self-attention и gating
// ============================================
console.log('\n[8] Enhanced TRM (self-attention + gating)');
console.log('-'.repeat(40));

// Сравнение базового и улучшенного TRM
const enhancedModel = createEnhancedTRM(2, 1, {
  hiddenDim: 32,
  numRecursions: 4,
  useSelfAttention: true,
  useGating: true,
});

const basicModel = createTinyTRM(2, 1, 32, 4);

console.log('Базовый TRM vs Enhanced TRM на XOR:');

// Обучаем обе модели
const enhancedOpt = new Adam(enhancedModel.parameters(), 0.01);
const basicOpt = new Adam(basicModel.parameters(), 0.01);

const trainXOR = generateNoisyXOR(100, 0.2);

let enhancedLoss = 0;
let basicLoss = 0;

for (let epoch = 0; epoch < 30; epoch++) {
  // Enhanced
  const enhOut = enhancedModel.forward(trainXOR.x);
  const enhL = enhOut.sub(trainXOR.y).pow(2).mean();
  enhancedOpt.zeroGrad();
  enhL.backward();
  enhancedOpt.step();
  enhancedLoss = enhL.item();

  // Basic
  const basOut = basicModel.forward(trainXOR.x);
  const basL = basOut.sub(trainXOR.y).pow(2).mean();
  basicOpt.zeroGrad();
  basL.backward();
  basicOpt.step();
  basicLoss = basL.item();
}

console.log(`  Basic TRM Loss: ${basicLoss.toFixed(4)}`);
console.log(`  Enhanced TRM Loss: ${enhancedLoss.toFixed(4)}`);

// Тест
const testXOR = generateNoisyXOR(50, 0.2);
const enhPred = enhancedModel.forward(testXOR.x);
const basPred = basicModel.forward(testXOR.x);

let enhCorrect = 0, basCorrect = 0;
for (let i = 0; i < 50; i++) {
  if ((enhPred.data[i] > 0.5 ? 1 : 0) === testXOR.y.data[i]) enhCorrect++;
  if ((basPred.data[i] > 0.5 ? 1 : 0) === testXOR.y.data[i]) basCorrect++;
}

console.log(`  Basic TRM Accuracy: ${(basCorrect / 50 * 100).toFixed(1)}%`);
console.log(`  Enhanced TRM Accuracy: ${(enhCorrect / 50 * 100).toFixed(1)}%`);

// ============================================
// 9. Multi-scale рекурсии
// ============================================
console.log('\n[9] Multi-scale рекурсии');
console.log('-'.repeat(40));

const multiScaleModel = createEnhancedTRM(3, 1, { hiddenDim: 32, numRecursions: 8 });
const multiScaleInput = new Tensor(new Float32Array([0.5, 0.3, 0.7]), [1, 3]);

console.log('Выход при разном числе шагов:');
for (const steps of [2, 4, 6, 8]) {
  const out = multiScaleModel.forward(multiScaleInput, steps);
  console.log(`  ${steps} шагов: ${out.data[0].toFixed(4)}`);
}

const multiScaleOut = multiScaleModel.forwardMultiScale(multiScaleInput, [2, 4, 8]);
console.log(`  Multi-scale (avg): ${multiScaleOut.data[0].toFixed(4)}`);

// ============================================
// 10. Конфигурация Enhanced TRM
// ============================================
console.log('\n[10] Конфигурация моделей');
console.log('-'.repeat(40));

const configs = [
  { name: 'Tiny TRM', model: createTinyTRM(4, 2, 32, 4) },
  { name: 'Reasoning TRM', model: createReasoningTRM(4, 2) },
  { name: 'Enhanced TRM', model: createEnhancedTRM(4, 2, { useSelfAttention: true, useGating: true }) },
];

for (const { name, model } of configs) {
  const config = model.getConfig();
  console.log(`  ${name}:`);
  console.log(`    Hidden: ${config.hiddenDim}, Recursions: ${config.numRecursions}`);
  console.log(`    Self-attention: ${config.useSelfAttention}, Gating: ${config.useGating}`);
}

console.log('\n' + '='.repeat(60));
console.log('TRM Enhanced демо завершено!');
console.log('='.repeat(60));
