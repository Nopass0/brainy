/**
 * @fileoverview TRM для математических рассуждений
 * @description Демонстрация TRM на задачах арифметики и логики
 *
 * TRM особенно хорошо подходит для математических задач,
 * где требуется пошаговое вычисление и рассуждение.
 */

import {
  Tensor,
  tensor,
  TRM,
  createTinyTRM,
  createMathTRM,
  createEnhancedTRM,
  createPonderingTRM,
  initTRMGPU,
  isTRMGPUAvailable,
  Adam,
  MSELoss,
} from '../src';

console.log('='.repeat(60));
console.log('TRM для математических рассуждений');
console.log('='.repeat(60));

// GPU initialization
initTRMGPU().catch(() => {});
console.log(`Устройство: ${isTRMGPUAvailable() ? 'GPU' : 'CPU'}`);

// ============================================
// 1. Простая арифметика: a + b
// ============================================
console.log('\n[1] Простое сложение: a + b');
console.log('-'.repeat(40));

function generateAddition(n: number, maxVal: number = 10): {
  x: Tensor;
  y: Tensor;
  pairs: [number, number][];
} {
  const xData = new Float32Array(n * 2);
  const yData = new Float32Array(n);
  const pairs: [number, number][] = [];

  for (let i = 0; i < n; i++) {
    const a = Math.floor(Math.random() * maxVal);
    const b = Math.floor(Math.random() * maxVal);
    pairs.push([a, b]);

    // Normalize inputs
    xData[i * 2] = a / maxVal;
    xData[i * 2 + 1] = b / maxVal;
    yData[i] = (a + b) / (maxVal * 2);  // Normalize output
  }

  return {
    x: new Tensor(xData, [n, 2], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    pairs,
  };
}

const addModel = createTinyTRM(2, 1, 32, 3);  // Faster basic model
const addOpt = new Adam(addModel.parameters(), 0.01);

const addTrain = generateAddition(100);  // Fewer samples

console.log('Обучение на сложении...');
for (let epoch = 0; epoch < 30; epoch++) {  // Fewer epochs
  const out = addModel.forward(addTrain.x);
  const loss = out.sub(addTrain.y).pow(2).mean();

  addOpt.zeroGrad();
  loss.backward();
  addOpt.step();

  if ((epoch + 1) % 10 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${loss.item().toFixed(6)}`);
  }
}

// Test
console.log('\nТест сложения:');
const addTest = generateAddition(5);
const addPred = addModel.forward(addTest.x);

for (let i = 0; i < 5; i++) {
  const [a, b] = addTest.pairs[i];
  const predicted = Math.round(addPred.data[i] * 20);  // Denormalize
  const actual = a + b;
  const correct = predicted === actual ? 'V' : 'X';
  console.log(`  ${a} + ${b} = ${predicted} (ожидается ${actual}) [${correct}]`);
}

// ============================================
// 2. Сложная арифметика: a * b + c
// ============================================
console.log('\n[2] Комплексная арифметика: a * b + c');
console.log('-'.repeat(40));

function generateComplex(n: number, maxVal: number = 5): {
  x: Tensor;
  y: Tensor;
  expressions: string[];
} {
  const xData = new Float32Array(n * 3);
  const yData = new Float32Array(n);
  const expressions: string[] = [];

  for (let i = 0; i < n; i++) {
    const a = Math.floor(Math.random() * maxVal) + 1;
    const b = Math.floor(Math.random() * maxVal) + 1;
    const c = Math.floor(Math.random() * maxVal);
    const result = a * b + c;

    expressions.push(`${a} * ${b} + ${c} = ${result}`);

    // Normalize
    xData[i * 3] = a / maxVal;
    xData[i * 3 + 1] = b / maxVal;
    xData[i * 3 + 2] = c / maxVal;
    yData[i] = result / (maxVal * maxVal + maxVal);
  }

  return {
    x: new Tensor(xData, [n, 3], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    expressions,
  };
}

// Use TinyTRM for faster training
const mathModel = createTinyTRM(3, 1, 48, 4);
const mathOpt = new Adam(mathModel.parameters(), 0.01);

const mathTrain = generateComplex(150);

console.log('Обучение на a * b + c...');
for (let epoch = 0; epoch < 50; epoch++) {
  const out = mathModel.forward(mathTrain.x);
  const loss = out.sub(mathTrain.y).pow(2).mean();

  mathOpt.zeroGrad();
  loss.backward();
  mathOpt.step();

  if ((epoch + 1) % 20 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${loss.item().toFixed(6)}`);
  }
}

// Test
console.log('\nТест комплексной арифметики:');
const mathTest = generateComplex(5);
const mathPred = mathModel.forward(mathTest.x);

for (let i = 0; i < 5; i++) {
  const predicted = Math.round(mathPred.data[i] * 30);  // Denormalize
  const expr = mathTest.expressions[i];
  const actual = parseInt(expr.split(' = ')[1]);
  const correct = Math.abs(predicted - actual) <= 1 ? 'V' : 'X';  // Allow ±1 error
  console.log(`  ${expr.split(' = ')[0]} = ${predicted} (ожидается ${actual}) [${correct}]`);
}

// ============================================
// 3. Последовательная арифметика
// ============================================
console.log('\n[3] Последовательность операций');
console.log('-'.repeat(40));

function generateSequence(n: number): {
  x: Tensor;
  y: Tensor;
  sequences: string[];
} {
  const xData = new Float32Array(n * 4);
  const yData = new Float32Array(n);
  const sequences: string[] = [];

  for (let i = 0; i < n; i++) {
    const a = Math.floor(Math.random() * 5) + 1;
    const b = Math.floor(Math.random() * 3) + 1;
    const c = Math.floor(Math.random() * 3) + 1;
    const d = Math.floor(Math.random() * 3) + 1;

    // ((a + b) * c) - d
    const step1 = a + b;
    const step2 = step1 * c;
    const result = step2 - d;

    sequences.push(`((${a} + ${b}) * ${c}) - ${d} = ${result}`);

    // Normalize
    xData[i * 4] = a / 5;
    xData[i * 4 + 1] = b / 3;
    xData[i * 4 + 2] = c / 3;
    xData[i * 4 + 3] = d / 3;
    yData[i] = (result + 10) / 50;  // Normalize to handle negative results
  }

  return {
    x: new Tensor(xData, [n, 4], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    sequences,
  };
}

const seqModel = createTinyTRM(4, 1, 48, 4);
const seqOpt = new Adam(seqModel.parameters(), 0.01);

const seqTrain = generateSequence(150);

console.log('Обучение на последовательных операциях...');
for (let epoch = 0; epoch < 40; epoch++) {
  const out = seqModel.forward(seqTrain.x);
  const loss = out.sub(seqTrain.y).pow(2).mean();

  seqOpt.zeroGrad();
  loss.backward();
  seqOpt.step();

  if ((epoch + 1) % 20 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${loss.item().toFixed(6)}`);
  }
}

// Test
console.log('\nТест последовательностей:');
const seqTest = generateSequence(5);
const seqPred = seqModel.forward(seqTest.x);

for (let i = 0; i < 5; i++) {
  const predicted = Math.round(seqPred.data[i] * 50 - 10);  // Denormalize
  const expr = seqTest.sequences[i];
  const actual = parseInt(expr.split(' = ')[1]);
  const correct = Math.abs(predicted - actual) <= 2 ? 'V' : 'X';  // Allow ±2 error
  console.log(`  ${expr.split(' = ')[0]} = ${predicted} (ожидается ${actual}) [${correct}]`);
}

// ============================================
// 4. Сравнение чисел
// ============================================
console.log('\n[4] Сравнение чисел (a > b?)');
console.log('-'.repeat(40));

function generateComparison(n: number): {
  x: Tensor;
  y: Tensor;
  pairs: [number, number, boolean][];
} {
  const xData = new Float32Array(n * 2);
  const yData = new Float32Array(n);
  const pairs: [number, number, boolean][] = [];

  for (let i = 0; i < n; i++) {
    const a = Math.random() * 10;
    const b = Math.random() * 10;
    const isGreater = a > b;

    pairs.push([Math.round(a * 10) / 10, Math.round(b * 10) / 10, isGreater]);

    xData[i * 2] = a / 10;
    xData[i * 2 + 1] = b / 10;
    yData[i] = isGreater ? 1 : 0;
  }

  return {
    x: new Tensor(xData, [n, 2], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    pairs,
  };
}

const compModel = createTinyTRM(2, 1, 24, 3);
const compOpt = new Adam(compModel.parameters(), 0.02);

const compTrain = generateComparison(100);

console.log('Обучение сравнению...');
for (let epoch = 0; epoch < 20; epoch++) {
  const out = compModel.forward(compTrain.x);
  const loss = out.sub(compTrain.y).pow(2).mean();

  compOpt.zeroGrad();
  loss.backward();
  compOpt.step();
}

// Test
console.log('Тест сравнения:');
const compTest = generateComparison(6);
const compPred = compModel.forward(compTest.x);

let compCorrect = 0;
for (let i = 0; i < 6; i++) {
  const [a, b, actual] = compTest.pairs[i];
  const predicted = compPred.data[i] > 0.5;
  const correct = predicted === actual;
  if (correct) compCorrect++;
  console.log(`  ${a} > ${b}? Предсказано: ${predicted ? 'Да' : 'Нет'}, Ожидается: ${actual ? 'Да' : 'Нет'} [${correct ? 'V' : 'X'}]`);
}
console.log(`  Точность: ${(compCorrect / 6 * 100).toFixed(1)}%`);

// ============================================
// 5. Простая логика (AND/OR/XOR)
// ============================================
console.log('\n[5] Логические операции');
console.log('-'.repeat(40));

function generateLogic(n: number, op: 'and' | 'or' | 'xor'): {
  x: Tensor;
  y: Tensor;
} {
  const xData = new Float32Array(n * 2);
  const yData = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;

    let result: number;
    switch (op) {
      case 'and': result = a && b ? 1 : 0; break;
      case 'or': result = a || b ? 1 : 0; break;
      case 'xor': result = a !== b ? 1 : 0; break;
    }

    xData[i * 2] = a;
    xData[i * 2 + 1] = b;
    yData[i] = result;
  }

  return {
    x: new Tensor(xData, [n, 2], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
  };
}

for (const op of ['and', 'or', 'xor'] as const) {
  const logicModel = createTinyTRM(2, 1, 16, 3);
  const logicOpt = new Adam(logicModel.parameters(), 0.03);

  const logicTrain = generateLogic(60, op);

  // Quick training
  for (let epoch = 0; epoch < 25; epoch++) {
    const out = logicModel.forward(logicTrain.x);
    const loss = out.sub(logicTrain.y).pow(2).mean();

    logicOpt.zeroGrad();
    loss.backward();
    logicOpt.step();
  }

  // Test
  const logicTest = generateLogic(50, op);
  const logicPred = logicModel.forward(logicTest.x);

  let logicCorrect = 0;
  for (let i = 0; i < 50; i++) {
    const predicted = logicPred.data[i] > 0.5 ? 1 : 0;
    if (predicted === logicTest.y.data[i]) logicCorrect++;
  }

  console.log(`  ${op.toUpperCase()}: ${(logicCorrect / 50 * 100).toFixed(0)}% точность`);
}

// ============================================
// 6. Числовые паттерны (следующее число)
// ============================================
console.log('\n[6] Паттерны последовательностей');
console.log('-'.repeat(40));

function generatePattern(n: number, type: 'linear' | 'quadratic'): {
  x: Tensor;
  y: Tensor;
  sequences: number[][];
} {
  const xData = new Float32Array(n * 3);
  const yData = new Float32Array(n);
  const sequences: number[][] = [];

  for (let i = 0; i < n; i++) {
    const start = Math.floor(Math.random() * 5) + 1;
    let seq: number[];

    if (type === 'linear') {
      const diff = Math.floor(Math.random() * 3) + 1;
      seq = [start, start + diff, start + 2 * diff, start + 3 * diff];
    } else {
      // Quadratic: n^2 based
      seq = [start, start + 1, start + 4, start + 9];
    }

    sequences.push(seq);

    // Input: first 3 numbers, predict 4th
    xData[i * 3] = seq[0] / 20;
    xData[i * 3 + 1] = seq[1] / 20;
    xData[i * 3 + 2] = seq[2] / 20;
    yData[i] = seq[3] / 30;
  }

  return {
    x: new Tensor(xData, [n, 3], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
    sequences,
  };
}

// Linear patterns
const linearModel = createTinyTRM(3, 1, 32, 4);
const linearOpt = new Adam(linearModel.parameters(), 0.01);
const linearTrain = generatePattern(100, 'linear');

console.log('Линейные последовательности (a, a+d, a+2d, ?)...');
for (let epoch = 0; epoch < 30; epoch++) {
  const out = linearModel.forward(linearTrain.x);
  const loss = out.sub(linearTrain.y).pow(2).mean();

  linearOpt.zeroGrad();
  loss.backward();
  linearOpt.step();
}

const linearTest = generatePattern(4, 'linear');
const linearPred = linearModel.forward(linearTest.x);

for (let i = 0; i < 4; i++) {
  const seq = linearTest.sequences[i];
  const predicted = Math.round(linearPred.data[i] * 30);
  const correct = Math.abs(predicted - seq[3]) <= 1 ? 'V' : 'X';
  console.log(`  [${seq.slice(0, 3).join(', ')}, ?] -> ${predicted} (ожидается ${seq[3]}) [${correct}]`);
}

// ============================================
// 7. Адаптивная сложность
// ============================================
console.log('\n[7] Адаптивное количество итераций по сложности');
console.log('-'.repeat(40));

const adaptiveMath = createTinyTRM(3, 1, 32, 4);

// Simple problem
const simpleProb = new Tensor(new Float32Array([0.2, 0.2, 0]), [1, 3]);
const { output: simpleOut, steps: simpleSteps } = adaptiveMath.forwardAdaptive(simpleProb, 16, 0.001);
console.log(`  Простая задача (2+2+0): ${simpleSteps} итераций`);

// Medium problem
const mediumProb = new Tensor(new Float32Array([0.5, 0.5, 0.5]), [1, 3]);
const { output: mediumOut, steps: mediumSteps } = adaptiveMath.forwardAdaptive(mediumProb, 16, 0.001);
console.log(`  Средняя задача (5*5+5): ${mediumSteps} итераций`);

// Complex problem
const complexProb = new Tensor(new Float32Array([0.9, 0.9, 0.9]), [1, 3]);
const { output: complexOut, steps: complexSteps } = adaptiveMath.forwardAdaptive(complexProb, 16, 0.001);
console.log(`  Сложная задача (9*9+9): ${complexSteps} итераций`);

// ============================================
// 8. История вычислений
// ============================================
console.log('\n[8] Визуализация пошагового вычисления');
console.log('-'.repeat(40));

const histModel = createTinyTRM(2, 1, 16, 4);
const histInput = new Tensor(new Float32Array([0.7, 0.3]), [1, 2]);

const { output: histOut, intermediates } = histModel.forwardWithHistory(histInput);

console.log('Эволюция ответа (0.7 + 0.3):');
for (let i = 0; i < intermediates.length; i++) {
  const yNorm = intermediates[i].y.data[0];
  const bar = '#'.repeat(Math.max(0, Math.min(20, Math.round(Math.abs(yNorm) * 20))));
  console.log(`  Шаг ${i}: ${bar} (${yNorm.toFixed(4)})`);
}
console.log(`  Финал: ${histOut.data[0].toFixed(4)}`);

console.log('\n' + '='.repeat(60));
console.log('TRM Math демо завершено!');
console.log('='.repeat(60));
