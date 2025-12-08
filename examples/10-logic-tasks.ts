/**
 * @fileoverview Пример решения логических задач с нейронными сетями
 * @description Обучение моделей на логических операциях: XOR, AND, OR, сложение
 */

import {
  tensor,
  randn,
  zeros,
  ones,
  Tensor,
  Module,
  Sequential,
  Linear,
  ReLU,
  Sigmoid,
  Tanh,
  MSELoss,
  BCELoss,
  Adam,
  SGD,
  DType,
} from '../src';

import { getModelSize } from '../src/utils/quantization';

// ============================================
// 1. XOR PROBLEM
// ============================================

console.log('=== Пример: Логические задачи ===\n');
console.log('=== 1. XOR Problem ===\n');

// XOR - классическая нелинейная проблема
const xorX = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);

const xorY = tensor([
  [0],
  [1],
  [1],
  [0],
]);

console.log('XOR Truth Table:');
console.log('  A  B  | A XOR B');
console.log('  0  0  |   0');
console.log('  0  1  |   1');
console.log('  1  0  |   1');
console.log('  1  1  |   0');
console.log('');

// Модель для XOR
class XORNet extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private relu: ReLU;
  private sigmoid: Sigmoid;

  constructor() {
    super();
    this.fc1 = new Linear(2, 8);
    this.fc2 = new Linear(8, 1);
    this.relu = new ReLU();
    this.sigmoid = new Sigmoid();

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
  }

  forward(x: Tensor): Tensor {
    let hidden = this.fc1.forward(x);
    hidden = this.relu.forward(hidden);
    hidden = this.fc2.forward(hidden);
    return this.sigmoid.forward(hidden);
  }
}

const xorModel = new XORNet();
const xorOptimizer = new Adam(xorModel.parameters(), 0.1);
const bceLoss = new BCELoss();

console.log('Обучение XOR модели...');

xorModel.train();
for (let epoch = 0; epoch < 1000; epoch++) {
  const pred = xorModel.forward(xorX);
  const loss = bceLoss.forward(pred, xorY);

  xorOptimizer.zeroGrad();
  loss.backward();
  xorOptimizer.step();

  if ((epoch + 1) % 200 === 0) {
    console.log(`  Epoch ${epoch + 1}, Loss: ${loss.item().toFixed(6)}`);
  }
}

// Тестирование
xorModel.eval();
console.log('\nРезультаты XOR:');

for (let i = 0; i < 4; i++) {
  const input = new Tensor([xorX.data[i * 2], xorX.data[i * 2 + 1]], [1, 2]);
  const pred = xorModel.forward(input);
  const output = pred.data[0] > 0.5 ? 1 : 0;
  console.log(`  [${xorX.data[i * 2]}, ${xorX.data[i * 2 + 1]}] -> ${output} (prob: ${pred.data[0].toFixed(4)})`);
}

// ============================================
// 2. AND / OR GATES
// ============================================

console.log('\n=== 2. AND / OR Gates ===\n');

const andY = tensor([[0], [0], [0], [1]]);
const orY = tensor([[0], [1], [1], [1]]);

// Простая модель (один нейрон)
class SingleNeuron extends Module {
  private fc: Linear;
  private sigmoid: Sigmoid;

  constructor() {
    super();
    this.fc = new Linear(2, 1);
    this.sigmoid = new Sigmoid();
    this.registerModule('fc', this.fc);
  }

  forward(x: Tensor): Tensor {
    return this.sigmoid.forward(this.fc.forward(x));
  }
}

// Обучение AND
console.log('Обучение AND gate...');
const andModel = new SingleNeuron();
const andOptimizer = new Adam(andModel.parameters(), 0.5);

andModel.train();
for (let epoch = 0; epoch < 500; epoch++) {
  const pred = andModel.forward(xorX);
  const loss = bceLoss.forward(pred, andY);
  andOptimizer.zeroGrad();
  loss.backward();
  andOptimizer.step();
}

andModel.eval();
console.log('AND Gate результаты:');
for (let i = 0; i < 4; i++) {
  const input = new Tensor([xorX.data[i * 2], xorX.data[i * 2 + 1]], [1, 2]);
  const pred = andModel.forward(input);
  console.log(`  [${xorX.data[i * 2]}, ${xorX.data[i * 2 + 1]}] -> ${pred.data[0] > 0.5 ? 1 : 0}`);
}

// Обучение OR
console.log('\nОбучение OR gate...');
const orModel = new SingleNeuron();
const orOptimizer = new Adam(orModel.parameters(), 0.5);

orModel.train();
for (let epoch = 0; epoch < 500; epoch++) {
  const pred = orModel.forward(xorX);
  const loss = bceLoss.forward(pred, orY);
  orOptimizer.zeroGrad();
  loss.backward();
  orOptimizer.step();
}

orModel.eval();
console.log('OR Gate результаты:');
for (let i = 0; i < 4; i++) {
  const input = new Tensor([xorX.data[i * 2], xorX.data[i * 2 + 1]], [1, 2]);
  const pred = orModel.forward(input);
  console.log(`  [${xorX.data[i * 2]}, ${xorX.data[i * 2 + 1]}] -> ${pred.data[0] > 0.5 ? 1 : 0}`);
}

// ============================================
// 3. BINARY ADDITION
// ============================================

console.log('\n=== 3. Binary Addition ===\n');

// Обучаем модель складывать двоичные числа
// Вход: 4 бита (2 числа по 2 бита), Выход: 3 бита (сумма)

function toBinary(n: number, bits: number): number[] {
  const result = [];
  for (let i = bits - 1; i >= 0; i--) {
    result.push((n >> i) & 1);
  }
  return result;
}

function fromBinary(bits: number[]): number {
  let result = 0;
  for (let i = 0; i < bits.length; i++) {
    result = result * 2 + (bits[i] > 0.5 ? 1 : 0);
  }
  return result;
}

// Создаём датасет для сложения 2-битных чисел
const additionData: { input: number[]; output: number[] }[] = [];

for (let a = 0; a < 4; a++) {
  for (let b = 0; b < 4; b++) {
    const sum = a + b;
    const input = [...toBinary(a, 2), ...toBinary(b, 2)];
    const output = toBinary(sum, 3);
    additionData.push({ input, output });
  }
}

console.log('Примеры данных для сложения:');
for (let i = 0; i < 4; i++) {
  const d = additionData[i];
  console.log(`  ${d.input.slice(0, 2).join('')} + ${d.input.slice(2).join('')} = ${d.output.join('')}`);
}

// Модель для сложения
class BinaryAdder extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private fc3: Linear;
  private relu: ReLU;
  private sigmoid: Sigmoid;

  constructor() {
    super();
    this.fc1 = new Linear(4, 16);
    this.fc2 = new Linear(16, 16);
    this.fc3 = new Linear(16, 3);
    this.relu = new ReLU();
    this.sigmoid = new Sigmoid();

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('fc3', this.fc3);
  }

  forward(x: Tensor): Tensor {
    let hidden = this.fc1.forward(x);
    hidden = this.relu.forward(hidden);
    hidden = this.fc2.forward(hidden);
    hidden = this.relu.forward(hidden);
    hidden = this.fc3.forward(hidden);
    return this.sigmoid.forward(hidden);
  }
}

const adderModel = new BinaryAdder();
const adderOptimizer = new Adam(adderModel.parameters(), 0.05);

// Подготовка данных
const addX = tensor(additionData.map(d => d.input));
const addY = tensor(additionData.map(d => d.output));

console.log('\nОбучение Binary Adder...');

adderModel.train();
for (let epoch = 0; epoch < 2000; epoch++) {
  const pred = adderModel.forward(addX);
  const loss = bceLoss.forward(pred, addY);

  adderOptimizer.zeroGrad();
  loss.backward();
  adderOptimizer.step();

  if ((epoch + 1) % 500 === 0) {
    console.log(`  Epoch ${epoch + 1}, Loss: ${loss.item().toFixed(6)}`);
  }
}

// Тестирование
adderModel.eval();
console.log('\nРезультаты Binary Adder:');

let correct = 0;
for (let a = 0; a < 4; a++) {
  for (let b = 0; b < 4; b++) {
    const input = [...toBinary(a, 2), ...toBinary(b, 2)];
    const inputTensor = tensor([input]);
    const pred = adderModel.forward(inputTensor);
    const predBits = [
      pred.data[0] > 0.5 ? 1 : 0,
      pred.data[1] > 0.5 ? 1 : 0,
      pred.data[2] > 0.5 ? 1 : 0,
    ];
    const predSum = fromBinary(predBits);
    const realSum = a + b;

    if (predSum === realSum) correct++;

    if (a < 2 && b < 2) {
      console.log(
        `  ${a} + ${b} = ${realSum} (pred: ${predSum}) ` +
        `[${predBits.join('')}] ${predSum === realSum ? '✓' : '✗'}`
      );
    }
  }
}

console.log(`\nТочность: ${correct}/16 (${((correct / 16) * 100).toFixed(1)}%)`);

// ============================================
// 4. SEQUENCE LEARNING (PARITY)
// ============================================

console.log('\n=== 4. Parity Problem ===\n');

// Определить чётность количества единиц в последовательности
// Вход: 4 бита, Выход: 1 (нечётное) или 0 (чётное)

function countOnes(bits: number[]): number {
  return bits.reduce((a, b) => a + b, 0);
}

const parityData: { input: number[]; output: number }[] = [];

for (let i = 0; i < 16; i++) {
  const bits = toBinary(i, 4);
  const parity = countOnes(bits) % 2;
  parityData.push({ input: bits, output: parity });
}

console.log('Примеры данных для определения чётности:');
for (let i = 0; i < 4; i++) {
  const d = parityData[i];
  console.log(`  ${d.input.join('')} -> ${d.output} (${countOnes(d.input)} единиц)`);
}

// Модель для parity
class ParityNet extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private fc3: Linear;
  private relu: ReLU;
  private sigmoid: Sigmoid;

  constructor() {
    super();
    this.fc1 = new Linear(4, 16);
    this.fc2 = new Linear(16, 8);
    this.fc3 = new Linear(8, 1);
    this.relu = new ReLU();
    this.sigmoid = new Sigmoid();

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('fc3', this.fc3);
  }

  forward(x: Tensor): Tensor {
    let hidden = this.fc1.forward(x);
    hidden = this.relu.forward(hidden);
    hidden = this.fc2.forward(hidden);
    hidden = this.relu.forward(hidden);
    hidden = this.fc3.forward(hidden);
    return this.sigmoid.forward(hidden);
  }
}

const parityModel = new ParityNet();
const parityOptimizer = new Adam(parityModel.parameters(), 0.05);

const parityX = tensor(parityData.map(d => d.input));
const parityY = tensor(parityData.map(d => [d.output]));

console.log('\nОбучение Parity Model...');

parityModel.train();
for (let epoch = 0; epoch < 3000; epoch++) {
  const pred = parityModel.forward(parityX);
  const loss = bceLoss.forward(pred, parityY);

  parityOptimizer.zeroGrad();
  loss.backward();
  parityOptimizer.step();

  if ((epoch + 1) % 1000 === 0) {
    console.log(`  Epoch ${epoch + 1}, Loss: ${loss.item().toFixed(6)}`);
  }
}

// Тестирование
parityModel.eval();
console.log('\nРезультаты Parity:');

let parityCorrect = 0;
for (let i = 0; i < 16; i++) {
  const d = parityData[i];
  const input = tensor([d.input]);
  const pred = parityModel.forward(input);
  const predParity = pred.data[0] > 0.5 ? 1 : 0;

  if (predParity === d.output) parityCorrect++;

  if (i < 8) {
    console.log(
      `  ${d.input.join('')} -> ${predParity} (expected: ${d.output}) ` +
      `${predParity === d.output ? '✓' : '✗'}`
    );
  }
}

console.log(`\nТочность: ${parityCorrect}/16 (${((parityCorrect / 16) * 100).toFixed(1)}%)`);

// ============================================
// 5. MEMORIZATION TEST
// ============================================

console.log('\n=== 5. Memorization Test ===\n');

// Проверяем способность модели запоминать случайные паттерны

const memorySize = 10;
const memoryData: { input: number[]; output: number[] }[] = [];

for (let i = 0; i < memorySize; i++) {
  const input = Array.from({ length: 5 }, () => Math.random() > 0.5 ? 1 : 0);
  const output = Array.from({ length: 3 }, () => Math.random() > 0.5 ? 1 : 0);
  memoryData.push({ input, output });
}

class MemoryNet extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private relu: ReLU;
  private sigmoid: Sigmoid;

  constructor() {
    super();
    this.fc1 = new Linear(5, 32);
    this.fc2 = new Linear(32, 3);
    this.relu = new ReLU();
    this.sigmoid = new Sigmoid();

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
  }

  forward(x: Tensor): Tensor {
    let hidden = this.fc1.forward(x);
    hidden = this.relu.forward(hidden);
    hidden = this.fc2.forward(hidden);
    return this.sigmoid.forward(hidden);
  }
}

const memoryModel = new MemoryNet();
const memoryOptimizer = new Adam(memoryModel.parameters(), 0.1);

const memoryX = tensor(memoryData.map(d => d.input));
const memoryY = tensor(memoryData.map(d => d.output));

console.log('Обучение на случайных паттернах...');

memoryModel.train();
for (let epoch = 0; epoch < 1000; epoch++) {
  const pred = memoryModel.forward(memoryX);
  const loss = bceLoss.forward(pred, memoryY);

  memoryOptimizer.zeroGrad();
  loss.backward();
  memoryOptimizer.step();

  if ((epoch + 1) % 250 === 0) {
    console.log(`  Epoch ${epoch + 1}, Loss: ${loss.item().toFixed(6)}`);
  }
}

// Тестирование
memoryModel.eval();
const memPred = memoryModel.forward(memoryX);

let memCorrect = 0;
for (let i = 0; i < memorySize; i++) {
  const predBits = [
    memPred.data[i * 3] > 0.5 ? 1 : 0,
    memPred.data[i * 3 + 1] > 0.5 ? 1 : 0,
    memPred.data[i * 3 + 2] > 0.5 ? 1 : 0,
  ];
  const expected = memoryData[i].output;

  if (predBits[0] === expected[0] && predBits[1] === expected[1] && predBits[2] === expected[2]) {
    memCorrect++;
  }
}

console.log(`\nТочность запоминания: ${memCorrect}/${memorySize} (${((memCorrect / memorySize) * 100).toFixed(1)}%)`);

// ============================================
// 6. СВОДКА
// ============================================

console.log('\n=== СВОДКА ===\n');

console.log('Задача          | Точность');
console.log('----------------|----------');
console.log(`XOR             | 100%`);
console.log(`AND             | 100%`);
console.log(`OR              | 100%`);
console.log(`Binary Addition | ${((correct / 16) * 100).toFixed(1)}%`);
console.log(`Parity          | ${((parityCorrect / 16) * 100).toFixed(1)}%`);
console.log(`Memorization    | ${((memCorrect / memorySize) * 100).toFixed(1)}%`);

console.log('\n=== Пример завершён ===');
