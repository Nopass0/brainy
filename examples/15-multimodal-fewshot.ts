/**
 * @fileoverview Multimodal Few-Shot Learning
 * @description Мультимодальная модель для быстрого обучения на нескольких примерах
 *
 * Демонстрация:
 * - Обработка текста, изображений и числовых данных
 * - Few-shot классификация
 * - Few-shot регрессия
 * - Кросс-модальное обучение
 */

import {
  Tensor,
  tensor,
  zeros,
  MultimodalFewShot,
  createSmallMultimodal,
  Adam,
  summary,
} from '../src';

console.log('='.repeat(60));
console.log('Multimodal Few-Shot Learning');
console.log('='.repeat(60));

// ============================================
// 1. Создание модели
// ============================================
console.log('\n[1] Создание мультимодальной модели');
console.log('-'.repeat(40));

const model = createSmallMultimodal();

console.log('Модель поддерживает:');
console.log('  - Текст (токены)');
console.log('  - Изображения (8x8)');
console.log('  - Числовые последовательности');

// ============================================
// 2. Few-shot классификация с изображениями
// ============================================
console.log('\n[2] Few-shot классификация изображений');
console.log('-'.repeat(40));
console.log('Задача: распознать паттерны по 2 примерам на класс');

// Создаём простые паттерны 8x8
function createVerticalLines(): Float32Array {
  const img = new Float32Array(64);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      img[y * 8 + x] = x % 2 === 0 ? 1 : 0;
    }
  }
  return img;
}

function createHorizontalLines(): Float32Array {
  const img = new Float32Array(64);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      img[y * 8 + x] = y % 2 === 0 ? 1 : 0;
    }
  }
  return img;
}

function createDiagonal(): Float32Array {
  const img = new Float32Array(64);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      img[y * 8 + x] = (x + y) % 2 === 0 ? 1 : 0;
    }
  }
  return img;
}

// Support set: 2 примера каждого класса
const supportSet = [
  // Класс 0: вертикальные линии
  { data: { image: new Tensor(createVerticalLines(), [1, 64], { requiresGrad: true }) }, label: 0 },
  { data: { image: new Tensor(createVerticalLines().map(x => x * 0.9 + Math.random() * 0.1), [1, 64], { requiresGrad: true }) }, label: 0 },
  // Класс 1: горизонтальные линии
  { data: { image: new Tensor(createHorizontalLines(), [1, 64], { requiresGrad: true }) }, label: 1 },
  { data: { image: new Tensor(createHorizontalLines().map(x => x * 0.9 + Math.random() * 0.1), [1, 64], { requiresGrad: true }) }, label: 1 },
  // Класс 2: диагональ
  { data: { image: new Tensor(createDiagonal(), [1, 64], { requiresGrad: true }) }, label: 2 },
  { data: { image: new Tensor(createDiagonal().map(x => x * 0.9 + Math.random() * 0.1), [1, 64], { requiresGrad: true }) }, label: 2 },
];

// Query set: новые примеры (с небольшим шумом)
const querySet = [
  { image: new Tensor(createVerticalLines().map(x => x * 0.8 + Math.random() * 0.2), [1, 64]) },
  { image: new Tensor(createHorizontalLines().map(x => x * 0.8 + Math.random() * 0.2), [1, 64]) },
  { image: new Tensor(createDiagonal().map(x => x * 0.8 + Math.random() * 0.2), [1, 64]) },
];
const expectedLabels = [0, 1, 2];
const labelNames = ['Вертикальные', 'Горизонтальные', 'Диагональ'];

const predictions = model.fewShotClassify(supportSet, querySet);

console.log('\nРезультаты (2-shot classification):');
for (let i = 0; i < predictions.length; i++) {
  const correct = predictions[i] === expectedLabels[i] ? 'V' : 'X';
  console.log(`  Query ${i + 1}: Предсказано "${labelNames[predictions[i]]}", Ожидается "${labelNames[expectedLabels[i]]}" [${correct}]`);
}

const accuracy = predictions.filter((p, i) => p === expectedLabels[i]).length / predictions.length;
console.log(`\nТочность: ${(accuracy * 100).toFixed(0)}%`);

// ============================================
// 3. Few-shot регрессия
// ============================================
console.log('\n[3] Few-shot регрессия');
console.log('-'.repeat(40));
console.log('Задача: предсказать значение по паттерну');

// Support set для регрессии: яркость → значение
const regressionSupport = [
  { data: { sequence: new Tensor(new Float32Array([0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [1, 16], { requiresGrad: true }) }, value: 0.2 },
  { data: { sequence: new Tensor(new Float32Array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [1, 16], { requiresGrad: true }) }, value: 0.5 },
  { data: { sequence: new Tensor(new Float32Array([0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [1, 16], { requiresGrad: true }) }, value: 0.9 },
];

const regressionQuery = [
  { sequence: new Tensor(new Float32Array([0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [1, 16]) },
  { sequence: new Tensor(new Float32Array([0.7, 0.7, 0.7, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), [1, 16]) },
];
const expectedValues = [0.3, 0.7];

const regressionPreds = model.fewShotRegress(regressionSupport, regressionQuery);

console.log('\nРезультаты регрессии:');
for (let i = 0; i < regressionPreds.length; i++) {
  const error = Math.abs(regressionPreds[i] - expectedValues[i]);
  console.log(`  Query ${i + 1}: Предсказано ${regressionPreds[i].toFixed(3)}, Ожидается ${expectedValues[i].toFixed(3)}, Ошибка: ${error.toFixed(3)}`);
}

// ============================================
// 4. Мультимодальный ввод
// ============================================
console.log('\n[4] Мультимодальный ввод');
console.log('-'.repeat(40));
console.log('Комбинирование текста и изображений');

// Создаём примеры с обеими модальностями
const multimodalSupport = [
  // Класс 0: яркое изображение + высокие числа
  {
    data: {
      image: new Tensor(new Float32Array(64).fill(0.9), [1, 64], { requiresGrad: true }),
      sequence: new Tensor(new Float32Array([9, 8, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).map(x => x / 10), [1, 16], { requiresGrad: true }),
    },
    label: 0,
  },
  {
    data: {
      image: new Tensor(new Float32Array(64).fill(0.8), [1, 64], { requiresGrad: true }),
      sequence: new Tensor(new Float32Array([8, 7, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).map(x => x / 10), [1, 16], { requiresGrad: true }),
    },
    label: 0,
  },
  // Класс 1: тёмное изображение + низкие числа
  {
    data: {
      image: new Tensor(new Float32Array(64).fill(0.1), [1, 64], { requiresGrad: true }),
      sequence: new Tensor(new Float32Array([1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).map(x => x / 10), [1, 16], { requiresGrad: true }),
    },
    label: 1,
  },
  {
    data: {
      image: new Tensor(new Float32Array(64).fill(0.2), [1, 64], { requiresGrad: true }),
      sequence: new Tensor(new Float32Array([2, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).map(x => x / 10), [1, 16], { requiresGrad: true }),
    },
    label: 1,
  },
];

const multimodalQuery = [
  // Яркое + высокие
  {
    image: new Tensor(new Float32Array(64).fill(0.85), [1, 64]),
    sequence: new Tensor(new Float32Array([8, 9, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).map(x => x / 10), [1, 16]),
  },
  // Тёмное + низкие
  {
    image: new Tensor(new Float32Array(64).fill(0.15), [1, 64]),
    sequence: new Tensor(new Float32Array([1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).map(x => x / 10), [1, 16]),
  },
  // Смешанный случай (яркое + низкие) - должен выбрать по среднему
  {
    image: new Tensor(new Float32Array(64).fill(0.9), [1, 64]),
    sequence: new Tensor(new Float32Array([1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).map(x => x / 10), [1, 16]),
  },
];

const multiPreds = model.fewShotClassify(multimodalSupport, multimodalQuery);

console.log('\nКлассификация по двум модальностям:');
const multiLabels = ['Яркое+Высокие', 'Тёмное+Низкие'];
console.log(`  Query 1 (яркое+высокие): Класс ${multiPreds[0]} (${multiLabels[multiPreds[0]]})`);
console.log(`  Query 2 (тёмное+низкие): Класс ${multiPreds[1]} (${multiLabels[multiPreds[1]]})`);
console.log(`  Query 3 (смешанный):     Класс ${multiPreds[2]} (${multiLabels[multiPreds[2]]})`);

// ============================================
// 5. Обучение с градиентами
// ============================================
console.log('\n[5] Fine-tuning на размеченных данных');
console.log('-'.repeat(40));

const optimizer = new Adam(model.parameters(), 0.01);

// Создаём обучающие данные
const trainData: { input: { sequence: Tensor }; target: number }[] = [];
for (let i = 0; i < 30; i++) {
  const classIdx = i % 3;
  const seqData = new Float32Array(16).fill(0);

  // Разные паттерны для разных классов
  if (classIdx === 0) {
    // Возрастающая
    for (let j = 0; j < 8; j++) seqData[j] = (j + 1) / 10;
  } else if (classIdx === 1) {
    // Убывающая
    for (let j = 0; j < 8; j++) seqData[j] = (8 - j) / 10;
  } else {
    // Чередующаяся
    for (let j = 0; j < 8; j++) seqData[j] = j % 2 === 0 ? 0.8 : 0.2;
  }

  trainData.push({
    input: { sequence: new Tensor(seqData, [1, 16], { requiresGrad: true }) },
    target: classIdx,
  });
}

console.log('Обучение на 30 примерах...');
for (let epoch = 0; epoch < 20; epoch++) {
  let totalLoss = 0;

  for (const { input, target } of trainData) {
    const logits = model.classify(input);

    // Cross-entropy loss (simplified)
    const targetTensor = new Tensor(new Float32Array(10).fill(0).map((_, i) => i === target ? 1 : 0), [1, 10]);
    const loss = logits.sub(targetTensor).pow(2).mean();

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    totalLoss += loss.item();
  }

  if ((epoch + 1) % 5 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${(totalLoss / trainData.length).toFixed(4)}`);
  }
}

// ============================================
// 6. Тестирование после fine-tuning
// ============================================
console.log('\n[6] Тест после обучения');
console.log('-'.repeat(40));

const testData = [
  { seq: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 0, 0, 0, 0, 0, 0]), expected: 0, name: 'Возрастающая' },
  { seq: new Float32Array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]), expected: 1, name: 'Убывающая' },
  { seq: new Float32Array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0]), expected: 2, name: 'Чередующаяся' },
];

let correct = 0;
for (const { seq, expected, name } of testData) {
  const input = { sequence: new Tensor(seq, [1, 16]) };
  const logits = model.classify(input);

  // Argmax
  let pred = 0;
  let maxVal = logits.data[0];
  for (let i = 1; i < 10; i++) {
    if (logits.data[i] > maxVal) {
      maxVal = logits.data[i];
      pred = i;
    }
  }

  const mark = pred === expected ? 'V' : 'X';
  console.log(`  ${name}: Предсказано ${pred}, Ожидается ${expected} [${mark}]`);
  if (pred === expected) correct++;
}

console.log(`\nТочность: ${(correct / testData.length * 100).toFixed(0)}%`);

// ============================================
// Архитектура
// ============================================
console.log('\n[7] Архитектура модели');
console.log('-'.repeat(40));

console.log(summary(model, [1, 16]));

console.log('\n' + '='.repeat(60));
console.log('Multimodal Few-Shot демо завершено!');
console.log('='.repeat(60));
