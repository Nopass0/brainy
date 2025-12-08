/**
 * @fileoverview Real-time Learning - Обучение во время работы
 * @description Демонстрация онлайн-обучения, continual learning и meta-learning
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
  OnlineLearner,
  ContinualLearner,
  MetaLearner,
  SelfTrainer,
} from '../src';

console.log('='.repeat(60));
console.log('Real-time Learning - Обучение во время работы');
console.log('='.repeat(60));

// ============================================
// 1. Online Learning - постоянное обновление
// ============================================
console.log('\n[1] Online Learning');
console.log('-'.repeat(40));
console.log('Модель адаптируется к потоку данных в реальном времени');

// Простая модель
const onlineModel = new Sequential(
  new Linear(2, 16),
  new ReLU(),
  new Linear(16, 1)
);

const onlineLearner = new OnlineLearner(onlineModel, {
  lr: 0.01,
  bufferSize: 50,
  updateFrequency: 1,
  adaptiveLR: true,
});

// Симулируем поток данных (синусоида с дрейфом)
console.log('\nСимуляция потока данных с дрейфом концепта:');

let phase = 0;
const streamSize = 100;

for (let i = 0; i < streamSize; i++) {
  // Дрейф: меняем функцию со временем
  if (i < 30) {
    // Линейная функция
    var fn = (x: number, y: number) => x + y;
  } else if (i < 60) {
    // Квадратичная
    var fn = (x: number, y: number) => x * x + y;
  } else {
    // Синусоида
    var fn = (x: number, y: number) => Math.sin(x * 3) + y;
  }

  const x = Math.random() * 2 - 1;
  const y = Math.random() * 2 - 1;
  const target = fn(x, y);

  const input = new Tensor(new Float32Array([x, y]), [1, 2], { requiresGrad: true });
  const targetTensor = new Tensor(new Float32Array([target]), [1, 1]);

  const { prediction, loss } = onlineLearner.predictAndLearn(input, targetTensor);

  if ((i + 1) % 20 === 0) {
    const stats = onlineLearner.getStats();
    console.log(`  Шаг ${i + 1}: Loss = ${loss.toFixed(4)}, Буфер = ${stats.bufferSize}`);
  }
}

const finalStats = onlineLearner.getStats();
console.log(`\nИтого: ${finalStats.stepCount} шагов, Avg Loss = ${finalStats.avgLoss.toFixed(4)}`);

// ============================================
// 2. Continual Learning - без забывания
// ============================================
console.log('\n[2] Continual Learning');
console.log('-'.repeat(40));
console.log('Обучение на последовательности задач без катастрофического забывания');

const continualModel = new Sequential(
  new Linear(4, 32),
  new ReLU(),
  new Linear(32, 2)
);

const continualLearner = new ContinualLearner(continualModel, 0.01, 30, 0.3);

// Задача 1: XOR
const task1Data: { x: Tensor; y: Tensor }[] = [];
for (let i = 0; i < 50; i++) {
  const a = Math.random() > 0.5 ? 1 : 0;
  const b = Math.random() > 0.5 ? 1 : 0;
  task1Data.push({
    x: new Tensor(new Float32Array([a, b, 0, 0]), [1, 4], { requiresGrad: true }),
    y: new Tensor(new Float32Array([a !== b ? 1 : 0, 0]), [1, 2]),
  });
}

// Задача 2: AND
const task2Data: { x: Tensor; y: Tensor }[] = [];
for (let i = 0; i < 50; i++) {
  const a = Math.random() > 0.5 ? 1 : 0;
  const b = Math.random() > 0.5 ? 1 : 0;
  task2Data.push({
    x: new Tensor(new Float32Array([0, 0, a, b]), [1, 4], { requiresGrad: true }),
    y: new Tensor(new Float32Array([0, a && b ? 1 : 0]), [1, 2]),
  });
}

// Обучаем на задаче 1
console.log('\nОбучение на задаче 1 (XOR):');
const losses1 = continualLearner.trainOnTask('xor', task1Data, 10);
console.log(`  Финальный loss: ${losses1[losses1.length - 1].toFixed(4)}`);

// Обучаем на задаче 2 с replay
console.log('\nОбучение на задаче 2 (AND) с replay:');
const losses2 = continualLearner.trainOnTask('and', task2Data, 10);
console.log(`  Финальный loss: ${losses2[losses2.length - 1].toFixed(4)}`);

// Оценка на обеих задачах
console.log('\nОценка на обеих задачах:');
const tasks = new Map([
  ['xor', task1Data.slice(0, 10)],
  ['and', task2Data.slice(0, 10)],
]);
const results = continualLearner.evaluate(tasks);
for (const [task, loss] of results.entries()) {
  console.log(`  ${task.toUpperCase()}: Loss = ${loss.toFixed(4)}`);
}

// ============================================
// 3. Meta-Learning - быстрая адаптация
// ============================================
console.log('\n[3] Meta-Learning (MAML-style)');
console.log('-'.repeat(40));
console.log('Быстрая адаптация к новым задачам за несколько шагов');

const metaModel = new Sequential(
  new Linear(2, 32),
  new ReLU(),
  new Linear(32, 1)
);

const metaLearner = new MetaLearner(metaModel, 0.001, 0.1, 5);

// Разные целевые функции
const functions = [
  (x: number) => Math.sin(x * Math.PI),
  (x: number) => x * x,
  (x: number) => Math.abs(x),
  (x: number) => x * 2,
];

console.log('\nАдаптация к разным функциям:');

for (let f = 0; f < functions.length; f++) {
  const fn = functions[f];
  const fnNames = ['sin(x*pi)', 'x^2', '|x|', '2x'];

  // Support set (3 примера)
  const supportXData = new Float32Array(3 * 2);
  const supportYData = new Float32Array(3);
  for (let i = 0; i < 3; i++) {
    const x = (i - 1) * 0.5;
    supportXData[i * 2] = x;
    supportXData[i * 2 + 1] = 0;
    supportYData[i] = fn(x);
  }
  const supportX = new Tensor(supportXData, [3, 2], { requiresGrad: true });
  const supportY = new Tensor(supportYData, [3, 1]);

  // Query
  const queryXData = new Float32Array([0.3, 0]);
  const queryX = new Tensor(queryXData, [1, 2]);
  const expected = fn(0.3);

  // Предсказание после адаптации
  const pred = metaLearner.predictAfterAdapt(supportX, supportY, queryX);

  console.log(`  ${fnNames[f]}: Предсказано = ${pred.data[0].toFixed(3)}, Ожидается = ${expected.toFixed(3)}`);
}

// ============================================
// 4. Self-Training - псевдо-разметка
// ============================================
console.log('\n[4] Self-Training');
console.log('-'.repeat(40));
console.log('Модель обучается на своих уверенных предсказаниях');

const selfModel = new Sequential(
  new Linear(3, 16),
  new ReLU(),
  new Linear(16, 3)
);

// Сначала обучаем на малом количестве данных
const labeledData: { x: Tensor; y: Tensor }[] = [];
for (let i = 0; i < 10; i++) {
  const classIdx = i % 3;
  const x = new Float32Array(3).fill(0);
  x[classIdx] = 1 + Math.random() * 0.1;
  labeledData.push({
    x: new Tensor(x, [1, 3], { requiresGrad: true }),
    y: new Tensor(new Float32Array([classIdx === 0 ? 1 : 0, classIdx === 1 ? 1 : 0, classIdx === 2 ? 1 : 0]), [1, 3]),
  });
}

// Начальное обучение
const initOpt = new Adam(selfModel.parameters(), 0.1);
console.log('\nНачальное обучение на размеченных данных:');
for (let epoch = 0; epoch < 20; epoch++) {
  let loss = 0;
  for (const { x, y } of labeledData) {
    const pred = selfModel.forward(x);
    const l = pred.sub(y).pow(2).mean();
    initOpt.zeroGrad();
    l.backward();
    initOpt.step();
    loss += l.item();
  }
  if ((epoch + 1) % 10 === 0) {
    console.log(`  Эпоха ${epoch + 1}: Loss = ${(loss / labeledData.length).toFixed(4)}`);
  }
}

// Self-training
const selfTrainer = new SelfTrainer(selfModel, 0.01, 0.7);

console.log('\nSelf-training на неразмеченных данных:');
let accepted = 0;

for (let i = 0; i < 30; i++) {
  const classIdx = Math.floor(Math.random() * 3);
  const x = new Float32Array(3).fill(0);
  x[classIdx] = 0.8 + Math.random() * 0.4;

  const result = selfTrainer.addUnlabeled(new Tensor(x, [1, 3], { requiresGrad: true }));
  if (result.accepted) accepted++;
}

console.log(`  Принято ${accepted} из 30 примеров (confidence >= 0.7)`);

if (accepted > 0) {
  const avgLoss = selfTrainer.trainOnPseudoLabels(5);
  console.log(`  Loss после self-training: ${avgLoss.toFixed(4)}`);
}

// ============================================
// 5. Симуляция реального применения
// ============================================
console.log('\n[5] Симуляция: Рекомендательная система');
console.log('-'.repeat(40));
console.log('Модель адаптируется к предпочтениям пользователя в реальном времени');

const recommenderModel = new Sequential(
  new Linear(5, 16),
  new ReLU(),
  new Linear(16, 1),
  new Tanh()
);

const recommender = new OnlineLearner(recommenderModel, {
  lr: 0.05,
  bufferSize: 20,
  updateFrequency: 1,
});

// Симулируем предпочтения пользователя
// Пользователь любит: высокий рейтинг, низкую цену, определённый жанр
function userPreference(features: number[]): number {
  const [rating, price, genre1, genre2, genre3] = features;
  // Предпочитает высокий рейтинг, низкую цену, жанр 1
  return rating * 0.4 - price * 0.3 + genre1 * 0.5 - genre2 * 0.2;
}

console.log('\nОбучение на feedback пользователя:');

for (let i = 0; i < 30; i++) {
  // Случайный товар
  const features = [
    Math.random(),          // рейтинг
    Math.random(),          // цена
    Math.random() > 0.5 ? 1 : 0,  // жанр 1
    Math.random() > 0.5 ? 1 : 0,  // жанр 2
    Math.random() > 0.5 ? 1 : 0,  // жанр 3
  ];

  const input = new Tensor(new Float32Array(features), [1, 5], { requiresGrad: true });

  // Предсказание модели
  const prediction = recommender.predict(input);
  const predicted = prediction.data[0];

  // Реальный feedback пользователя
  const actual = userPreference(features);
  const target = new Tensor(new Float32Array([actual]), [1, 1]);

  // Обучаем на feedback
  recommender.addExample(input, target);

  if ((i + 1) % 10 === 0) {
    const error = Math.abs(predicted - actual);
    console.log(`  Товар ${i + 1}: Предсказано = ${predicted.toFixed(2)}, Реально = ${actual.toFixed(2)}, Ошибка = ${error.toFixed(2)}`);
  }
}

// Тест на новых товарах
console.log('\nТест на новых товарах:');
const testItems = [
  [0.9, 0.1, 1, 0, 0],  // Идеальный товар (высокий рейтинг, низкая цена, жанр 1)
  [0.2, 0.9, 0, 1, 0],  // Плохой товар (низкий рейтинг, высокая цена, нелюбимый жанр)
  [0.5, 0.5, 0, 0, 1],  // Средний товар
];

for (const features of testItems) {
  const input = new Tensor(new Float32Array(features), [1, 5]);
  const pred = recommender.predict(input);
  const actual = userPreference(features);
  console.log(`  Товар [${features.join(', ')}]: Предсказано = ${pred.data[0].toFixed(2)}, Реально = ${actual.toFixed(2)}`);
}

console.log('\n' + '='.repeat(60));
console.log('Real-time Learning демо завершено!');
console.log('='.repeat(60));
