/**
 * Пример 04: Нейронная сеть для XOR
 * 
 * Этот пример демонстрирует:
 * - Создание многослойной нейронной сети
 * - Использование Sequential для построения модели
 * - Активации ReLU и Sigmoid
 * - Обучение на классическом нелинейном XOR датасете
 * 
 * XOR: классическая задача, которую нельзя решить линейной моделью!
 * 
 * Запуск: bun run examples/04-xor-neural-network.ts
 */

import {
  tensor,
  Sequential,
  Linear,
  ReLU,
  Sigmoid,
  MSELoss,
  Adam,
  Module,
  Tensor,
} from '../src';

console.log('🧠 Brainy - Пример 04: Нейронная сеть для XOR\n');
console.log('='.repeat(60));

// ============================================
// 1. Данные XOR
// ============================================
console.log('\n📊 1. Данные XOR\n');

// XOR таблица истинности:
// 0 XOR 0 = 0
// 0 XOR 1 = 1
// 1 XOR 0 = 1
// 1 XOR 1 = 0

const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]);

const Y = tensor([
  [0],
  [1],
  [1],
  [0]
]);

console.log('XOR таблица:');
console.log('┌─────────┬─────────┬─────────┐');
console.log('│   A     │    B    │  A XOR B │');
console.log('├─────────┼─────────┼─────────┤');
for (let i = 0; i < 4; i++) {
  console.log(`│    ${X.get(i, 0)}    │    ${X.get(i, 1)}    │    ${Y.get(i, 0)}     │`);
}
console.log('└─────────┴─────────┴─────────┘');

console.log('\n⚠️ Это классическая нелинейная задача - линейная модель не может её решить!');

// ============================================
// 2. Создание модели
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🏗️ 2. Создание модели\n');

// Многослойная нейронная сеть:
// Input(2) -> Linear(8) -> ReLU -> Linear(4) -> ReLU -> Linear(1) -> Sigmoid
const model = new Sequential(
  new Linear(2, 8),     // 2 входа -> 8 скрытых нейронов
  new ReLU(),
  new Linear(8, 4),     // 8 -> 4 скрытых нейрона  
  new ReLU(),
  new Linear(4, 1),     // 4 -> 1 выход
  new Sigmoid()         // Sigmoid для вероятности [0, 1]
);

console.log('Архитектура модели:');
console.log(model.toString());
console.log(`\nВсего параметров: ${model.numParameters()}`);

// ============================================
// 3. Настройка обучения
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n⚙️ 3. Настройка обучения\n');

const criterion = new MSELoss();
const optimizer = new Adam(model.parameters(), 0.1);

const EPOCHS = 1000;
console.log(`Функция потерь: ${criterion.toString()}`);
console.log(`Оптимизатор: ${optimizer.toString()}`);
console.log(`Эпох: ${EPOCHS}`);

// ============================================
// 4. Обучение
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🎓 4. Обучение\n');

let bestLoss = Infinity;
const losses: number[] = [];

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // Forward pass
  const predictions = model.forward(X);
  
  // Compute loss
  const loss = criterion.forward(predictions, Y);
  const lossVal = loss.item();
  losses.push(lossVal);
  
  if (lossVal < bestLoss) {
    bestLoss = lossVal;
  }
  
  // Backward pass
  optimizer.zeroGrad();
  loss.backward();
  
  // Update
  optimizer.step();
  
  // Logging
  if (epoch % 100 === 0 || epoch === EPOCHS - 1) {
    const preds = predictions.toArray() as number[][];
    const accuracy = preds.reduce((acc, pred, i) => {
      const predicted = pred[0] > 0.5 ? 1 : 0;
      const actual = (Y.toArray() as number[][])[i][0];
      return acc + (predicted === actual ? 1 : 0);
    }, 0) / 4 * 100;
    
    console.log(
      `Эпоха ${epoch.toString().padStart(4)}: ` +
      `loss = ${lossVal.toFixed(6)}, ` +
      `accuracy = ${accuracy.toFixed(0)}%`
    );
  }
}

// ============================================
// 5. Результаты
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n📈 5. Финальные предсказания\n');

const finalPreds = model.forward(X);

console.log('XOR предсказания:');
console.log('┌─────────┬─────────┬─────────┬────────────────┬───────────┐');
console.log('│    A    │    B    │  Истина │  Предсказание  │  Округл.  │');
console.log('├─────────┼─────────┼─────────┼────────────────┼───────────┤');

let correct = 0;
for (let i = 0; i < 4; i++) {
  const a = X.get(i, 0);
  const b = X.get(i, 1);
  const truth = Y.get(i, 0);
  const pred = finalPreds.get(i, 0);
  const rounded = pred > 0.5 ? 1 : 0;
  const isCorrect = rounded === truth;
  if (isCorrect) correct++;
  
  console.log(
    `│    ${a}    │    ${b}    │    ${truth}    │` +
    `     ${pred.toFixed(4)}     │` +
    `     ${rounded}     │` +
    (isCorrect ? ' ✅' : ' ❌')
  );
}
console.log('└─────────┴─────────┴─────────┴────────────────┴───────────┘');

const accuracy = (correct / 4) * 100;
console.log(`\n📊 Точность: ${correct}/4 = ${accuracy}%`);

// ============================================
// 6. График обучения (ASCII)
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n📉 6. График loss (каждые 100 эпох)\n');

const displayEpochs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999];
const maxLoss = Math.max(...displayEpochs.map(e => losses[e]));
const width = 40;

for (const epoch of displayEpochs) {
  const loss = losses[epoch];
  const barLen = Math.max(1, Math.round((loss / maxLoss) * width));
  const bar = '█'.repeat(barLen);
  console.log(`${epoch.toString().padStart(4)}: ${bar} ${loss.toFixed(6)}`);
}

// ============================================
// Итоги
// ============================================
console.log('\n' + '='.repeat(60));

if (accuracy === 100) {
  console.log('\n🎉 XOR задача решена! Нейросеть научилась нелинейной функции!');
} else {
  console.log(`\n⚠️ Точность ${accuracy}%. Попробуйте увеличить количество эпох или изменить архитектуру.`);
}

console.log('\nСледующий пример: bun run examples/05-mnist-classification.ts');
