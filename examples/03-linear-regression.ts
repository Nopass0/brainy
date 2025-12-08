/**
 * Пример 03: Линейная регрессия
 * 
 * Этот пример демонстрирует:
 * - Создание простой модели Linear
 * - Использование MSELoss
 * - Обучение с SGD оптимизатором
 * - Визуализация процесса обучения
 * 
 * Запуск: bun run examples/03-linear-regression.ts
 */

import {
  tensor,
  randn,
  Linear,
  MSELoss,
  SGD,
  Module,
  Tensor,
  Sequential,
} from '../src';

console.log('🧠 Brainy - Пример 03: Линейная регрессия\n');
console.log('='.repeat(60));

// ============================================
// 1. Генерация данных
// ============================================
console.log('\n📊 1. Генерация данных\n');

// Истинные параметры: y = 2*x + 3 + noise
const TRUE_WEIGHT = 2.0;
const TRUE_BIAS = 3.0;

const NUM_SAMPLES = 100;

// Генерируем X
const xData: number[] = [];
const yData: number[] = [];

for (let i = 0; i < NUM_SAMPLES; i++) {
  const x = Math.random() * 10 - 5; // [-5, 5]
  const noise = (Math.random() - 0.5) * 0.5;
  const y = TRUE_WEIGHT * x + TRUE_BIAS + noise;
  xData.push(x);
  yData.push(y);
}

const X = tensor(xData.map(x => [x])); // [N, 1]
const Y = tensor(yData.map(y => [y])); // [N, 1]

console.log(`Сгенерировано ${NUM_SAMPLES} точек`);
console.log(`Истинные параметры: weight = ${TRUE_WEIGHT}, bias = ${TRUE_BIAS}`);
console.log(`X shape: [${X.shape}], Y shape: [${Y.shape}]`);

// Показываем несколько примеров
console.log('\nПримеры данных:');
for (let i = 0; i < 5; i++) {
  console.log(`  x = ${xData[i].toFixed(2)}, y = ${yData[i].toFixed(2)}`);
}

// ============================================
// 2. Создание модели
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🏗️ 2. Создание модели\n');

// Простая линейная модель: y = Wx + b
const model = new Linear(1, 1);

console.log('Модель:', model.toString());
console.log(`Количество параметров: ${model.numParameters()}`);

// Начальные значения весов
console.log('\nНачальные параметры:');
console.log(`  weight: ${model.weight.data.item().toFixed(4)}`);
console.log(`  bias: ${model.bias!.data.item().toFixed(4)}`);

// ============================================
// 3. Настройка обучения
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n⚙️ 3. Настройка обучения\n');

const criterion = new MSELoss();
const optimizer = new SGD(model.parameters(), 0.01, { momentum: 0.9 });

const EPOCHS = 100;
console.log(`Функция потерь: ${criterion.toString()}`);
console.log(`Оптимизатор: ${optimizer.toString()}`);
console.log(`Эпох: ${EPOCHS}`);

// ============================================
// 4. Обучение
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🎓 4. Обучение\n');

const losses: number[] = [];

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // Forward pass
  const predictions = model.forward(X);
  
  // Вычисляем loss
  const loss = criterion.forward(predictions, Y);
  const lossVal = loss.item();
  losses.push(lossVal);
  
  // Backward pass
  optimizer.zeroGrad();
  loss.backward();
  
  // Update weights
  optimizer.step();
  
  // Логирование
  if (epoch % 10 === 0 || epoch === EPOCHS - 1) {
    const w = model.weight.data.item();
    const b = model.bias!.data.item();
    console.log(
      `Эпоха ${epoch.toString().padStart(3)}: ` +
      `loss = ${lossVal.toFixed(4)}, ` +
      `w = ${w.toFixed(4)}, ` +
      `b = ${b.toFixed(4)}`
    );
  }
}

// ============================================
// 5. Результаты
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n📈 5. Результаты\n');

const learnedWeight = model.weight.data.item();
const learnedBias = model.bias!.data.item();

console.log('Сравнение параметров:');
console.log('┌──────────┬───────────────┬───────────────┐');
console.log('│ Параметр │    Истинное   │   Выученное   │');
console.log('├──────────┼───────────────┼───────────────┤');
console.log(`│  weight  │     ${TRUE_WEIGHT.toFixed(4)}     │     ${learnedWeight.toFixed(4)}     │`);
console.log(`│   bias   │     ${TRUE_BIAS.toFixed(4)}     │     ${learnedBias.toFixed(4)}     │`);
console.log('└──────────┴───────────────┴───────────────┘');

const weightError = Math.abs(learnedWeight - TRUE_WEIGHT);
const biasError = Math.abs(learnedBias - TRUE_BIAS);

console.log(`\nОшибка weight: ${(weightError * 100).toFixed(2)}%`);
console.log(`Ошибка bias: ${(biasError * 100).toFixed(2)}%`);

// Визуализация loss (ASCII график)
console.log('\n📉 График loss:');
const maxLoss = Math.max(...losses);
const width = 50;
for (let i = 0; i < EPOCHS; i += 10) {
  const barLen = Math.round((losses[i] / maxLoss) * width);
  const bar = '█'.repeat(barLen) + '░'.repeat(width - barLen);
  console.log(`Эпоха ${i.toString().padStart(3)}: ${bar} ${losses[i].toFixed(4)}`);
}

// ============================================
// 6. Предсказания
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔮 6. Тестовые предсказания\n');

const testX = tensor([[-5], [0], [5], [10]]);
const testY = model.forward(testX);

console.log('Тестовые предсказания:');
for (let i = 0; i < testX.shape[0]; i++) {
  const xVal = testX.get(i, 0);
  const predicted = testY.get(i, 0);
  const actual = TRUE_WEIGHT * xVal + TRUE_BIAS;
  console.log(
    `  x = ${xVal.toFixed(1)}: ` +
    `предсказано = ${predicted.toFixed(2)}, ` +
    `истинное = ${actual.toFixed(2)}`
  );
}

// ============================================
// Итоги
// ============================================
console.log('\n' + '='.repeat(60));

if (weightError < 0.1 && biasError < 0.1) {
  console.log('\n✅ Линейная регрессия успешно обучена!');
} else {
  console.log('\n⚠️ Модель обучилась, но есть погрешность. Попробуйте увеличить количество эпох.');
}

console.log('\nСледующий пример: bun run examples/04-xor-neural-network.ts');
