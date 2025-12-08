/**
 * Пример 05: Классификация MNIST (упрощённая версия)
 * 
 * Этот пример демонстрирует:
 * - Создание CNN модели для классификации изображений
 * - Использование Conv2d, MaxPool2d, Flatten
 * - CrossEntropyLoss для многоклассовой классификации
 * - Работа с синтетическими данными в формате MNIST
 * 
 * Примечание: Для полноценного MNIST нужно загрузить датасет.
 * Здесь мы используем синтетические данные для демонстрации архитектуры.
 * 
 * Запуск: bun run examples/05-mnist-classification.ts
 */

import {
  tensor,
  zeros,
  randn,
  Sequential,
  Linear,
  Conv2d,
  MaxPool2d,
  Flatten,
  ReLU,
  Dropout,
  Module,
  Tensor,
  CrossEntropyLoss,
  Adam,
  softmax,
} from '../src';

console.log('🧠 Brainy - Пример 05: Классификация MNIST (демо)\n');
console.log('='.repeat(60));

// ============================================
// 1. Создание синтетических данных
// ============================================
console.log('\n📊 1. Создание синтетических данных\n');

const BATCH_SIZE = 8;
const NUM_CLASSES = 10;
const IMAGE_SIZE = 14; // Уменьшенный размер для быстроты

console.log(`Размер батча: ${BATCH_SIZE}`);
console.log(`Количество классов: ${NUM_CLASSES}`);
console.log(`Размер изображения: ${IMAGE_SIZE}x${IMAGE_SIZE}`);

// Генерируем случайные "изображения" [batch, 1, 14, 14]
const X = randn([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE]);

// Генерируем случайные метки [batch]
const labelsData: number[] = [];
for (let i = 0; i < BATCH_SIZE; i++) {
  labelsData.push(Math.floor(Math.random() * NUM_CLASSES));
}
const Y = tensor(labelsData);

console.log(`X shape: [${X.shape}]`);
console.log(`Y shape: [${Y.shape}]`);
console.log(`Labels: [${labelsData.join(', ')}]`);

// ============================================
// 2. Создание CNN модели
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🏗️ 2. Создание CNN модели\n');

/**
 * LeNet-подобная архитектура:
 * Conv2d(1, 8, 3) -> ReLU -> MaxPool2d(2) ->
 * Conv2d(8, 16, 3) -> ReLU -> MaxPool2d(2) ->
 * Flatten -> Linear -> ReLU -> Linear(10)
 */
class MNISTModel extends Module {
  conv1: Conv2d;
  conv2: Conv2d;
  pool: MaxPool2d;
  flatten: Flatten;
  fc1: Linear;
  fc2: Linear;
  relu: ReLU;
  dropout: Dropout;

  constructor() {
    super();
    
    // Свёрточные слои
    this.conv1 = new Conv2d(1, 8, 3, 1, 1);   // [B, 1, 14, 14] -> [B, 8, 14, 14]
    this.conv2 = new Conv2d(8, 16, 3, 1, 1);  // [B, 8, 7, 7] -> [B, 16, 7, 7]
    this.pool = new MaxPool2d(2);              // Уменьшает в 2 раза
    this.flatten = new Flatten();
    
    // Полносвязные слои
    // После conv1+pool: 14/2 = 7
    // После conv2+pool: 7/2 = 3
    // Flatten: 16 * 3 * 3 = 144
    this.fc1 = new Linear(144, 64);
    this.fc2 = new Linear(64, NUM_CLASSES);
    
    this.relu = new ReLU();
    this.dropout = new Dropout(0.5);
    
    // Регистрируем подмодули
    this.registerModule('conv1', this.conv1);
    this.registerModule('conv2', this.conv2);
    this.registerModule('pool', this.pool);
    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
  }

  forward(x: Tensor): Tensor {
    // Conv block 1
    let out = this.conv1.forward(x);        // [B, 8, 14, 14]
    out = this.relu.forward(out);
    out = this.pool.forward(out);           // [B, 8, 7, 7]
    
    // Conv block 2
    out = this.conv2.forward(out);          // [B, 16, 7, 7]
    out = this.relu.forward(out);
    out = this.pool.forward(out);           // [B, 16, 3, 3]
    
    // Classifier
    out = this.flatten.forward(out);        // [B, 144]
    out = this.fc1.forward(out);            // [B, 64]
    out = this.relu.forward(out);
    out = this.dropout.forward(out);
    out = this.fc2.forward(out);            // [B, 10]
    
    return out;
  }
}

const model = new MNISTModel();

console.log('Архитектура CNN:');
console.log('┌─────────────────────────────────────────────────┐');
console.log('│  Input: [B, 1, 14, 14]                          │');
console.log('│    ↓                                             │');
console.log('│  Conv2d(1→8, 3x3) + ReLU                         │');
console.log('│    ↓                                             │');
console.log('│  MaxPool2d(2) → [B, 8, 7, 7]                     │');
console.log('│    ↓                                             │');
console.log('│  Conv2d(8→16, 3x3) + ReLU                        │');
console.log('│    ↓                                             │');
console.log('│  MaxPool2d(2) → [B, 16, 3, 3]                    │');
console.log('│    ↓                                             │');
console.log('│  Flatten → [B, 144]                              │');
console.log('│    ↓                                             │');
console.log('│  Linear(144→64) + ReLU + Dropout                 │');
console.log('│    ↓                                             │');
console.log('│  Linear(64→10) → [B, 10] (logits)                │');
console.log('└─────────────────────────────────────────────────┘');
console.log(`\nВсего параметров: ${model.numParameters()}`);

// ============================================
// 3. Настройка обучения
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n⚙️ 3. Настройка обучения\n');

const criterion = new CrossEntropyLoss();
const optimizer = new Adam(model.parameters(), 0.001);

const EPOCHS = 50;
console.log(`Функция потерь: CrossEntropyLoss`);
console.log(`Оптимизатор: Adam(lr=0.001)`);
console.log(`Эпох: ${EPOCHS}`);

// ============================================
// 4. Обучение
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🎓 4. Обучение\n');

model.train();

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // Forward
  const logits = model.forward(X);
  
  // Loss
  const loss = criterion.forward(logits, Y);
  const lossVal = loss.item();
  
  // Backward
  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();
  
  // Calculate accuracy
  const probs = softmax(logits, -1);
  let correct = 0;
  for (let i = 0; i < BATCH_SIZE; i++) {
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let j = 0; j < NUM_CLASSES; j++) {
      const val = probs.get(i, j);
      if (val > maxVal) {
        maxVal = val;
        maxIdx = j;
      }
    }
    if (maxIdx === labelsData[i]) correct++;
  }
  const accuracy = (correct / BATCH_SIZE) * 100;
  
  if (epoch % 10 === 0 || epoch === EPOCHS - 1) {
    console.log(
      `Эпоха ${epoch.toString().padStart(2)}: ` +
      `loss = ${lossVal.toFixed(4)}, ` +
      `accuracy = ${accuracy.toFixed(0)}%`
    );
  }
}

// ============================================
// 5. Inference
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n🔮 5. Inference (eval mode)\n');

model.eval(); // Отключаем dropout

const testLogits = model.forward(X);
const testProbs = softmax(testLogits, -1);

console.log('Предсказания для каждого изображения:');
console.log('┌─────────┬────────────┬──────────────┬──────────────┐');
console.log('│  Image  │   Label    │  Predicted   │  Confidence  │');
console.log('├─────────┼────────────┼──────────────┼──────────────┤');

for (let i = 0; i < BATCH_SIZE; i++) {
  let maxIdx = 0;
  let maxProb = 0;
  for (let j = 0; j < NUM_CLASSES; j++) {
    const prob = testProbs.get(i, j);
    if (prob > maxProb) {
      maxProb = prob;
      maxIdx = j;
    }
  }
  
  const correct = maxIdx === labelsData[i] ? '✅' : '❌';
  console.log(
    `│    ${i}    │     ${labelsData[i]}      │      ${maxIdx}       │    ${(maxProb * 100).toFixed(1).padStart(5)}%    │ ${correct}`
  );
}
console.log('└─────────┴────────────┴──────────────┴──────────────┘');

// ============================================
// Итоги
// ============================================
console.log('\n' + '='.repeat(60));
console.log('\n✅ CNN модель для классификации изображений создана!');
console.log('\n📝 Примечание: Это демонстрация архитектуры на синтетических данных.');
console.log('   Для реального MNIST нужно загрузить датасет и обучить на 60k изображениях.');
console.log('\nСледующий пример: bun run examples/06-custom-layer.ts');
