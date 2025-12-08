/**
 * @fileoverview Пример генерации текста с GPT-подобной моделью
 * @description Демонстрация создания, обучения и генерации текста
 */

import {
  tensor,
  randn,
  zeros,
  Adam,
  DType,
} from '../src';

import { GPT, createSmallGPT } from '../src/models/gpt';
import { CharTokenizer } from '../src/text/tokenizer';
import { saveCheckpoint, loadCheckpoint, CompressionFormat } from '../src/utils/checkpoint';
import { dynamicQuantize } from '../src/utils/quantization';

// ============================================
// 1. ПОДГОТОВКА ДАННЫХ
// ============================================

console.log('=== Пример: Генерация текста с GPT ===\n');

// Простой корпус для демонстрации
const corpus = `
Привет мир! Это пример обучения языковой модели.
Модель учится предсказывать следующий символ.
Чем больше данных, тем лучше результаты.
Нейронные сети могут генерировать текст.
Трансформеры революционизировали NLP.
Attention is all you need.
GPT модели используют decoder-only архитектуру.
Каждый токен предсказывается на основе предыдущих.
`.trim();

console.log('Корпус для обучения:');
console.log(corpus.slice(0, 100) + '...\n');

// ============================================
// 2. СОЗДАНИЕ ТОКЕНИЗАТОРА
// ============================================

console.log('Создание токенизатора...');

const tokenizer = new CharTokenizer({
  vocabSize: 256,
  maxLength: 64,
});

console.log(`Размер словаря: ${tokenizer.vocabSize}`);
console.log(`PAD token ID: ${tokenizer.padTokenId}`);
console.log(`BOS token ID: ${tokenizer.bosTokenId}`);
console.log(`EOS token ID: ${tokenizer.eosTokenId}\n`);

// Токенизация примера
const testText = 'Привет мир!';
const encoded = tokenizer.encode(testText, true);
console.log(`Текст: "${testText}"`);
console.log(`Токены: [${encoded.inputIds.slice(0, 20).join(', ')}...]`);
console.log(`Decoded: "${tokenizer.decode(encoded.inputIds, true)}"\n`);

// ============================================
// 3. СОЗДАНИЕ МОДЕЛИ
// ============================================

console.log('Создание GPT модели...');

const model = createSmallGPT(tokenizer.vocabSize);
const config = model.getConfig();

console.log('Конфигурация модели:');
console.log(`  - Hidden size: ${config.hiddenSize}`);
console.log(`  - Num heads: ${config.numHeads}`);
console.log(`  - Num layers: ${config.numLayers}`);
console.log(`  - Max seq length: ${config.maxSeqLength}`);
console.log(`  - Vocab size: ${config.vocabSize}`);
console.log(`  - Total parameters: ${model.numParameters()}\n`);

// ============================================
// 4. ПОДГОТОВКА ОБУЧАЮЩИХ ДАННЫХ
// ============================================

console.log('Подготовка обучающих данных...');

// Разбиваем корпус на последовательности
const seqLength = 32;
const sequences: number[][] = [];

// Токенизируем напрямую без padding
const tokens = tokenizer.tokenize(corpus);
const tokenIds = tokens.map(t => tokenizer.getTokenId(t));

console.log(`Всего токенов в корпусе: ${tokenIds.length}`);

// Создаём последовательности с 50% overlap
for (let i = 0; i < tokenIds.length - seqLength; i += seqLength / 2) {
  const seq = tokenIds.slice(i, i + seqLength + 1);
  if (seq.length === seqLength + 1) {
    sequences.push(seq);
  }
}

console.log(`Создано ${sequences.length} обучающих последовательностей\n`);

// ============================================
// 5. ОБУЧЕНИЕ МОДЕЛИ
// ============================================

console.log('Обучение модели...');

const optimizer = new Adam(model.parameters(), 0.001);
const numEpochs = 50;
const batchSize = 4;

model.train();

for (let epoch = 0; epoch < numEpochs; epoch++) {
  let totalLoss = 0;
  let numBatches = 0;

  // Перемешиваем последовательности
  const shuffled = [...sequences].sort(() => Math.random() - 0.5);

  for (let i = 0; i < shuffled.length; i += batchSize) {
    const batch = shuffled.slice(i, i + batchSize);
    if (batch.length < batchSize) continue;

    // Создаём входные данные и метки
    const inputData: number[] = [];
    const labelData: number[] = [];

    for (const seq of batch) {
      inputData.push(...seq.slice(0, -1));
      labelData.push(...seq.slice(1));
    }

    const inputIds = tensor(
      batch.map(seq => seq.slice(0, -1)),
      { dtype: DType.Int32 }
    );
    const labels = tensor(
      batch.map(seq => seq.slice(1)),
      { dtype: DType.Int32 }
    );

    // Forward pass
    const loss = model.computeLoss(inputIds, labels);

    // Backward pass
    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    totalLoss += loss.item();
    numBatches++;
  }

  if ((epoch + 1) % 10 === 0 || epoch === 0) {
    console.log(`Epoch ${epoch + 1}/${numEpochs}, Loss: ${(totalLoss / numBatches).toFixed(4)}`);
  }
}

console.log('\nОбучение завершено!\n');

// ============================================
// 6. ГЕНЕРАЦИЯ ТЕКСТА
// ============================================

console.log('Генерация текста...\n');

model.eval();

const prompts = ['Привет', 'Модель', 'Нейрон'];

for (const prompt of prompts) {
  console.log(`Prompt: "${prompt}"`);

  // Кодируем prompt
  const promptEncoded = tokenizer.encode(prompt, true);
  const inputIds = tensor([promptEncoded.inputIds], { dtype: DType.Int32 });

  // Генерируем
  const outputIds = model.generate(inputIds, {
    maxNewTokens: 30,
    temperature: 0.8,
    topK: 10,
    topP: 0.9,
    doSample: true,
  });

  // Декодируем
  const generated = tokenizer.decode(
    Array.from(outputIds.data).map(x => Math.floor(x)),
    true
  );

  console.log(`Generated: "${generated}"\n`);
}

// ============================================
// 7. GREEDY GENERATION
// ============================================

console.log('Greedy generation (temperature=0):');

const greedyPrompt = 'GPT';
const greedyEncoded = tokenizer.encode(greedyPrompt, true);
const greedyInput = tensor([greedyEncoded.inputIds], { dtype: DType.Int32 });

const greedyOutput = model.generate(greedyInput, {
  maxNewTokens: 40,
  doSample: false, // Greedy
});

const greedyGenerated = tokenizer.decode(
  Array.from(greedyOutput.data).map(x => Math.floor(x)),
  true
);

console.log(`Prompt: "${greedyPrompt}"`);
console.log(`Generated: "${greedyGenerated}"\n`);

// ============================================
// 8. СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛИ
// ============================================

console.log('Сохранение модели...');

// Сохраняем с GZIP сжатием
await saveCheckpoint('./model_checkpoint.brainy', model, optimizer, {
  compression: CompressionFormat.GZIP,
  metadata: {
    epoch: numEpochs,
  },
});

console.log('Модель сохранена!\n');

// Создаём новую модель и загружаем веса
console.log('Загрузка модели...');

const loadedModel = createSmallGPT(tokenizer.vocabSize);
const metadata = await loadCheckpoint('./model_checkpoint.brainy', loadedModel);

console.log('Метаданные:');
console.log(`  - Version: ${metadata.version}`);
console.log(`  - Created: ${metadata.createdAt}`);
console.log(`  - Compression: ${metadata.compression}\n`);

// ============================================
// 9. КВАНТИЗАЦИЯ
// ============================================

console.log('Квантизация модели...');

const quantizedModel = dynamicQuantize(model, 8);
const stats = quantizedModel.getQuantizationStats();

console.log('Статистика квантизации:');
console.log(`  - Original size: ${(stats.totalOriginalSize / 1024).toFixed(2)} KB`);
console.log(`  - Compressed size: ${(stats.totalCompressedSize / 1024).toFixed(2)} KB`);
console.log(`  - Compression ratio: ${stats.compressionRatio.toFixed(2)}x`);
console.log(`  - Quantized layers: ${stats.numQuantizedLayers}\n`);

// ============================================
// 10. ГЕНЕРАЦИЯ С РАЗНЫМИ ТЕМПЕРАТУРАМИ
// ============================================

console.log('Влияние температуры на генерацию:\n');

const tempPrompt = 'Трансформ';
const temperatures = [0.5, 1.0, 1.5];

for (const temp of temperatures) {
  const tempEncoded = tokenizer.encode(tempPrompt, true);
  const tempInput = tensor([tempEncoded.inputIds], { dtype: DType.Int32 });

  const tempOutput = model.generate(tempInput, {
    maxNewTokens: 30,
    temperature: temp,
    topK: 20,
    doSample: true,
  });

  const tempGenerated = tokenizer.decode(
    Array.from(tempOutput.data).map(x => Math.floor(x)),
    true
  );

  console.log(`Temperature ${temp}: "${tempGenerated}"`);
}

console.log('\n=== Пример завершён ===');
