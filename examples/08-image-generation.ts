/**
 * @fileoverview Пример генерации изображений с VAE
 * @description Демонстрация обучения VAE и генерации изображений
 */

import {
  tensor,
  randn,
  zeros,
  ones,
  rand,
  Adam,
  DType,
  TensorDataset,
  DataLoader,
} from '../src';

import { VAE, ConvVAE, createMNISTVAE } from '../src/models/vae';
import { saveCheckpoint, CompressionFormat } from '../src/utils/checkpoint';

// ============================================
// 1. СОЗДАНИЕ СИНТЕТИЧЕСКИХ ДАННЫХ
// ============================================

console.log('=== Пример: Генерация изображений с VAE ===\n');

// Создаём синтетический датасет простых паттернов
// (в реальности используйте MNIST или другой датасет)

function createSyntheticData(numSamples: number, imageSize: number = 28): Float32Array[] {
  const images: Float32Array[] = [];

  for (let i = 0; i < numSamples; i++) {
    const image = new Float32Array(imageSize * imageSize);

    // Создаём случайный паттерн
    const patternType = i % 4;

    switch (patternType) {
      case 0:
        // Горизонтальные полосы
        for (let y = 0; y < imageSize; y++) {
          const intensity = y % 4 < 2 ? 0.8 : 0.2;
          for (let x = 0; x < imageSize; x++) {
            image[y * imageSize + x] = intensity + (Math.random() * 0.1 - 0.05);
          }
        }
        break;

      case 1:
        // Вертикальные полосы
        for (let y = 0; y < imageSize; y++) {
          for (let x = 0; x < imageSize; x++) {
            const intensity = x % 4 < 2 ? 0.8 : 0.2;
            image[y * imageSize + x] = intensity + (Math.random() * 0.1 - 0.05);
          }
        }
        break;

      case 2:
        // Круг
        const cx = imageSize / 2;
        const cy = imageSize / 2;
        const radius = imageSize / 4;
        for (let y = 0; y < imageSize; y++) {
          for (let x = 0; x < imageSize; x++) {
            const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
            image[y * imageSize + x] = dist < radius ? 0.9 : 0.1;
          }
        }
        break;

      case 3:
        // Градиент
        for (let y = 0; y < imageSize; y++) {
          for (let x = 0; x < imageSize; x++) {
            image[y * imageSize + x] = (x + y) / (2 * imageSize);
          }
        }
        break;
    }

    images.push(image);
  }

  return images;
}

console.log('Создание синтетического датасета...');

const imageSize = 28;
const numTrainSamples = 200;
const numTestSamples = 50;

const trainImages = createSyntheticData(numTrainSamples, imageSize);
const testImages = createSyntheticData(numTestSamples, imageSize);

console.log(`Train samples: ${trainImages.length}`);
console.log(`Test samples: ${testImages.length}`);
console.log(`Image size: ${imageSize}x${imageSize}\n`);

// ============================================
// 2. СОЗДАНИЕ VAE МОДЕЛИ
// ============================================

console.log('Создание VAE модели...');

const latentDim = 16;
const vae = createMNISTVAE(latentDim);

console.log('Конфигурация VAE:');
console.log(`  - Image size: ${imageSize}x${imageSize}`);
console.log(`  - Image channels: 1 (grayscale)`);
console.log(`  - Latent dimension: ${latentDim}`);
console.log(`  - Total parameters: ${vae.numParameters()}\n`);

// ============================================
// 3. ОБУЧЕНИЕ VAE
// ============================================

console.log('Обучение VAE...');

const optimizer = new Adam(vae.parameters(), 0.001);
const numEpochs = 30;
const batchSize = 16;

vae.train();

const trainTensors = trainImages.map(img =>
  new (tensor as any)(img, [1, imageSize, imageSize])
);

for (let epoch = 0; epoch < numEpochs; epoch++) {
  let totalLoss = 0;
  let totalReconLoss = 0;
  let totalKLLoss = 0;
  let numBatches = 0;

  // Перемешиваем
  const shuffled = [...trainTensors].sort(() => Math.random() - 0.5);

  for (let i = 0; i < shuffled.length; i += batchSize) {
    const batch = shuffled.slice(i, i + batchSize);
    if (batch.length < batchSize) continue;

    // Создаём батч тензор
    const batchData = new Float32Array(batchSize * imageSize * imageSize);
    for (let j = 0; j < batch.length; j++) {
      batchData.set(batch[j].data, j * imageSize * imageSize);
    }
    const batchTensor = new (tensor as any)(batchData, [batchSize, 1, imageSize, imageSize]);

    // Forward pass
    const output = vae.forward(batchTensor);
    const { loss, reconLoss, klLoss } = vae.computeLoss(batchTensor, output);

    // Backward pass
    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    totalLoss += loss.item();
    totalReconLoss += reconLoss;
    totalKLLoss += klLoss;
    numBatches++;
  }

  if ((epoch + 1) % 5 === 0 || epoch === 0) {
    console.log(
      `Epoch ${epoch + 1}/${numEpochs} - ` +
      `Loss: ${(totalLoss / numBatches).toFixed(4)}, ` +
      `Recon: ${(totalReconLoss / numBatches).toFixed(4)}, ` +
      `KL: ${(totalKLLoss / numBatches).toFixed(4)}`
    );
  }
}

console.log('\nОбучение завершено!\n');

// ============================================
// 4. РЕКОНСТРУКЦИЯ ИЗОБРАЖЕНИЙ
// ============================================

console.log('Реконструкция тестовых изображений...');

vae.eval();

// Берём несколько тестовых изображений
const testSamples = testImages.slice(0, 4);
const testBatchData = new Float32Array(4 * imageSize * imageSize);
for (let i = 0; i < testSamples.length; i++) {
  testBatchData.set(testSamples[i], i * imageSize * imageSize);
}
const testBatch = new (tensor as any)(testBatchData, [4, 1, imageSize, imageSize]);

// Реконструируем
const reconstructed = vae.reconstruct(testBatch);

// Вычисляем MSE между оригиналом и реконструкцией
let mse = 0;
for (let i = 0; i < testBatch.size; i++) {
  mse += (testBatch.data[i] - reconstructed.data[i]) ** 2;
}
mse /= testBatch.size;

console.log(`Reconstruction MSE: ${mse.toFixed(6)}\n`);

// Выводим пример (ASCII art)
console.log('Пример реконструкции (ASCII art):');
console.log('Original:');
printImageASCII(testSamples[0], imageSize);
console.log('\nReconstructed:');
const reconData = new Float32Array(imageSize * imageSize);
for (let i = 0; i < imageSize * imageSize; i++) {
  reconData[i] = reconstructed.data[i];
}
printImageASCII(reconData, imageSize);
console.log('');

// ============================================
// 5. ГЕНЕРАЦИЯ НОВЫХ ИЗОБРАЖЕНИЙ
// ============================================

console.log('Генерация новых изображений из случайного шума...');

const numGenerated = 4;
const generated = vae.generate(numGenerated);

console.log(`Сгенерировано ${numGenerated} изображений\n`);

// Выводим сгенерированные изображения
for (let i = 0; i < numGenerated; i++) {
  console.log(`Generated image ${i + 1}:`);
  const imgData = new Float32Array(imageSize * imageSize);
  for (let j = 0; j < imageSize * imageSize; j++) {
    imgData[j] = Math.max(0, Math.min(1, generated.data[i * imageSize * imageSize + j]));
  }
  printImageASCII(imgData, imageSize);
  console.log('');
}

// ============================================
// 6. ИНТЕРПОЛЯЦИЯ В ЛАТЕНТНОМ ПРОСТРАНСТВЕ
// ============================================

console.log('Интерполяция между двумя изображениями...');

// Берём два разных изображения
const img1Data = new Float32Array(imageSize * imageSize);
const img2Data = new Float32Array(imageSize * imageSize);
img1Data.set(trainImages[0]);
img2Data.set(trainImages[1]);

const img1 = new (tensor as any)(img1Data, [1, 1, imageSize, imageSize]);
const img2 = new (tensor as any)(img2Data, [1, 1, imageSize, imageSize]);

const interpolations = vae.interpolate(img1, img2, 5);

console.log(`\nИнтерполяция (${interpolations.length} шагов):`);
for (let i = 0; i < interpolations.length; i++) {
  console.log(`Step ${i}:`);
  const interpData = new Float32Array(imageSize * imageSize);
  for (let j = 0; j < imageSize * imageSize; j++) {
    interpData[j] = Math.max(0, Math.min(1, interpolations[i].data[j]));
  }
  printImageASCII(interpData, imageSize, 14); // Меньший размер для компактности
  console.log('');
}

// ============================================
// 7. ИССЛЕДОВАНИЕ ЛАТЕНТНОГО ПРОСТРАНСТВА
// ============================================

console.log('Исследование латентного пространства...');

// Кодируем все тестовые изображения
const allTestData = new Float32Array(testImages.length * imageSize * imageSize);
for (let i = 0; i < testImages.length; i++) {
  allTestData.set(testImages[i], i * imageSize * imageSize);
}
const allTestTensor = new (tensor as any)(allTestData, [testImages.length, 1, imageSize, imageSize]);

const latentVectors = vae.encode(allTestTensor);

// Статистика латентного пространства
let latentMean = 0;
let latentStd = 0;
for (let i = 0; i < latentVectors.size; i++) {
  latentMean += latentVectors.data[i];
}
latentMean /= latentVectors.size;

for (let i = 0; i < latentVectors.size; i++) {
  latentStd += (latentVectors.data[i] - latentMean) ** 2;
}
latentStd = Math.sqrt(latentStd / latentVectors.size);

console.log(`Latent space statistics:`);
console.log(`  - Mean: ${latentMean.toFixed(4)}`);
console.log(`  - Std: ${latentStd.toFixed(4)}`);
console.log(`  - Shape: [${latentVectors.shape.join(', ')}]\n`);

// ============================================
// 8. СОХРАНЕНИЕ МОДЕЛИ
// ============================================

console.log('Сохранение VAE модели...');

await saveCheckpoint('./vae_checkpoint.brainy', vae, optimizer, {
  compression: CompressionFormat.FLOAT16,
  metadata: {
    epoch: numEpochs,
    modelConfig: vae.getConfig(),
  },
});

console.log('Модель сохранена!\n');

console.log('=== Пример завершён ===');

// ============================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================

/**
 * Выводит изображение в ASCII формате
 */
function printImageASCII(data: Float32Array, size: number, displaySize: number = 14): void {
  const chars = ' .:-=+*#%@';
  const step = Math.ceil(size / displaySize);

  for (let y = 0; y < size; y += step) {
    let row = '';
    for (let x = 0; x < size; x += step) {
      const val = data[y * size + x];
      const charIdx = Math.floor(Math.max(0, Math.min(1, val)) * (chars.length - 1));
      row += chars[charIdx] + chars[charIdx];
    }
    console.log(row);
  }
}
