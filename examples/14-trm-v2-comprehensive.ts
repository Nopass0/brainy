/**
 * @fileoverview TRM v2 Comprehensive Testing & Training
 * @description Extensive testing of TRM v2 on various tasks to achieve 98%+ accuracy
 *
 * Test Suite:
 * 1. XOR Problem (classic) - должна быть 100% точность
 * 2. Complex XOR with noise - целевая точность 98%+
 * 3. Multi-class classification - целевая точность 98%+
 * 4. Visual pattern recognition - целевая точность 95%+
 * 5. Arithmetic operations - целевая точность 98%+
 * 6. Sequence patterns - целевая точность 95%+
 * 7. Few-shot learning - целевая точность 90%+
 * 8. Adaptive computation analysis
 */

import {
  tensor,
  randn,
  zeros,
  ones,
  TRMv2,
  TRMv2Classifier,
  createTinyTRMv2,
  createReasoningTRMv2,
  Adam,
  MSELoss,
  CrossEntropyLoss,
  DataLoader,
  TensorDataset,
} from '../src';
import { Tensor } from '../src/core/tensor';

// ============================================
// UTILITY FUNCTIONS
// ============================================

function shuffleArrays(x: number[][], y: number[][]): void {
  for (let i = x.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [x[i], x[j]] = [x[j], x[i]];
    [y[i], y[j]] = [y[j], y[i]];
  }
}

function accuracy(predictions: Tensor, targets: Tensor): number {
  const batchSize = predictions.shape[0];
  let correct = 0;

  for (let i = 0; i < batchSize; i++) {
    // Find predicted class
    let predClass = 0;
    let maxVal = -Infinity;
    for (let j = 0; j < predictions.shape[1]; j++) {
      const val = predictions.data[i * predictions.shape[1] + j];
      if (val > maxVal) {
        maxVal = val;
        predClass = j;
      }
    }

    // Find target class
    let targetClass = 0;
    maxVal = -Infinity;
    for (let j = 0; j < targets.shape[1]; j++) {
      const val = targets.data[i * targets.shape[1] + j];
      if (val > maxVal) {
        maxVal = val;
        targetClass = j;
      }
    }

    if (predClass === targetClass) {
      correct++;
    }
  }

  return correct / batchSize;
}

function binaryAccuracy(predictions: Tensor, targets: Tensor): number {
  const batchSize = predictions.shape[0];
  let correct = 0;

  for (let i = 0; i < batchSize; i++) {
    const pred = predictions.data[i] > 0.5 ? 1 : 0;
    const target = targets.data[i] > 0.5 ? 1 : 0;
    if (pred === target) {
      correct++;
    }
  }

  return correct / batchSize;
}

function regressionAccuracy(predictions: Tensor, targets: Tensor, threshold: number = 0.1): number {
  const batchSize = predictions.shape[0];
  let correct = 0;

  for (let i = 0; i < batchSize; i++) {
    const pred = predictions.data[i];
    const target = targets.data[i];
    if (Math.abs(pred - target) < threshold) {
      correct++;
    }
  }

  return correct / batchSize;
}

// ============================================
// TEST 1: XOR Problem (Must be 100%)
// ============================================

async function testXOR(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 1: XOR Problem');
  console.log('='.repeat(60));

  const model = createTinyTRMv2(2, 1, 64, 8);
  const optimizer = new Adam(model.parameters(), 0.01);
  const lossFunc = new MSELoss();

  // XOR dataset
  const xData = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  const yData = [[0], [1], [1], [0]];

  // Training with data augmentation
  const epochs = 2000;
  let bestAccuracy = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    // Train on each sample multiple times with slight noise
    for (let rep = 0; rep < 4; rep++) {
      for (let i = 0; i < xData.length; i++) {
        // Add small noise for regularization
        const noise = Math.random() * 0.1 - 0.05;
        const x = tensor([[xData[i][0] + noise, xData[i][1] + noise]]);
        const y = tensor([[yData[i][0]]]);

        optimizer.zeroGrad();
        const output = model.forward(x);
        const loss = lossFunc.forward(output, y);
        loss.backward();
        optimizer.step();

        totalLoss += loss.item();
      }
    }

    // Evaluate every 200 epochs
    if ((epoch + 1) % 200 === 0) {
      const testX = tensor(xData);
      const testY = tensor(yData);
      const predictions = model.forward(testX);
      const acc = binaryAccuracy(predictions, testY);
      bestAccuracy = Math.max(bestAccuracy, acc);

      console.log(`  Epoch ${epoch + 1}: Loss = ${(totalLoss / 16).toFixed(4)}, Accuracy = ${(acc * 100).toFixed(1)}%`);
    }
  }

  // Final evaluation
  const finalX = tensor(xData);
  const finalY = tensor(yData);
  const finalPred = model.forward(finalX);
  const finalAcc = binaryAccuracy(finalPred, finalY);

  console.log(`\n  Final Accuracy: ${(finalAcc * 100).toFixed(1)}%`);
  console.log(`  Target: 100%`);
  console.log(`  Status: ${finalAcc >= 0.99 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: finalAcc, passed: finalAcc >= 0.99 };
}

// ============================================
// TEST 2: Noisy XOR (Target 98%+)
// ============================================

async function testNoisyXOR(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 2: Noisy XOR Problem');
  console.log('='.repeat(60));

  const model = createTinyTRMv2(2, 1, 128, 10);
  const optimizer = new Adam(model.parameters(), 0.005);
  const lossFunc = new MSELoss();

  // Generate noisy XOR dataset
  const generateData = (n: number, noiseLevel: number = 0.15) => {
    const x: number[][] = [];
    const y: number[][] = [];

    for (let i = 0; i < n; i++) {
      const a = Math.random() > 0.5 ? 1 : 0;
      const b = Math.random() > 0.5 ? 1 : 0;
      const xorResult = a !== b ? 1 : 0;

      x.push([
        a + (Math.random() - 0.5) * noiseLevel * 2,
        b + (Math.random() - 0.5) * noiseLevel * 2,
      ]);
      y.push([xorResult]);
    }

    return { x, y };
  };

  const trainData = generateData(500, 0.2);
  const testData = generateData(100, 0.2);

  const epochs = 100;
  const batchSize = 32;

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle training data
    shuffleArrays(trainData.x, trainData.y);

    let totalLoss = 0;
    let batches = 0;

    for (let i = 0; i < trainData.x.length; i += batchSize) {
      const batchX = trainData.x.slice(i, i + batchSize);
      const batchY = trainData.y.slice(i, i + batchSize);

      if (batchX.length === 0) continue;

      const x = tensor(batchX);
      const y = tensor(batchY);

      optimizer.zeroGrad();
      const output = model.forward(x);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();

      totalLoss += loss.item();
      batches++;
    }

    if ((epoch + 1) % 20 === 0) {
      const testX = tensor(testData.x);
      const testY = tensor(testData.y);
      const predictions = model.forward(testX);
      const acc = binaryAccuracy(predictions, testY);

      console.log(`  Epoch ${epoch + 1}: Loss = ${(totalLoss / batches).toFixed(4)}, Test Accuracy = ${(acc * 100).toFixed(1)}%`);
    }
  }

  // Final evaluation
  const finalX = tensor(testData.x);
  const finalY = tensor(testData.y);
  const finalPred = model.forward(finalX);
  const finalAcc = binaryAccuracy(finalPred, finalY);

  console.log(`\n  Final Accuracy: ${(finalAcc * 100).toFixed(1)}%`);
  console.log(`  Target: 98%`);
  console.log(`  Status: ${finalAcc >= 0.98 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: finalAcc, passed: finalAcc >= 0.98 };
}

// ============================================
// TEST 3: Multi-class Classification (Target 98%+)
// ============================================

async function testMultiClassification(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 3: Multi-class Classification (Gaussian Blobs)');
  console.log('='.repeat(60));

  const numClasses = 5;
  const model = new TRMv2({
    inputDim: 4,
    hiddenDim: 128,
    outputDim: numClasses,
    numRecursions: 10,
    numHeads: 4,
    numExperts: 4,
    memorySlots: 32,
    adaptiveComputation: true,
  });

  const optimizer = new Adam(model.parameters(), 0.003);
  const lossFunc = new CrossEntropyLoss();

  // Generate Gaussian blob dataset
  const generateBlobs = (n: number, numClasses: number) => {
    const x: number[][] = [];
    const y: number[][] = [];

    // Class centers
    const centers = [
      [2, 2, 0, 0],
      [-2, -2, 0, 0],
      [2, -2, 1, 0],
      [-2, 2, -1, 0],
      [0, 0, 0, 2],
    ];

    for (let i = 0; i < n; i++) {
      const classIdx = Math.floor(Math.random() * numClasses);
      const center = centers[classIdx];

      // Sample around center with noise
      const sample = center.map(c => c + (Math.random() - 0.5) * 0.8);
      x.push(sample);

      // One-hot encode
      const oneHot = new Array(numClasses).fill(0);
      oneHot[classIdx] = 1;
      y.push(oneHot);
    }

    return { x, y };
  };

  const trainData = generateBlobs(1000, numClasses);
  const testData = generateBlobs(200, numClasses);

  const epochs = 80;
  const batchSize = 64;

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleArrays(trainData.x, trainData.y);

    let totalLoss = 0;
    let batches = 0;

    for (let i = 0; i < trainData.x.length; i += batchSize) {
      const batchX = trainData.x.slice(i, i + batchSize);
      const batchY = trainData.y.slice(i, i + batchSize);

      if (batchX.length === 0) continue;

      const x = tensor(batchX);
      const y = tensor(batchY);

      optimizer.zeroGrad();
      const output = model.forward(x);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();

      totalLoss += loss.item();
      batches++;
    }

    if ((epoch + 1) % 20 === 0) {
      const testX = tensor(testData.x);
      const testY = tensor(testData.y);
      const predictions = model.forward(testX);
      const acc = accuracy(predictions, testY);

      console.log(`  Epoch ${epoch + 1}: Loss = ${(totalLoss / batches).toFixed(4)}, Test Accuracy = ${(acc * 100).toFixed(1)}%`);
    }
  }

  // Final evaluation
  const finalX = tensor(testData.x);
  const finalY = tensor(testData.y);
  const finalPred = model.forward(finalX);
  const finalAcc = accuracy(finalPred, finalY);

  console.log(`\n  Final Accuracy: ${(finalAcc * 100).toFixed(1)}%`);
  console.log(`  Target: 98%`);
  console.log(`  Status: ${finalAcc >= 0.98 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: finalAcc, passed: finalAcc >= 0.98 };
}

// ============================================
// TEST 4: Pattern Recognition (Target 95%+)
// ============================================

async function testPatternRecognition(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 4: Visual Pattern Recognition (3x3 Grid)');
  console.log('='.repeat(60));

  const model = createReasoningTRMv2(9, 4); // 9 inputs (3x3), 4 pattern classes
  const optimizer = new Adam(model.parameters(), 0.002);
  const lossFunc = new CrossEntropyLoss();

  // Pattern definitions (3x3 grids)
  const patterns = {
    // Horizontal line
    horizontal: [
      [0, 0, 0, 1, 1, 1, 0, 0, 0],
      [1, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    // Vertical line
    vertical: [
      [0, 1, 0, 0, 1, 0, 0, 1, 0],
      [1, 0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0, 1],
    ],
    // Diagonal
    diagonal: [
      [1, 0, 0, 0, 1, 0, 0, 0, 1],
      [0, 0, 1, 0, 1, 0, 1, 0, 0],
    ],
    // Cross
    cross: [
      [0, 1, 0, 1, 1, 1, 0, 1, 0],
      [1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
  };

  // Generate training data
  const generatePatternData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];
    const patternKeys = Object.keys(patterns);

    for (let i = 0; i < n; i++) {
      const classIdx = Math.floor(Math.random() * patternKeys.length);
      const patternKey = patternKeys[classIdx] as keyof typeof patterns;
      const patternList = patterns[patternKey];
      const basePattern = patternList[Math.floor(Math.random() * patternList.length)];

      // Add noise
      const noisyPattern = basePattern.map(v => {
        const noise = (Math.random() - 0.5) * 0.3;
        return Math.max(0, Math.min(1, v + noise));
      });

      x.push(noisyPattern);

      const oneHot = new Array(4).fill(0);
      oneHot[classIdx] = 1;
      y.push(oneHot);
    }

    return { x, y };
  };

  const trainData = generatePatternData(800);
  const testData = generatePatternData(200);

  const epochs = 100;
  const batchSize = 32;

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleArrays(trainData.x, trainData.y);

    let totalLoss = 0;
    let batches = 0;

    for (let i = 0; i < trainData.x.length; i += batchSize) {
      const batchX = trainData.x.slice(i, i + batchSize);
      const batchY = trainData.y.slice(i, i + batchSize);

      if (batchX.length === 0) continue;

      const x = tensor(batchX);
      const y = tensor(batchY);

      optimizer.zeroGrad();
      const output = model.forward(x);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();

      totalLoss += loss.item();
      batches++;
    }

    if ((epoch + 1) % 20 === 0) {
      const testX = tensor(testData.x);
      const testY = tensor(testData.y);
      const predictions = model.forward(testX);
      const acc = accuracy(predictions, testY);

      console.log(`  Epoch ${epoch + 1}: Loss = ${(totalLoss / batches).toFixed(4)}, Test Accuracy = ${(acc * 100).toFixed(1)}%`);
    }
  }

  // Final evaluation
  const finalX = tensor(testData.x);
  const finalY = tensor(testData.y);
  const finalPred = model.forward(finalX);
  const finalAcc = accuracy(finalPred, finalY);

  console.log(`\n  Final Accuracy: ${(finalAcc * 100).toFixed(1)}%`);
  console.log(`  Target: 95%`);
  console.log(`  Status: ${finalAcc >= 0.95 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: finalAcc, passed: finalAcc >= 0.95 };
}

// ============================================
// TEST 5: Arithmetic Operations (Target 98%+)
// ============================================

async function testArithmetic(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 5: Arithmetic Operations (a + b * c)');
  console.log('='.repeat(60));

  const model = createReasoningTRMv2(3, 1);
  const optimizer = new Adam(model.parameters(), 0.001);
  const lossFunc = new MSELoss();

  // Generate arithmetic dataset
  const generateArithmeticData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];

    for (let i = 0; i < n; i++) {
      const a = Math.random() * 2 - 1; // [-1, 1]
      const b = Math.random() * 2 - 1;
      const c = Math.random() * 2 - 1;

      const result = a + b * c;

      x.push([a, b, c]);
      y.push([result]);
    }

    return { x, y };
  };

  const trainData = generateArithmeticData(2000);
  const testData = generateArithmeticData(500);

  const epochs = 150;
  const batchSize = 64;

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleArrays(trainData.x, trainData.y);

    let totalLoss = 0;
    let batches = 0;

    for (let i = 0; i < trainData.x.length; i += batchSize) {
      const batchX = trainData.x.slice(i, i + batchSize);
      const batchY = trainData.y.slice(i, i + batchSize);

      if (batchX.length === 0) continue;

      const x = tensor(batchX);
      const y = tensor(batchY);

      optimizer.zeroGrad();
      const output = model.forward(x);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();

      totalLoss += loss.item();
      batches++;
    }

    if ((epoch + 1) % 30 === 0) {
      const testX = tensor(testData.x);
      const testY = tensor(testData.y);
      const predictions = model.forward(testX);
      const acc = regressionAccuracy(predictions, testY, 0.15);

      console.log(`  Epoch ${epoch + 1}: Loss = ${(totalLoss / batches).toFixed(4)}, Test Accuracy = ${(acc * 100).toFixed(1)}%`);
    }
  }

  // Final evaluation
  const finalX = tensor(testData.x);
  const finalY = tensor(testData.y);
  const finalPred = model.forward(finalX);
  const finalAcc = regressionAccuracy(finalPred, finalY, 0.15);

  console.log(`\n  Final Accuracy: ${(finalAcc * 100).toFixed(1)}%`);
  console.log(`  Target: 98%`);
  console.log(`  Status: ${finalAcc >= 0.98 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: finalAcc, passed: finalAcc >= 0.98 };
}

// ============================================
// TEST 6: Sequence Pattern (Target 95%+)
// ============================================

async function testSequencePattern(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 6: Sequence Pattern Detection (Fibonacci-like)');
  console.log('='.repeat(60));

  const model = new TRMv2({
    inputDim: 5,
    hiddenDim: 128,
    outputDim: 1,
    numRecursions: 12,
    numHeads: 4,
    numExperts: 4,
    memorySlots: 32,
    adaptiveComputation: true,
  });

  const optimizer = new Adam(model.parameters(), 0.002);
  const lossFunc = new MSELoss();

  // Generate sequence data
  // Pattern: next = a*x[0] + b*x[1] + c*x[2] where a,b,c are constants
  const generateSequenceData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];

    for (let i = 0; i < n; i++) {
      // Generate 5 numbers following a pattern
      const base = [
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
        0, 0, 0
      ];

      // Linear recurrence: x[n] = 0.5*x[n-1] + 0.3*x[n-2] + small_noise
      base[2] = 0.5 * base[1] + 0.3 * base[0] + (Math.random() - 0.5) * 0.1;
      base[3] = 0.5 * base[2] + 0.3 * base[1] + (Math.random() - 0.5) * 0.1;
      base[4] = 0.5 * base[3] + 0.3 * base[2] + (Math.random() - 0.5) * 0.1;

      // Predict next in sequence
      const next = 0.5 * base[4] + 0.3 * base[3];

      x.push(base);
      y.push([next]);
    }

    return { x, y };
  };

  const trainData = generateSequenceData(1500);
  const testData = generateSequenceData(300);

  const epochs = 120;
  const batchSize = 64;

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleArrays(trainData.x, trainData.y);

    let totalLoss = 0;
    let batches = 0;

    for (let i = 0; i < trainData.x.length; i += batchSize) {
      const batchX = trainData.x.slice(i, i + batchSize);
      const batchY = trainData.y.slice(i, i + batchSize);

      if (batchX.length === 0) continue;

      const x = tensor(batchX);
      const y = tensor(batchY);

      optimizer.zeroGrad();
      const output = model.forward(x);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();

      totalLoss += loss.item();
      batches++;
    }

    if ((epoch + 1) % 20 === 0) {
      const testX = tensor(testData.x);
      const testY = tensor(testData.y);
      const predictions = model.forward(testX);
      const acc = regressionAccuracy(predictions, testY, 0.1);

      console.log(`  Epoch ${epoch + 1}: Loss = ${(totalLoss / batches).toFixed(4)}, Test Accuracy = ${(acc * 100).toFixed(1)}%`);
    }
  }

  // Final evaluation
  const finalX = tensor(testData.x);
  const finalY = tensor(testData.y);
  const finalPred = model.forward(finalX);
  const finalAcc = regressionAccuracy(finalPred, finalY, 0.1);

  console.log(`\n  Final Accuracy: ${(finalAcc * 100).toFixed(1)}%`);
  console.log(`  Target: 95%`);
  console.log(`  Status: ${finalAcc >= 0.95 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: finalAcc, passed: finalAcc >= 0.95 };
}

// ============================================
// TEST 7: Few-Shot Learning (Target 90%+)
// ============================================

async function testFewShot(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 7: Few-Shot Learning (5-way 3-shot)');
  console.log('='.repeat(60));

  const numClasses = 5;
  const shotsPerClass = 3;
  const queryPerClass = 20;

  const classifier = new TRMv2Classifier(8, 128, numClasses, 10);
  const optimizer = new Adam(classifier.parameters(), 0.003);
  const lossFunc = new CrossEntropyLoss();

  // Generate class prototypes
  const classPrototypes = [
    [2, 0, 0, 0, 1, 0, 0, 0],
    [-2, 0, 0, 0, -1, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, 0, 0],
    [0, -2, 0, 0, 0, -1, 0, 0],
    [0, 0, 2, 2, 0, 0, 1, 1],
  ];

  const generateSamples = (classIdx: number, n: number): number[][] => {
    const samples: number[][] = [];
    const prototype = classPrototypes[classIdx];

    for (let i = 0; i < n; i++) {
      const sample = prototype.map(v => v + (Math.random() - 0.5) * 0.6);
      samples.push(sample);
    }

    return samples;
  };

  // Pre-train the feature extractor on some data
  console.log('  Pre-training feature extractor...');

  const preTrainX: number[][] = [];
  const preTrainY: number[][] = [];

  for (let c = 0; c < numClasses; c++) {
    const samples = generateSamples(c, 100);
    for (const sample of samples) {
      preTrainX.push(sample);
      const oneHot = new Array(numClasses).fill(0);
      oneHot[c] = 1;
      preTrainY.push(oneHot);
    }
  }

  // Pre-training epochs
  for (let epoch = 0; epoch < 50; epoch++) {
    shuffleArrays(preTrainX, preTrainY);

    let totalLoss = 0;
    const batchSize = 32;

    for (let i = 0; i < preTrainX.length; i += batchSize) {
      const batchX = preTrainX.slice(i, i + batchSize);
      const batchY = preTrainY.slice(i, i + batchSize);

      if (batchX.length === 0) continue;

      const x = tensor(batchX);
      const y = tensor(batchY);

      optimizer.zeroGrad();
      const output = classifier.forward(x);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();

      totalLoss += loss.item();
    }

    if ((epoch + 1) % 10 === 0) {
      console.log(`    Pre-train epoch ${epoch + 1}: Loss = ${(totalLoss / Math.ceil(preTrainX.length / 32)).toFixed(4)}`);
    }
  }

  // Now test few-shot
  console.log('\n  Testing few-shot classification...');

  let totalCorrect = 0;
  let totalQueries = 0;
  const numEpisodes = 20;

  for (let episode = 0; episode < numEpisodes; episode++) {
    // Generate support set
    const supportX: number[][] = [];
    const supportY: number[] = [];

    for (let c = 0; c < numClasses; c++) {
      const samples = generateSamples(c, shotsPerClass);
      for (const sample of samples) {
        supportX.push(sample);
        supportY.push(c);
      }
    }

    // Generate query set
    const queryX: number[][] = [];
    const queryYTrue: number[] = [];

    for (let c = 0; c < numClasses; c++) {
      const samples = generateSamples(c, queryPerClass);
      for (const sample of samples) {
        queryX.push(sample);
        queryYTrue.push(c);
      }
    }

    // Few-shot prediction
    const supportTensor = tensor(supportX);
    const queryTensor = tensor(queryX);

    const result = classifier.fewShotPredict(supportTensor, supportY, queryTensor);

    // Calculate accuracy
    for (let i = 0; i < queryYTrue.length; i++) {
      if (result.predictions[i] === queryYTrue[i]) {
        totalCorrect++;
      }
      totalQueries++;
    }
  }

  const finalAcc = totalCorrect / totalQueries;

  console.log(`\n  Final Few-Shot Accuracy: ${(finalAcc * 100).toFixed(1)}%`);
  console.log(`  Target: 90%`);
  console.log(`  Status: ${finalAcc >= 0.90 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: finalAcc, passed: finalAcc >= 0.90 };
}

// ============================================
// TEST 8: Adaptive Computation Analysis
// ============================================

async function testAdaptiveComputation(): Promise<{ accuracy: number; passed: boolean }> {
  console.log('\n' + '='.repeat(60));
  console.log('TEST 8: Adaptive Computation Analysis');
  console.log('='.repeat(60));

  const model = createReasoningTRMv2(4, 2);
  const optimizer = new Adam(model.parameters(), 0.005);
  const lossFunc = new CrossEntropyLoss();

  // Easy vs Hard classification task
  // Easy: clearly separable classes
  // Hard: overlapping classes

  const generateEasyData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];

    for (let i = 0; i < n; i++) {
      const classIdx = Math.random() > 0.5 ? 1 : 0;
      if (classIdx === 0) {
        x.push([2 + Math.random() * 0.3, 2 + Math.random() * 0.3, 0, 0]);
      } else {
        x.push([-2 + Math.random() * 0.3, -2 + Math.random() * 0.3, 0, 0]);
      }
      y.push(classIdx === 0 ? [1, 0] : [0, 1]);
    }

    return { x, y };
  };

  const generateHardData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];

    for (let i = 0; i < n; i++) {
      const classIdx = Math.random() > 0.5 ? 1 : 0;
      const noise = (Math.random() - 0.5) * 1.5;
      if (classIdx === 0) {
        x.push([0.5 + noise, 0.5 + noise, Math.random() - 0.5, Math.random() - 0.5]);
      } else {
        x.push([-0.5 + noise, -0.5 + noise, Math.random() - 0.5, Math.random() - 0.5]);
      }
      y.push(classIdx === 0 ? [1, 0] : [0, 1]);
    }

    return { x, y };
  };

  // Train on mixed data
  const trainEasy = generateEasyData(500);
  const trainHard = generateHardData(500);
  const trainData = {
    x: [...trainEasy.x, ...trainHard.x],
    y: [...trainEasy.y, ...trainHard.y],
  };

  console.log('  Training model...');

  const epochs = 80;
  const batchSize = 64;

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleArrays(trainData.x, trainData.y);

    let totalLoss = 0;
    let batches = 0;

    for (let i = 0; i < trainData.x.length; i += batchSize) {
      const batchX = trainData.x.slice(i, i + batchSize);
      const batchY = trainData.y.slice(i, i + batchSize);

      if (batchX.length === 0) continue;

      const x = tensor(batchX);
      const y = tensor(batchY);

      optimizer.zeroGrad();
      const output = model.forward(x);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();

      totalLoss += loss.item();
      batches++;
    }

    if ((epoch + 1) % 20 === 0) {
      console.log(`    Epoch ${epoch + 1}: Loss = ${(totalLoss / batches).toFixed(4)}`);
    }
  }

  // Test adaptive computation
  console.log('\n  Analyzing adaptive computation...');

  const testEasy = generateEasyData(100);
  const testHard = generateHardData(100);

  let easyStepsTotal = 0;
  let hardStepsTotal = 0;
  let easyCorrect = 0;
  let hardCorrect = 0;

  // Test on easy samples
  for (let i = 0; i < testEasy.x.length; i++) {
    const x = tensor([testEasy.x[i]]);
    const result = model.forwardAdaptive(x, 24, 0.005);

    easyStepsTotal += result.steps;

    const pred = result.output.data[0] > result.output.data[1] ? 0 : 1;
    const target = testEasy.y[i][0] > testEasy.y[i][1] ? 0 : 1;
    if (pred === target) {
      easyCorrect++;
    }
  }

  // Test on hard samples
  for (let i = 0; i < testHard.x.length; i++) {
    const x = tensor([testHard.x[i]]);
    const result = model.forwardAdaptive(x, 24, 0.005);

    hardStepsTotal += result.steps;

    const pred = result.output.data[0] > result.output.data[1] ? 0 : 1;
    const target = testHard.y[i][0] > testHard.y[i][1] ? 0 : 1;
    if (pred === target) {
      hardCorrect++;
    }
  }

  const avgEasySteps = easyStepsTotal / testEasy.x.length;
  const avgHardSteps = hardStepsTotal / testHard.x.length;
  const easyAcc = easyCorrect / testEasy.x.length;
  const hardAcc = hardCorrect / testHard.x.length;

  console.log(`\n  Easy samples:`);
  console.log(`    Average steps: ${avgEasySteps.toFixed(1)}`);
  console.log(`    Accuracy: ${(easyAcc * 100).toFixed(1)}%`);

  console.log(`\n  Hard samples:`);
  console.log(`    Average steps: ${avgHardSteps.toFixed(1)}`);
  console.log(`    Accuracy: ${(hardAcc * 100).toFixed(1)}%`);

  const overallAcc = (easyAcc + hardAcc) / 2;
  const adaptiveWorking = avgHardSteps > avgEasySteps;

  console.log(`\n  Adaptive behavior: ${adaptiveWorking ? 'WORKING (hard takes more steps)' : 'NOT DETECTED'}`);
  console.log(`  Overall Accuracy: ${(overallAcc * 100).toFixed(1)}%`);
  console.log(`  Target: Overall 90%+ accuracy`);
  console.log(`  Status: ${overallAcc >= 0.90 ? 'PASSED' : 'NEEDS IMPROVEMENT'}`);

  return { accuracy: overallAcc, passed: overallAcc >= 0.90 };
}

// ============================================
// MAIN TEST RUNNER
// ============================================

async function runAllTests(): Promise<void> {
  console.log('\n');
  console.log('*'.repeat(70));
  console.log('*' + ' '.repeat(20) + 'TRM v2 COMPREHENSIVE TEST SUITE' + ' '.repeat(17) + '*');
  console.log('*'.repeat(70));

  const results: { name: string; accuracy: number; passed: boolean }[] = [];

  // Run all tests
  const test1 = await testXOR();
  results.push({ name: 'XOR Problem', ...test1 });

  const test2 = await testNoisyXOR();
  results.push({ name: 'Noisy XOR', ...test2 });

  const test3 = await testMultiClassification();
  results.push({ name: 'Multi-class Classification', ...test3 });

  const test4 = await testPatternRecognition();
  results.push({ name: 'Pattern Recognition', ...test4 });

  const test5 = await testArithmetic();
  results.push({ name: 'Arithmetic Operations', ...test5 });

  const test6 = await testSequencePattern();
  results.push({ name: 'Sequence Pattern', ...test6 });

  const test7 = await testFewShot();
  results.push({ name: 'Few-Shot Learning', ...test7 });

  const test8 = await testAdaptiveComputation();
  results.push({ name: 'Adaptive Computation', ...test8 });

  // Summary
  console.log('\n');
  console.log('*'.repeat(70));
  console.log('*' + ' '.repeat(25) + 'TEST SUMMARY' + ' '.repeat(32) + '*');
  console.log('*'.repeat(70));
  console.log('');

  let totalPassed = 0;
  let avgAccuracy = 0;

  for (const result of results) {
    const status = result.passed ? 'PASSED' : 'FAILED';
    const icon = result.passed ? '[+]' : '[-]';
    console.log(`  ${icon} ${result.name.padEnd(25)} ${(result.accuracy * 100).toFixed(1).padStart(5)}%  ${status}`);

    if (result.passed) totalPassed++;
    avgAccuracy += result.accuracy;
  }

  avgAccuracy /= results.length;

  console.log('');
  console.log('-'.repeat(70));
  console.log(`  Tests Passed: ${totalPassed}/${results.length}`);
  console.log(`  Average Accuracy: ${(avgAccuracy * 100).toFixed(1)}%`);
  console.log('');

  if (totalPassed === results.length) {
    console.log('  STATUS: ALL TESTS PASSED! TRM v2 is performing excellently!');
  } else if (totalPassed >= results.length * 0.75) {
    console.log('  STATUS: Most tests passed. Some improvements needed.');
  } else {
    console.log('  STATUS: Multiple tests failed. Significant improvements needed.');
  }

  console.log('*'.repeat(70));
}

// Run tests
runAllTests().catch(console.error);
