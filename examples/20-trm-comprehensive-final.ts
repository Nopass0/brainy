/**
 * @fileoverview TRM Comprehensive Final Example
 * @description Complete demonstration of TRM capabilities with multiple tests
 *
 * Tests:
 * 1. Noisy XOR (following working pattern)
 * 2. Multi-class classification
 * 3. Arithmetic reasoning
 * 4. Sequence pattern prediction
 * 5. Simple text embedding task
 * 6. Few-shot learning
 */

import {
  Tensor,
  tensor,
  zeros,
  randn,
  TRM,
  TRMClassifier,
  createTinyTRM,
  createReasoningTRM,
  TRMv2,
  createTinyTRMv2,
  Adam,
  MSELoss,
  CrossEntropyLoss,
  Sequential,
  Linear,
  ReLU,
  GELU,
  Embedding,
} from '../src';

console.log('='.repeat(70));
console.log('TRM COMPREHENSIVE TEST SUITE');
console.log('='.repeat(70));

// ============================================
// Test 1: Noisy XOR (Proven Pattern)
// ============================================
async function testNoisyXOR(): Promise<number> {
  console.log('\n[1] NOISY XOR');
  console.log('-'.repeat(50));

  function generateNoisyXOR(n: number, noiseLevel: number = 0.2): { x: Tensor; y: Tensor } {
    const xData = new Float32Array(n * 2);
    const yData = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const a = Math.random() > 0.5 ? 1 : 0;
      const b = Math.random() > 0.5 ? 1 : 0;
      const xorResult = a !== b ? 1 : 0;

      xData[i * 2] = a + (Math.random() - 0.5) * noiseLevel * 2;
      xData[i * 2 + 1] = b + (Math.random() - 0.5) * noiseLevel * 2;
      yData[i] = xorResult;
    }

    return {
      x: new Tensor(xData, [n, 2], { requiresGrad: true }),
      y: new Tensor(yData, [n, 1]),
    };
  }

  const model = createTinyTRM(2, 1, 32, 4);
  const optimizer = new Adam(model.parameters(), 0.01);

  const trainData = generateNoisyXOR(300, 0.25);
  const testData = generateNoisyXOR(100, 0.25);

  for (let epoch = 0; epoch < 60; epoch++) {
    const output = model.forward(trainData.x);
    const loss = output.sub(trainData.y).pow(2).mean();

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 15 === 0) {
      const testPred = model.forward(testData.x);
      let correct = 0;
      for (let i = 0; i < 100; i++) {
        if ((testPred.data[i] > 0.5 ? 1 : 0) === testData.y.data[i]) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Loss = ${loss.item().toFixed(4)}, Test Acc = ${correct}%`);
    }
  }

  const finalPred = model.forward(testData.x);
  let correct = 0;
  for (let i = 0; i < 100; i++) {
    if ((finalPred.data[i] > 0.5 ? 1 : 0) === testData.y.data[i]) correct++;
  }
  console.log(`  Final: ${correct}%`);
  return correct / 100;
}

// ============================================
// Test 2: Multi-class Classification
// ============================================
async function testMultiClass(): Promise<number> {
  console.log('\n[2] MULTI-CLASS CLASSIFICATION');
  console.log('-'.repeat(50));

  const numClasses = 4;
  const centers = [[2, 2], [-2, 2], [2, -2], [-2, -2]];

  function generateData(n: number): { x: Tensor; y: Tensor } {
    const xData = new Float32Array(n * 2);
    const yData = new Float32Array(n * numClasses);

    for (let i = 0; i < n; i++) {
      const c = Math.floor(Math.random() * numClasses);
      xData[i * 2] = centers[c][0] + (Math.random() - 0.5) * 1.2;
      xData[i * 2 + 1] = centers[c][1] + (Math.random() - 0.5) * 1.2;

      for (let j = 0; j < numClasses; j++) {
        yData[i * numClasses + j] = j === c ? 1 : 0;
      }
    }

    return {
      x: new Tensor(xData, [n, 2], { requiresGrad: true }),
      y: new Tensor(yData, [n, numClasses]),
    };
  }

  const model = createReasoningTRM(2, numClasses);
  const optimizer = new Adam(model.parameters(), 0.005);
  const criterion = new CrossEntropyLoss();

  const trainData = generateData(400);
  const testData = generateData(100);

  for (let epoch = 0; epoch < 80; epoch++) {
    const output = model.forward(trainData.x);
    const loss = criterion.forward(output, trainData.y);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 20 === 0) {
      const testPred = model.forward(testData.x);
      let correct = 0;
      for (let i = 0; i < 100; i++) {
        let predClass = 0, trueClass = 0;
        let maxPred = -Infinity, maxTrue = -Infinity;
        for (let j = 0; j < numClasses; j++) {
          if (testPred.data[i * numClasses + j] > maxPred) {
            maxPred = testPred.data[i * numClasses + j];
            predClass = j;
          }
          if (testData.y.data[i * numClasses + j] > maxTrue) {
            maxTrue = testData.y.data[i * numClasses + j];
            trueClass = j;
          }
        }
        if (predClass === trueClass) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Loss = ${loss.item().toFixed(4)}, Test Acc = ${correct}%`);
    }
  }

  const finalPred = model.forward(testData.x);
  let correct = 0;
  for (let i = 0; i < 100; i++) {
    let predClass = 0, trueClass = 0;
    let maxPred = -Infinity, maxTrue = -Infinity;
    for (let j = 0; j < numClasses; j++) {
      if (finalPred.data[i * numClasses + j] > maxPred) {
        maxPred = finalPred.data[i * numClasses + j];
        predClass = j;
      }
      if (testData.y.data[i * numClasses + j] > maxTrue) {
        maxTrue = testData.y.data[i * numClasses + j];
        trueClass = j;
      }
    }
    if (predClass === trueClass) correct++;
  }
  console.log(`  Final: ${correct}%`);
  return correct / 100;
}

// ============================================
// Test 3: Arithmetic Reasoning
// ============================================
async function testArithmetic(): Promise<number> {
  console.log('\n[3] ARITHMETIC REASONING (a + b * c)');
  console.log('-'.repeat(50));

  function generateData(n: number): { x: Tensor; y: Tensor } {
    const xData = new Float32Array(n * 3);
    const yData = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const a = Math.random() * 2 - 1;
      const b = Math.random() * 2 - 1;
      const c = Math.random() * 2 - 1;

      xData[i * 3] = a;
      xData[i * 3 + 1] = b;
      xData[i * 3 + 2] = c;
      yData[i] = a + b * c;
    }

    return {
      x: new Tensor(xData, [n, 3], { requiresGrad: true }),
      y: new Tensor(yData, [n, 1]),
    };
  }

  const model = createReasoningTRM(3, 1);
  const optimizer = new Adam(model.parameters(), 0.005);

  const trainData = generateData(800);
  const testData = generateData(200);

  for (let epoch = 0; epoch < 120; epoch++) {
    const output = model.forward(trainData.x);
    const loss = output.sub(trainData.y).pow(2).mean();

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 30 === 0) {
      const testPred = model.forward(testData.x);
      let correct = 0;
      for (let i = 0; i < 200; i++) {
        if (Math.abs(testPred.data[i] - testData.y.data[i]) < 0.15) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Loss = ${loss.item().toFixed(4)}, Test Acc = ${(correct / 2).toFixed(1)}%`);
    }
  }

  const finalPred = model.forward(testData.x);
  let correct = 0;
  for (let i = 0; i < 200; i++) {
    if (Math.abs(finalPred.data[i] - testData.y.data[i]) < 0.15) correct++;
  }
  console.log(`  Final: ${(correct / 2).toFixed(1)}%`);
  return correct / 200;
}

// ============================================
// Test 4: Sequence Pattern
// ============================================
async function testSequence(): Promise<number> {
  console.log('\n[4] SEQUENCE PATTERN (Linear Recurrence)');
  console.log('-'.repeat(50));

  // Pattern: x[n] = 0.5*x[n-1] + 0.3*x[n-2]
  function generateData(n: number): { x: Tensor; y: Tensor } {
    const xData: number[] = [];
    const yData: number[] = [];

    for (let i = 0; i < n; i++) {
      const seq = [Math.random() * 2 - 1, Math.random() * 2 - 1];
      seq.push(0.5 * seq[1] + 0.3 * seq[0] + (Math.random() - 0.5) * 0.1);
      seq.push(0.5 * seq[2] + 0.3 * seq[1] + (Math.random() - 0.5) * 0.1);
      seq.push(0.5 * seq[3] + 0.3 * seq[2] + (Math.random() - 0.5) * 0.1);

      xData.push(...seq);
      yData.push(0.5 * seq[4] + 0.3 * seq[3]);
    }

    return {
      x: new Tensor(new Float32Array(xData), [n, 5], { requiresGrad: true }),
      y: new Tensor(new Float32Array(yData), [n, 1]),
    };
  }

  const model = createReasoningTRM(5, 1);
  const optimizer = new Adam(model.parameters(), 0.003);

  const trainData = generateData(600);
  const testData = generateData(150);

  for (let epoch = 0; epoch < 100; epoch++) {
    const output = model.forward(trainData.x);
    const loss = output.sub(trainData.y).pow(2).mean();

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 25 === 0) {
      const testPred = model.forward(testData.x);
      let correct = 0;
      for (let i = 0; i < 150; i++) {
        if (Math.abs(testPred.data[i] - testData.y.data[i]) < 0.1) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Loss = ${loss.item().toFixed(4)}, Test Acc = ${(correct / 1.5).toFixed(1)}%`);
    }
  }

  const finalPred = model.forward(testData.x);
  let correct = 0;
  for (let i = 0; i < 150; i++) {
    if (Math.abs(finalPred.data[i] - testData.y.data[i]) < 0.1) correct++;
  }
  console.log(`  Final: ${(correct / 1.5).toFixed(1)}%`);
  return correct / 150;
}

// ============================================
// Test 5: Simple Text Task (Sentiment-like)
// ============================================
async function testSimpleText(): Promise<number> {
  console.log('\n[5] SIMPLE TEXT CLASSIFICATION');
  console.log('-'.repeat(50));

  // Simple "sentiment" based on word patterns
  // positive words: [1, 0, 0], negative: [0, 1, 0], neutral: [0, 0, 1]
  // Sentence = average of word embeddings

  const vocabSize = 20;
  const embeddingDim = 8;
  const numClasses = 3;

  // Create embedding layer
  const embedding = new Embedding(vocabSize, embeddingDim);

  // TRM for classification
  const model = createTinyTRM(embeddingDim, numClasses, 32, 6);
  const optimizer = new Adam([...model.parameters(), ...embedding.parameters()], 0.01);
  const criterion = new CrossEntropyLoss();

  // Generate simple patterns
  // Class 0: words 0-5, Class 1: words 6-12, Class 2: words 13-19
  function generateData(n: number): { x: Tensor; y: Tensor } {
    const xData: number[][] = [];
    const yData: number[][] = [];

    for (let i = 0; i < n; i++) {
      const classIdx = Math.floor(Math.random() * 3);
      const startWord = classIdx * 6 + Math.floor(Math.random() * 2);

      // Create "sentence" of 3 words
      const words = [
        startWord + Math.floor(Math.random() * 5),
        startWord + Math.floor(Math.random() * 5),
        startWord + Math.floor(Math.random() * 5),
      ].map(w => Math.min(w, vocabSize - 1));

      // Get embeddings and average
      const wordIndices = new Tensor(new Float32Array(words), [3]);
      const wordEmbeds = embedding.forward(wordIndices);

      // Average
      const avgEmbed = new Float32Array(embeddingDim);
      for (let d = 0; d < embeddingDim; d++) {
        avgEmbed[d] = (wordEmbeds.data[d] + wordEmbeds.data[embeddingDim + d] +
                       wordEmbeds.data[2 * embeddingDim + d]) / 3;
      }

      xData.push(Array.from(avgEmbed));

      const oneHot = [0, 0, 0];
      oneHot[classIdx] = 1;
      yData.push(oneHot);
    }

    return {
      x: tensor(xData),
      y: tensor(yData),
    };
  }

  const trainData = generateData(300);
  const testData = generateData(100);

  for (let epoch = 0; epoch < 60; epoch++) {
    const output = model.forward(trainData.x);
    const loss = criterion.forward(output, trainData.y);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 15 === 0) {
      const testPred = model.forward(testData.x);
      let correct = 0;
      for (let i = 0; i < 100; i++) {
        let predClass = 0, trueClass = 0;
        let maxPred = -Infinity, maxTrue = -Infinity;
        for (let j = 0; j < numClasses; j++) {
          if (testPred.data[i * numClasses + j] > maxPred) {
            maxPred = testPred.data[i * numClasses + j];
            predClass = j;
          }
          if (testData.y.data[i * numClasses + j] > maxTrue) {
            maxTrue = testData.y.data[i * numClasses + j];
            trueClass = j;
          }
        }
        if (predClass === trueClass) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Loss = ${loss.item().toFixed(4)}, Test Acc = ${correct}%`);
    }
  }

  const finalPred = model.forward(testData.x);
  let correct = 0;
  for (let i = 0; i < 100; i++) {
    let predClass = 0, trueClass = 0;
    let maxPred = -Infinity, maxTrue = -Infinity;
    for (let j = 0; j < numClasses; j++) {
      if (finalPred.data[i * numClasses + j] > maxPred) {
        maxPred = finalPred.data[i * numClasses + j];
        predClass = j;
      }
      if (testData.y.data[i * numClasses + j] > maxTrue) {
        maxTrue = testData.y.data[i * numClasses + j];
        trueClass = j;
      }
    }
    if (predClass === trueClass) correct++;
  }
  console.log(`  Final: ${correct}%`);
  return correct / 100;
}

// ============================================
// Test 6: Few-Shot Learning
// ============================================
async function testFewShot(): Promise<number> {
  console.log('\n[6] FEW-SHOT LEARNING (5-way 3-shot)');
  console.log('-'.repeat(50));

  const numClasses = 5;
  const shotsPerClass = 3;
  const queriesPerClass = 10;

  const classifier = new TRMClassifier(6, 64, numClasses, 6);
  const optimizer = new Adam(classifier.parameters(), 0.005);
  const criterion = new CrossEntropyLoss();

  // Class prototypes
  const prototypes = [
    [2, 0, 0, 0, 0, 0],
    [-2, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, -2, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0],
  ];

  function generateSamples(classIdx: number, n: number): number[][] {
    const samples: number[][] = [];
    for (let i = 0; i < n; i++) {
      samples.push(prototypes[classIdx].map(v => v + (Math.random() - 0.5) * 0.8));
    }
    return samples;
  }

  // Pre-train on more data
  console.log('  Pre-training...');
  for (let epoch = 0; epoch < 40; epoch++) {
    const xData: number[][] = [];
    const yData: number[][] = [];

    for (let c = 0; c < numClasses; c++) {
      const samples = generateSamples(c, 30);
      for (const s of samples) {
        xData.push(s);
        const oneHot = new Array(numClasses).fill(0);
        oneHot[c] = 1;
        yData.push(oneHot);
      }
    }

    const x = tensor(xData);
    const y = tensor(yData);

    const output = classifier.forward(x);
    const loss = criterion.forward(output, y);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();
  }

  // Test few-shot
  let totalCorrect = 0;
  let totalQueries = 0;

  for (let episode = 0; episode < 10; episode++) {
    // Support set
    const supportX: number[][] = [];
    const supportY: number[] = [];

    for (let c = 0; c < numClasses; c++) {
      const samples = generateSamples(c, shotsPerClass);
      for (const s of samples) {
        supportX.push(s);
        supportY.push(c);
      }
    }

    // Query set
    const queryX: number[][] = [];
    const queryYTrue: number[] = [];

    for (let c = 0; c < numClasses; c++) {
      const samples = generateSamples(c, queriesPerClass);
      for (const s of samples) {
        queryX.push(s);
        queryYTrue.push(c);
      }
    }

    // Predict
    const result = classifier.fewShotPredict(
      tensor(supportX),
      supportY,
      tensor(queryX)
    );

    for (let i = 0; i < queryYTrue.length; i++) {
      if (result[i] === queryYTrue[i]) totalCorrect++;
      totalQueries++;
    }
  }

  const acc = totalCorrect / totalQueries;
  console.log(`  Final Few-Shot Accuracy: ${(acc * 100).toFixed(1)}%`);
  return acc;
}

// ============================================
// Main
// ============================================
async function main() {
  const results: { name: string; acc: number; target: number }[] = [];

  results.push({ name: 'Noisy XOR', acc: await testNoisyXOR(), target: 0.95 });
  results.push({ name: 'Multi-class', acc: await testMultiClass(), target: 0.90 });
  results.push({ name: 'Arithmetic', acc: await testArithmetic(), target: 0.85 });
  results.push({ name: 'Sequence', acc: await testSequence(), target: 0.85 });
  results.push({ name: 'Simple Text', acc: await testSimpleText(), target: 0.80 });
  results.push({ name: 'Few-Shot', acc: await testFewShot(), target: 0.75 });

  console.log('\n' + '='.repeat(70));
  console.log('FINAL SUMMARY');
  console.log('='.repeat(70));

  let passed = 0;
  for (const r of results) {
    const status = r.acc >= r.target ? 'PASS' : 'FAIL';
    const icon = status === 'PASS' ? '[+]' : '[-]';
    console.log(`${icon} ${r.name.padEnd(15)} ${(r.acc * 100).toFixed(1).padStart(6)}% / ${(r.target * 100).toFixed(0)}%  ${status}`);
    if (r.acc >= r.target) passed++;
  }

  const avgAcc = results.reduce((s, r) => s + r.acc, 0) / results.length;
  console.log('-'.repeat(70));
  console.log(`Tests Passed: ${passed}/${results.length}`);
  console.log(`Average Accuracy: ${(avgAcc * 100).toFixed(1)}%`);
  console.log('='.repeat(70));

  if (passed >= 5) {
    console.log('\nEXCELLENT! TRM is performing at breakthrough level!');
  } else if (passed >= 4) {
    console.log('\nVERY GOOD! TRM is performing well.');
  } else {
    console.log('\nGOOD PROGRESS! Some areas need more optimization.');
  }
}

main().catch(console.error);
