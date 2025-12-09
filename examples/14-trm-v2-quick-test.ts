/**
 * @fileoverview TRM v2 Quick Tests - Faster version for initial validation
 */

import {
  tensor,
  randn,
  TRMv2,
  TRMv2Classifier,
  createTinyTRMv2,
  Adam,
  MSELoss,
  CrossEntropyLoss,
} from '../src';
import { Tensor } from '../src/core/tensor';

// Utility functions
function binaryAccuracy(predictions: Tensor, targets: Tensor): number {
  const batchSize = predictions.shape[0];
  let correct = 0;
  for (let i = 0; i < batchSize; i++) {
    const pred = predictions.data[i] > 0.5 ? 1 : 0;
    const target = targets.data[i] > 0.5 ? 1 : 0;
    if (pred === target) correct++;
  }
  return correct / batchSize;
}

function classAccuracy(predictions: Tensor, targets: Tensor): number {
  const batchSize = predictions.shape[0];
  const numClasses = predictions.shape[1];
  let correct = 0;
  for (let i = 0; i < batchSize; i++) {
    let predClass = 0, targetClass = 0;
    let maxPred = -Infinity, maxTarget = -Infinity;
    for (let j = 0; j < numClasses; j++) {
      const predVal = predictions.data[i * numClasses + j];
      const targetVal = targets.data[i * numClasses + j];
      if (predVal > maxPred) { maxPred = predVal; predClass = j; }
      if (targetVal > maxTarget) { maxTarget = targetVal; targetClass = j; }
    }
    if (predClass === targetClass) correct++;
  }
  return correct / batchSize;
}

function shuffleArrays(x: number[][], y: number[][]): void {
  for (let i = x.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [x[i], x[j]] = [x[j], x[i]];
    [y[i], y[j]] = [y[j], y[i]];
  }
}

// Test 1: XOR
async function testXOR() {
  console.log('\n=== TEST 1: XOR Problem ===');

  const model = createTinyTRMv2(2, 1, 32, 4);
  const optimizer = new Adam(model.parameters(), 0.02);
  const lossFunc = new MSELoss();

  const xData = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yData = [[0], [1], [1], [0]];

  for (let epoch = 0; epoch < 500; epoch++) {
    for (let i = 0; i < 4; i++) {
      const x = tensor([[xData[i][0], xData[i][1]]]);
      const y = tensor([[yData[i][0]]]);
      optimizer.zeroGrad();
      const output = model.forward(x, 4);
      const loss = lossFunc.forward(output, y);
      loss.backward();
      optimizer.step();
    }

    if ((epoch + 1) % 100 === 0) {
      const testX = tensor(xData);
      const testY = tensor(yData);
      const preds = model.forward(testX, 4);
      const acc = binaryAccuracy(preds, testY);
      console.log(`  Epoch ${epoch + 1}: Accuracy = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const finalPreds = model.forward(tensor(xData), 4);
  const finalAcc = binaryAccuracy(finalPreds, tensor(yData));
  console.log(`  Final: ${(finalAcc * 100).toFixed(1)}% (target: 100%)`);
  return { name: 'XOR', accuracy: finalAcc, target: 1.0 };
}

// Test 2: Multi-class
async function testMultiClass() {
  console.log('\n=== TEST 2: Multi-class (3 Gaussians) ===');

  const model = new TRMv2({
    inputDim: 2,
    hiddenDim: 64,
    outputDim: 3,
    numRecursions: 6,
    numHeads: 4,
    numExperts: 2,
    memorySlots: 16,
  });

  const optimizer = new Adam(model.parameters(), 0.01);
  const lossFunc = new CrossEntropyLoss();

  // Generate data - 3 well-separated clusters
  const centers = [[2, 0], [-1, 1.7], [-1, -1.7]];
  const generateData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const c = Math.floor(Math.random() * 3);
      x.push([
        centers[c][0] + (Math.random() - 0.5) * 0.8,
        centers[c][1] + (Math.random() - 0.5) * 0.8,
      ]);
      const oneHot = [0, 0, 0];
      oneHot[c] = 1;
      y.push(oneHot);
    }
    return { x, y };
  };

  const train = generateData(300);
  const test = generateData(100);

  for (let epoch = 0; epoch < 50; epoch++) {
    shuffleArrays(train.x, train.y);
    let totalLoss = 0;
    const bs = 32;

    for (let i = 0; i < train.x.length; i += bs) {
      const bx = train.x.slice(i, i + bs);
      const by = train.y.slice(i, i + bs);
      if (bx.length === 0) continue;

      optimizer.zeroGrad();
      const output = model.forward(tensor(bx), 6);
      const loss = lossFunc.forward(output, tensor(by));
      loss.backward();
      optimizer.step();
      totalLoss += loss.item();
    }

    if ((epoch + 1) % 10 === 0) {
      const preds = model.forward(tensor(test.x), 6);
      const acc = classAccuracy(preds, tensor(test.y));
      console.log(`  Epoch ${epoch + 1}: Loss=${(totalLoss / 10).toFixed(3)}, Acc=${(acc * 100).toFixed(1)}%`);
    }
  }

  const finalPreds = model.forward(tensor(test.x), 6);
  const finalAcc = classAccuracy(finalPreds, tensor(test.y));
  console.log(`  Final: ${(finalAcc * 100).toFixed(1)}% (target: 95%)`);
  return { name: 'Multi-class', accuracy: finalAcc, target: 0.95 };
}

// Test 3: Arithmetic (a + b)
async function testArithmetic() {
  console.log('\n=== TEST 3: Arithmetic (a + b) ===');

  const model = createTinyTRMv2(2, 1, 64, 6);
  const optimizer = new Adam(model.parameters(), 0.005);
  const lossFunc = new MSELoss();

  const generateData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() * 2 - 1;
      const b = Math.random() * 2 - 1;
      x.push([a, b]);
      y.push([a + b]);
    }
    return { x, y };
  };

  const train = generateData(500);
  const test = generateData(100);

  for (let epoch = 0; epoch < 80; epoch++) {
    shuffleArrays(train.x, train.y);
    let totalLoss = 0;
    const bs = 32;

    for (let i = 0; i < train.x.length; i += bs) {
      const bx = train.x.slice(i, i + bs);
      const by = train.y.slice(i, i + bs);
      if (bx.length === 0) continue;

      optimizer.zeroGrad();
      const output = model.forward(tensor(bx), 6);
      const loss = lossFunc.forward(output, tensor(by));
      loss.backward();
      optimizer.step();
      totalLoss += loss.item();
    }

    if ((epoch + 1) % 20 === 0) {
      const preds = model.forward(tensor(test.x), 6);
      let correct = 0;
      for (let i = 0; i < test.y.length; i++) {
        if (Math.abs(preds.data[i] - test.y[i][0]) < 0.15) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Loss=${(totalLoss / 16).toFixed(4)}, Acc=${((correct / 100) * 100).toFixed(1)}%`);
    }
  }

  const finalPreds = model.forward(tensor(test.x), 6);
  let correct = 0;
  for (let i = 0; i < test.y.length; i++) {
    if (Math.abs(finalPreds.data[i] - test.y[i][0]) < 0.15) correct++;
  }
  const finalAcc = correct / 100;
  console.log(`  Final: ${(finalAcc * 100).toFixed(1)}% (target: 95%)`);
  return { name: 'Arithmetic', accuracy: finalAcc, target: 0.95 };
}

// Test 4: Noisy XOR
async function testNoisyXOR() {
  console.log('\n=== TEST 4: Noisy XOR ===');

  const model = createTinyTRMv2(2, 1, 64, 6);
  const optimizer = new Adam(model.parameters(), 0.01);
  const lossFunc = new MSELoss();

  const generateData = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() > 0.5 ? 1 : 0;
      const b = Math.random() > 0.5 ? 1 : 0;
      x.push([a + (Math.random() - 0.5) * 0.3, b + (Math.random() - 0.5) * 0.3]);
      y.push([a !== b ? 1 : 0]);
    }
    return { x, y };
  };

  const train = generateData(400);
  const test = generateData(100);

  for (let epoch = 0; epoch < 60; epoch++) {
    shuffleArrays(train.x, train.y);
    let totalLoss = 0;
    const bs = 32;

    for (let i = 0; i < train.x.length; i += bs) {
      const bx = train.x.slice(i, i + bs);
      const by = train.y.slice(i, i + bs);
      if (bx.length === 0) continue;

      optimizer.zeroGrad();
      const output = model.forward(tensor(bx), 6);
      const loss = lossFunc.forward(output, tensor(by));
      loss.backward();
      optimizer.step();
      totalLoss += loss.item();
    }

    if ((epoch + 1) % 15 === 0) {
      const preds = model.forward(tensor(test.x), 6);
      const acc = binaryAccuracy(preds, tensor(test.y.map(y => y)));
      console.log(`  Epoch ${epoch + 1}: Loss=${(totalLoss / 13).toFixed(4)}, Acc=${(acc * 100).toFixed(1)}%`);
    }
  }

  const finalPreds = model.forward(tensor(test.x), 6);
  const finalAcc = binaryAccuracy(finalPreds, tensor(test.y.map(y => y)));
  console.log(`  Final: ${(finalAcc * 100).toFixed(1)}% (target: 95%)`);
  return { name: 'Noisy XOR', accuracy: finalAcc, target: 0.95 };
}

// Test 5: Adaptive computation
async function testAdaptive() {
  console.log('\n=== TEST 5: Adaptive Computation ===');

  const model = new TRMv2({
    inputDim: 2,
    hiddenDim: 64,
    outputDim: 2,
    numRecursions: 8,
    numHeads: 4,
    numExperts: 2,
    memorySlots: 16,
    adaptiveComputation: true,
    convergenceThreshold: 0.01,
    maxAdaptiveSteps: 16,
  });

  const optimizer = new Adam(model.parameters(), 0.01);
  const lossFunc = new CrossEntropyLoss();

  // Easy: well separated. Hard: overlapping
  const generateEasy = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const c = Math.random() > 0.5 ? 1 : 0;
      x.push(c === 0 ? [2, 2] : [-2, -2]);
      y.push(c === 0 ? [1, 0] : [0, 1]);
    }
    return { x, y };
  };

  const generateHard = (n: number) => {
    const x: number[][] = [];
    const y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const c = Math.random() > 0.5 ? 1 : 0;
      const noise = (Math.random() - 0.5) * 2;
      x.push(c === 0 ? [0.3 + noise, 0.3 + noise] : [-0.3 + noise, -0.3 + noise]);
      y.push(c === 0 ? [1, 0] : [0, 1]);
    }
    return { x, y };
  };

  const train = { x: [...generateEasy(200).x, ...generateHard(200).x],
                  y: [...generateEasy(200).y, ...generateHard(200).y] };

  for (let epoch = 0; epoch < 40; epoch++) {
    shuffleArrays(train.x, train.y);
    for (let i = 0; i < train.x.length; i += 32) {
      const bx = train.x.slice(i, i + 32);
      const by = train.y.slice(i, i + 32);
      if (bx.length === 0) continue;

      optimizer.zeroGrad();
      const output = model.forward(tensor(bx), 8);
      const loss = lossFunc.forward(output, tensor(by));
      loss.backward();
      optimizer.step();
    }
  }

  // Test adaptive steps
  const testEasy = generateEasy(50);
  const testHard = generateHard(50);

  let easySteps = 0, hardSteps = 0;
  let easyCorrect = 0, hardCorrect = 0;

  for (let i = 0; i < 50; i++) {
    const resultEasy = model.forwardAdaptive(tensor([testEasy.x[i]]), 16, 0.01);
    easySteps += resultEasy.steps;
    const predEasy = resultEasy.output.data[0] > resultEasy.output.data[1] ? 0 : 1;
    if (predEasy === (testEasy.y[i][0] > 0.5 ? 0 : 1)) easyCorrect++;

    const resultHard = model.forwardAdaptive(tensor([testHard.x[i]]), 16, 0.01);
    hardSteps += resultHard.steps;
    const predHard = resultHard.output.data[0] > resultHard.output.data[1] ? 0 : 1;
    if (predHard === (testHard.y[i][0] > 0.5 ? 0 : 1)) hardCorrect++;
  }

  console.log(`  Easy: steps=${(easySteps / 50).toFixed(1)}, acc=${((easyCorrect / 50) * 100).toFixed(1)}%`);
  console.log(`  Hard: steps=${(hardSteps / 50).toFixed(1)}, acc=${((hardCorrect / 50) * 100).toFixed(1)}%`);

  const avgAcc = (easyCorrect + hardCorrect) / 100;
  console.log(`  Final: ${(avgAcc * 100).toFixed(1)}% (target: 85%)`);
  return { name: 'Adaptive', accuracy: avgAcc, target: 0.85 };
}

// Run all tests
async function main() {
  console.log('\n' + '='.repeat(50));
  console.log('TRM v2 QUICK TEST SUITE');
  console.log('='.repeat(50));

  const results = [];

  results.push(await testXOR());
  results.push(await testMultiClass());
  results.push(await testArithmetic());
  results.push(await testNoisyXOR());
  results.push(await testAdaptive());

  console.log('\n' + '='.repeat(50));
  console.log('SUMMARY');
  console.log('='.repeat(50));

  let passed = 0;
  let avgAcc = 0;

  for (const r of results) {
    const status = r.accuracy >= r.target ? 'PASS' : 'FAIL';
    const icon = status === 'PASS' ? '[+]' : '[-]';
    console.log(`${icon} ${r.name.padEnd(15)} ${(r.accuracy * 100).toFixed(1).padStart(5)}% / ${(r.target * 100).toFixed(0)}%  ${status}`);
    if (r.accuracy >= r.target) passed++;
    avgAcc += r.accuracy;
  }

  avgAcc /= results.length;
  console.log('-'.repeat(50));
  console.log(`Passed: ${passed}/${results.length}, Avg: ${(avgAcc * 100).toFixed(1)}%`);
  console.log('='.repeat(50));
}

main().catch(console.error);
