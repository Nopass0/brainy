/**
 * Exact copy of working TRM example for verification
 */

import {
  Tensor,
  TRM,
  createTinyTRM,
  Adam,
} from '../src';

console.log('='.repeat(50));
console.log('TRM EXACT COPY TEST');
console.log('='.repeat(50));

// Exact copy from 13-trm-reasoning.ts
function generateNoisyXOR(n: number, noiseLevel: number = 0.1): { x: Tensor; y: Tensor } {
  const xData = new Float32Array(n * 2);
  const yData = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const xorResult = a !== b ? 1 : 0;

    // Add noise to inputs
    xData[i * 2] = a + (Math.random() - 0.5) * noiseLevel;
    xData[i * 2 + 1] = b + (Math.random() - 0.5) * noiseLevel;
    yData[i] = xorResult;
  }

  return {
    x: new Tensor(xData, [n, 2], { requiresGrad: true }),
    y: new Tensor(yData, [n, 1]),
  };
}

// Exact same model and training as working example
const trmModel = createTinyTRM(2, 1, 32, 4);  // 4 recursions
const trmOptimizer = new Adam(trmModel.parameters(), 0.01);

console.log('\nTraining TRM on noisy XOR (exact copy)...');

const trainData = generateNoisyXOR(200, 0.3);
const epochs = 50;

for (let epoch = 0; epoch < epochs; epoch++) {
  // TRM training - exact same as working example
  const trmOut = trmModel.forward(trainData.x);
  const trmL = trmOut.sub(trainData.y).pow(2).mean();
  trmOptimizer.zeroGrad();
  trmL.backward();
  trmOptimizer.step();

  if ((epoch + 1) % 10 === 0) {
    console.log(`  Epoch ${epoch + 1}: TRM Loss = ${trmL.item().toFixed(4)}`);
  }
}

// Testing
const testData = generateNoisyXOR(50, 0.3);
const trmPred = trmModel.forward(testData.x);

let trmCorrect = 0;
for (let i = 0; i < 50; i++) {
  const trmAnswer = trmPred.data[i] > 0.5 ? 1 : 0;
  const actual = testData.y.data[i];
  if (trmAnswer === actual) trmCorrect++;
}

console.log(`\nTest Accuracy: ${(trmCorrect / 50 * 100).toFixed(1)}%`);
console.log('='.repeat(50));
