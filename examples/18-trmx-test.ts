/**
 * TRM-X Test Suite - Using proper training workflow
 *
 * Proper workflow (from working example):
 * 1. Forward pass
 * 2. Compute loss
 * 3. zeroGrad()
 * 4. backward()
 * 5. step()
 */

import {
  tensor,
  TRMX,
  createTinyTRMX,
  createStandardTRMX,
  createReasoningTRMX,
  Adam,
  MSELoss,
  CrossEntropyLoss,
  Tensor,
} from '../src';

function binaryAcc(pred: Tensor, target: Tensor): number {
  let c = 0;
  for (let i = 0; i < pred.shape[0]; i++) {
    if ((pred.data[i] > 0.5 ? 1 : 0) === (target.data[i] > 0.5 ? 1 : 0)) c++;
  }
  return c / pred.shape[0];
}

function classAcc(pred: Tensor, target: Tensor): number {
  const bs = pred.shape[0], nc = pred.shape[1];
  let c = 0;
  for (let i = 0; i < bs; i++) {
    let pi = 0, ti = 0, pm = -Infinity, tm = -Infinity;
    for (let j = 0; j < nc; j++) {
      if (pred.data[i * nc + j] > pm) { pm = pred.data[i * nc + j]; pi = j; }
      if (target.data[i * nc + j] > tm) { tm = target.data[i * nc + j]; ti = j; }
    }
    if (pi === ti) c++;
  }
  return c / bs;
}

// Test 1: XOR (using proper workflow)
async function testXOR() {
  console.log('\n=== TEST 1: XOR ===');

  // Use sigmoid output for binary classification
  const model = createTinyTRMX(2, 1, 8, 4, true);
  const optimizer = new Adam(model.parameters(), 0.1);
  const criterion = new MSELoss();

  const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const Y = tensor([[0], [1], [1], [0]]);

  for (let epoch = 0; epoch < 1000; epoch++) {
    // Forward
    const predictions = model.forward(X, 4);

    // Loss
    const loss = criterion.forward(predictions, Y);

    // Backward (proper order!)
    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 200 === 0) {
      const acc = binaryAcc(predictions, Y);
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(6)}, acc = ${(acc * 100).toFixed(0)}%`);
    }
  }

  const finalPred = model.forward(X, 4);
  console.log('\n  Predictions:');
  for (let i = 0; i < 4; i++) {
    const a = X.data[i * 2], b = X.data[i * 2 + 1];
    const p = finalPred.data[i];
    const exp = Y.data[i];
    console.log(`    ${a} XOR ${b} = ${p.toFixed(3)} (expected: ${exp})`);
  }

  return binaryAcc(finalPred, Y);
}

// Test 2: Noisy XOR
async function testNoisyXOR() {
  console.log('\n=== TEST 2: Noisy XOR ===');

  const model = createStandardTRMX(2, 1, 32, 6);
  model.getConfig(); // Force useSigmoidOutput to false, we'll handle manually

  // Actually create with sigmoid
  const modelWithSig = createTinyTRMX(2, 1, 32, 6, true);
  const optimizer = new Adam(modelWithSig.parameters(), 0.05);
  const criterion = new MSELoss();

  const genData = (n: number) => {
    const x: number[][] = [], y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() > 0.5 ? 1 : 0;
      const b = Math.random() > 0.5 ? 1 : 0;
      x.push([a + (Math.random() - 0.5) * 0.3, b + (Math.random() - 0.5) * 0.3]);
      y.push([a !== b ? 1 : 0]);
    }
    return { x, y };
  };

  const train = genData(200);
  const test = genData(100);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 50; epoch++) {
    const predictions = modelWithSig.forward(trainX, 6);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 10 === 0) {
      const testPred = modelWithSig.forward(tensor(test.x), 6);
      const acc = binaryAcc(testPred, tensor(test.y.map(y => y)));
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const testPred = modelWithSig.forward(tensor(test.x), 6);
  return binaryAcc(testPred, tensor(test.y.map(y => y)));
}

// Test 3: Multi-class (3 Gaussians)
async function testMultiClass() {
  console.log('\n=== TEST 3: Multi-class (3 Gaussians) ===');

  const model = createStandardTRMX(2, 3, 32, 6);
  const optimizer = new Adam(model.parameters(), 0.05);
  const criterion = new CrossEntropyLoss();

  const centers = [[2, 0], [-1, 1.7], [-1, -1.7]];
  const genData = (n: number) => {
    const x: number[][] = [], y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const c = Math.floor(Math.random() * 3);
      x.push([
        centers[c][0] + (Math.random() - 0.5) * 0.8,
        centers[c][1] + (Math.random() - 0.5) * 0.8,
      ]);
      const oh = [0, 0, 0]; oh[c] = 1;
      y.push(oh);
    }
    return { x, y };
  };

  const train = genData(300);
  const test = genData(100);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 50; epoch++) {
    const predictions = model.forward(trainX, 6);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 10 === 0) {
      const testPred = model.forward(tensor(test.x), 6);
      const acc = classAcc(testPred, tensor(test.y));
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const testPred = model.forward(tensor(test.x), 6);
  return classAcc(testPred, tensor(test.y));
}

// Test 4: Arithmetic (a + b)
async function testArithmetic() {
  console.log('\n=== TEST 4: Arithmetic (a + b) ===');

  const model = createStandardTRMX(2, 1, 32, 6);
  const optimizer = new Adam(model.parameters(), 0.02);
  const criterion = new MSELoss();

  const genData = (n: number) => {
    const x: number[][] = [], y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() * 2 - 1;
      const b = Math.random() * 2 - 1;
      x.push([a, b]);
      y.push([a + b]);
    }
    return { x, y };
  };

  const train = genData(500);
  const test = genData(100);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 80; epoch++) {
    const predictions = model.forward(trainX, 6);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 20 === 0) {
      const testPred = model.forward(tensor(test.x), 6);
      let correct = 0;
      for (let i = 0; i < 100; i++) {
        if (Math.abs(testPred.data[i] - test.y[i][0]) < 0.15) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${correct}%`);
    }
  }

  const testPred = model.forward(tensor(test.x), 6);
  let correct = 0;
  for (let i = 0; i < 100; i++) {
    if (Math.abs(testPred.data[i] - test.y[i][0]) < 0.15) correct++;
  }
  return correct / 100;
}

// Test 5: Complex Arithmetic (a + b * c)
async function testComplexArithmetic() {
  console.log('\n=== TEST 5: Complex Arithmetic (a + b * c) ===');

  const model = createReasoningTRMX(3, 1);
  const optimizer = new Adam(model.parameters(), 0.01);
  const criterion = new MSELoss();

  const genData = (n: number) => {
    const x: number[][] = [], y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() * 2 - 1;
      const b = Math.random() * 2 - 1;
      const c = Math.random() * 2 - 1;
      x.push([a, b, c]);
      y.push([a + b * c]);
    }
    return { x, y };
  };

  const train = genData(800);
  const test = genData(200);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 100; epoch++) {
    const predictions = model.forward(trainX, 8);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 25 === 0) {
      const testPred = model.forward(tensor(test.x), 8);
      let correct = 0;
      for (let i = 0; i < 200; i++) {
        if (Math.abs(testPred.data[i] - test.y[i][0]) < 0.2) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${(correct / 2).toFixed(1)}%`);
    }
  }

  const testPred = model.forward(tensor(test.x), 8);
  let correct = 0;
  for (let i = 0; i < 200; i++) {
    if (Math.abs(testPred.data[i] - test.y[i][0]) < 0.2) correct++;
  }
  return correct / 200;
}

// Main
async function main() {
  console.log('='.repeat(60));
  console.log('TRM-X TEST SUITE');
  console.log('='.repeat(60));

  const results: { name: string; acc: number; target: number }[] = [];

  results.push({ name: 'XOR', acc: await testXOR(), target: 1.0 });
  results.push({ name: 'Noisy XOR', acc: await testNoisyXOR(), target: 0.95 });
  results.push({ name: 'Multi-class', acc: await testMultiClass(), target: 0.95 });
  results.push({ name: 'Arithmetic (a+b)', acc: await testArithmetic(), target: 0.90 });
  results.push({ name: 'Complex (a+b*c)', acc: await testComplexArithmetic(), target: 0.80 });

  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  let passed = 0;
  for (const r of results) {
    const status = r.acc >= r.target ? 'PASS' : 'FAIL';
    const icon = status === 'PASS' ? '[+]' : '[-]';
    console.log(`${icon} ${r.name.padEnd(20)} ${(r.acc * 100).toFixed(1).padStart(5)}% / ${(r.target * 100).toFixed(0)}%  ${status}`);
    if (r.acc >= r.target) passed++;
  }

  const avgAcc = results.reduce((s, r) => s + r.acc, 0) / results.length;
  console.log('-'.repeat(60));
  console.log(`Tests Passed: ${passed}/${results.length}`);
  console.log(`Average Accuracy: ${(avgAcc * 100).toFixed(1)}%`);
  console.log('='.repeat(60));

  if (passed >= 4) {
    console.log('\nTRM-X is performing excellently!');
  } else if (passed >= 3) {
    console.log('\nTRM-X is performing well, some improvement possible.');
  } else {
    console.log('\nNeeds more optimization.');
  }
}

main().catch(console.error);
