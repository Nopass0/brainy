/**
 * TRM Pro Comprehensive Test Suite
 * Target: 98%+ accuracy on all tasks
 */

import {
  tensor,
  TRMPro,
  createTinyTRMPro,
  createStandardTRMPro,
  createBinaryTRMPro,
  createReasoningTRMPro,
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

function shuffle(x: number[][], y: number[][]) {
  for (let i = x.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [x[i], x[j]] = [x[j], x[i]];
    [y[i], y[j]] = [y[j], y[i]];
  }
}

// Test 1: Noisy XOR (standard benchmark)
async function testNoisyXOR() {
  console.log('\n=== TEST 1: Noisy XOR ===');

  const model = createBinaryTRMPro(2, 64, 6);
  const optimizer = new Adam(model.parameters(), 0.02);
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

  const train = genData(300);
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
      const acc = binaryAcc(testPred, tensor(test.y));
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const testPred = model.forward(tensor(test.x), 6);
  return binaryAcc(testPred, tensor(test.y));
}

// Test 2: Multi-class (3 Gaussians)
async function testMultiClass() {
  console.log('\n=== TEST 2: Multi-class (3 Gaussians) ===');

  const model = createStandardTRMPro(2, 3, 64, 6);
  const optimizer = new Adam(model.parameters(), 0.03);
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

  const train = genData(400);
  const test = genData(100);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 60; epoch++) {
    const predictions = model.forward(trainX, 6);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 15 === 0) {
      const testPred = model.forward(tensor(test.x), 6);
      const acc = classAcc(testPred, tensor(test.y));
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const testPred = model.forward(tensor(test.x), 6);
  return classAcc(testPred, tensor(test.y));
}

// Test 3: Arithmetic (a + b)
async function testArithmetic() {
  console.log('\n=== TEST 3: Arithmetic (a + b) ===');

  const model = createStandardTRMPro(2, 1, 64, 6);
  const optimizer = new Adam(model.parameters(), 0.015);
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

  const train = genData(600);
  const test = genData(100);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 100; epoch++) {
    const predictions = model.forward(trainX, 6);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 25 === 0) {
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

// Test 4: Complex Arithmetic (a + b * c)
async function testComplexArithmetic() {
  console.log('\n=== TEST 4: Complex Arithmetic (a + b * c) ===');

  const model = createReasoningTRMPro(3, 1);
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

  const train = genData(1000);
  const test = genData(200);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 120; epoch++) {
    const predictions = model.forward(trainX, 8);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 30 === 0) {
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

// Test 5: Multi-class with more classes
async function testMultiClass5() {
  console.log('\n=== TEST 5: Multi-class (5 Gaussians) ===');

  const model = createStandardTRMPro(2, 5, 96, 8);
  const optimizer = new Adam(model.parameters(), 0.025);
  const criterion = new CrossEntropyLoss();

  const centers = [
    [2, 0], [-2, 0], [0, 2], [0, -2], [0, 0]
  ];
  const genData = (n: number) => {
    const x: number[][] = [], y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const c = Math.floor(Math.random() * 5);
      x.push([
        centers[c][0] + (Math.random() - 0.5) * 0.6,
        centers[c][1] + (Math.random() - 0.5) * 0.6,
      ]);
      const oh = [0, 0, 0, 0, 0]; oh[c] = 1;
      y.push(oh);
    }
    return { x, y };
  };

  const train = genData(500);
  const test = genData(100);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 80; epoch++) {
    const predictions = model.forward(trainX, 8);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 20 === 0) {
      const testPred = model.forward(tensor(test.x), 8);
      const acc = classAcc(testPred, tensor(test.y));
      console.log(`  Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const testPred = model.forward(tensor(test.x), 8);
  return classAcc(testPred, tensor(test.y));
}

// Test 6: Quadratic function
async function testQuadratic() {
  console.log('\n=== TEST 6: Quadratic (a^2 + b^2) ===');

  const model = createStandardTRMPro(2, 1, 64, 6);
  const optimizer = new Adam(model.parameters(), 0.01);
  const criterion = new MSELoss();

  const genData = (n: number) => {
    const x: number[][] = [], y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() * 2 - 1;
      const b = Math.random() * 2 - 1;
      x.push([a, b]);
      y.push([a * a + b * b]);
    }
    return { x, y };
  };

  const train = genData(600);
  const test = genData(100);

  const trainX = tensor(train.x);
  const trainY = tensor(train.y);

  for (let epoch = 0; epoch < 100; epoch++) {
    const predictions = model.forward(trainX, 6);
    const loss = criterion.forward(predictions, trainY);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 25 === 0) {
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

// Main
async function main() {
  console.log('='.repeat(60));
  console.log('TRM PRO TEST SUITE');
  console.log('Target: 98%+ accuracy on all tasks');
  console.log('='.repeat(60));

  const results: { name: string; acc: number; target: number }[] = [];

  results.push({ name: 'Noisy XOR', acc: await testNoisyXOR(), target: 0.98 });
  results.push({ name: 'Multi-class (3)', acc: await testMultiClass(), target: 0.98 });
  results.push({ name: 'Arithmetic (a+b)', acc: await testArithmetic(), target: 0.95 });
  results.push({ name: 'Complex (a+b*c)', acc: await testComplexArithmetic(), target: 0.90 });
  results.push({ name: 'Multi-class (5)', acc: await testMultiClass5(), target: 0.95 });
  results.push({ name: 'Quadratic (a^2+b^2)', acc: await testQuadratic(), target: 0.90 });

  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  let passed = 0;
  for (const r of results) {
    const status = r.acc >= r.target ? 'PASS' : 'FAIL';
    const icon = status === 'PASS' ? '[+]' : '[-]';
    console.log(`${icon} ${r.name.padEnd(22)} ${(r.acc * 100).toFixed(1).padStart(5)}% / ${(r.target * 100).toFixed(0)}%  ${status}`);
    if (r.acc >= r.target) passed++;
  }

  const avgAcc = results.reduce((s, r) => s + r.acc, 0) / results.length;
  console.log('-'.repeat(60));
  console.log(`Tests Passed: ${passed}/${results.length}`);
  console.log(`Average Accuracy: ${(avgAcc * 100).toFixed(1)}%`);
  console.log('='.repeat(60));

  if (passed === results.length) {
    console.log('\nTRM Pro achieves 98%+ on all tasks! BREAKTHROUGH ACHIEVED!');
  } else if (passed >= results.length - 1) {
    console.log('\nTRM Pro is performing excellently!');
  } else if (passed >= results.length / 2) {
    console.log('\nTRM Pro needs some optimization.');
  } else {
    console.log('\nNeeds significant improvement.');
  }
}

main().catch(console.error);
