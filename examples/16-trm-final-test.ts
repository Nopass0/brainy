/**
 * TRM Final - Comprehensive Test Suite
 * Tests all TRM variants to find the best one
 */

import {
  tensor,
  TRMFinal,
  createTinyTRMFinal,
  createStandardTRMFinal,
  createReasoningTRMFinal,
  TRM,
  createTinyTRM,
  Adam,
  MSELoss,
  CrossEntropyLoss,
} from '../src';
import { Tensor } from '../src/core/tensor';

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

// XOR with data augmentation
async function testXOR() {
  console.log('\n=== TEST: XOR with Augmentation ===');

  const model = createTinyTRMFinal(2, 1, 32, 4);
  const opt = new Adam(model.parameters(), 0.05);
  const loss = new MSELoss();

  // Augmented XOR data (repeat + noise)
  const baseXOR = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const baseY = [[0], [1], [1], [0]];

  const augment = (x: number[], noise: number = 0.1): number[] => {
    return x.map(v => v + (Math.random() - 0.5) * noise * 2);
  };

  for (let ep = 0; ep < 400; ep++) {
    // Each epoch: augment data
    for (let rep = 0; rep < 8; rep++) {
      for (let i = 0; i < 4; i++) {
        const ax = augment(baseXOR[i], 0.15);
        opt.zeroGrad();
        const out = model.forward(tensor([[ax[0], ax[1]]]), 4);
        const l = loss.forward(out, tensor([[baseY[i][0]]]));
        l.backward();
        opt.step();
      }
    }

    if ((ep + 1) % 100 === 0) {
      const pred = model.forward(tensor(baseXOR), 4);
      const acc = binaryAcc(pred, tensor(baseY.map(y => y)));
      console.log(`  Epoch ${ep + 1}: Acc = ${(acc * 100).toFixed(0)}%`);

      // Show predictions
      if ((ep + 1) === 400) {
        console.log('  Predictions:');
        for (let i = 0; i < 4; i++) {
          const p = pred.data[i];
          console.log(`    ${baseXOR[i][0]} XOR ${baseXOR[i][1]} = ${p.toFixed(3)} (exp: ${baseY[i][0]})`);
        }
      }
    }
  }

  const pred = model.forward(tensor(baseXOR), 4);
  return binaryAcc(pred, tensor(baseY.map(y => y)));
}

// Noisy XOR
async function testNoisyXOR() {
  console.log('\n=== TEST: Noisy XOR ===');

  const model = createStandardTRMFinal(2, 1, 64, 6);
  const opt = new Adam(model.parameters(), 0.02);
  const loss = new MSELoss();

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

  const train = genData(400);
  const test = genData(100);

  for (let ep = 0; ep < 40; ep++) {
    shuffle(train.x, train.y);
    let totalLoss = 0;
    for (let i = 0; i < train.x.length; i += 32) {
      const bx = train.x.slice(i, i + 32);
      const by = train.y.slice(i, i + 32);
      if (bx.length === 0) continue;

      opt.zeroGrad();
      const out = model.forward(tensor(bx), 6);
      const l = loss.forward(out, tensor(by));
      l.backward();
      opt.step();
      totalLoss += l.item();
    }

    if ((ep + 1) % 10 === 0) {
      const pred = model.forward(tensor(test.x), 6);
      const acc = binaryAcc(pred, tensor(test.y.map(y => y)));
      console.log(`  Epoch ${ep + 1}: Loss = ${(totalLoss / 13).toFixed(4)}, Acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const pred = model.forward(tensor(test.x), 6);
  return binaryAcc(pred, tensor(test.y.map(y => y)));
}

// Multi-class
async function testMultiClass() {
  console.log('\n=== TEST: Multi-class (3 Gaussians) ===');

  const model = createStandardTRMFinal(2, 3, 64, 6);
  const opt = new Adam(model.parameters(), 0.02);
  const loss = new CrossEntropyLoss();

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

  for (let ep = 0; ep < 30; ep++) {
    shuffle(train.x, train.y);
    let totalLoss = 0;
    for (let i = 0; i < train.x.length; i += 32) {
      const bx = train.x.slice(i, i + 32);
      const by = train.y.slice(i, i + 32);
      if (bx.length === 0) continue;

      opt.zeroGrad();
      const out = model.forward(tensor(bx), 6);
      const l = loss.forward(out, tensor(by));
      l.backward();
      opt.step();
      totalLoss += l.item();
    }

    if ((ep + 1) % 10 === 0) {
      const pred = model.forward(tensor(test.x), 6);
      const acc = classAcc(pred, tensor(test.y));
      console.log(`  Epoch ${ep + 1}: Loss = ${(totalLoss / 10).toFixed(4)}, Acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const pred = model.forward(tensor(test.x), 6);
  return classAcc(pred, tensor(test.y));
}

// Arithmetic: a + b
async function testArithmetic() {
  console.log('\n=== TEST: Arithmetic (a + b) ===');

  const model = createStandardTRMFinal(2, 1, 64, 6);
  const opt = new Adam(model.parameters(), 0.01);
  const loss = new MSELoss();

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

  for (let ep = 0; ep < 50; ep++) {
    shuffle(train.x, train.y);
    let totalLoss = 0;
    for (let i = 0; i < train.x.length; i += 32) {
      const bx = train.x.slice(i, i + 32);
      const by = train.y.slice(i, i + 32);
      if (bx.length === 0) continue;

      opt.zeroGrad();
      const out = model.forward(tensor(bx), 6);
      const l = loss.forward(out, tensor(by));
      l.backward();
      opt.step();
      totalLoss += l.item();
    }

    if ((ep + 1) % 10 === 0) {
      const pred = model.forward(tensor(test.x), 6);
      let c = 0;
      for (let i = 0; i < 100; i++) {
        if (Math.abs(pred.data[i] - test.y[i][0]) < 0.15) c++;
      }
      console.log(`  Epoch ${ep + 1}: Loss = ${(totalLoss / 19).toFixed(4)}, Acc = ${c}%`);
    }
  }

  const pred = model.forward(tensor(test.x), 6);
  let c = 0;
  for (let i = 0; i < 100; i++) {
    if (Math.abs(pred.data[i] - test.y[i][0]) < 0.15) c++;
  }
  return c / 100;
}

// Complex arithmetic: a + b * c
async function testComplexArithmetic() {
  console.log('\n=== TEST: Complex Arithmetic (a + b * c) ===');

  const model = createReasoningTRMFinal(3, 1);
  const opt = new Adam(model.parameters(), 0.008);
  const loss = new MSELoss();

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

  for (let ep = 0; ep < 60; ep++) {
    shuffle(train.x, train.y);
    let totalLoss = 0;
    for (let i = 0; i < train.x.length; i += 32) {
      const bx = train.x.slice(i, i + 32);
      const by = train.y.slice(i, i + 32);
      if (bx.length === 0) continue;

      opt.zeroGrad();
      const out = model.forward(tensor(bx), 8);
      const l = loss.forward(out, tensor(by));
      l.backward();
      opt.step();
      totalLoss += l.item();
    }

    if ((ep + 1) % 15 === 0) {
      const pred = model.forward(tensor(test.x), 8);
      let c = 0;
      for (let i = 0; i < 200; i++) {
        if (Math.abs(pred.data[i] - test.y[i][0]) < 0.2) c++;
      }
      console.log(`  Epoch ${ep + 1}: Loss = ${(totalLoss / 32).toFixed(4)}, Acc = ${(c / 2).toFixed(1)}%`);
    }
  }

  const pred = model.forward(tensor(test.x), 8);
  let c = 0;
  for (let i = 0; i < 200; i++) {
    if (Math.abs(pred.data[i] - test.y[i][0]) < 0.2) c++;
  }
  return c / 200;
}

// Compare with original TRM
async function compareWithOriginal() {
  console.log('\n=== COMPARISON: TRM Final vs TRM Original ===');

  // Test on Noisy XOR
  const genData = (n: number) => {
    const x: number[][] = [], y: number[][] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() > 0.5 ? 1 : 0;
      const b = Math.random() > 0.5 ? 1 : 0;
      x.push([a + (Math.random() - 0.5) * 0.25, b + (Math.random() - 0.5) * 0.25]);
      y.push([a !== b ? 1 : 0]);
    }
    return { x, y };
  };

  const train = genData(300);
  const test = genData(100);
  const loss = new MSELoss();

  // TRM Final
  console.log('  Training TRM Final...');
  const modelFinal = createStandardTRMFinal(2, 1, 64, 6);
  const optFinal = new Adam(modelFinal.parameters(), 0.02);

  for (let ep = 0; ep < 30; ep++) {
    shuffle(train.x, train.y);
    for (let i = 0; i < train.x.length; i += 32) {
      const bx = train.x.slice(i, i + 32);
      const by = train.y.slice(i, i + 32);
      if (bx.length === 0) continue;

      optFinal.zeroGrad();
      const out = modelFinal.forward(tensor(bx), 6);
      const l = loss.forward(out, tensor(by));
      l.backward();
      optFinal.step();
    }
  }

  const predFinal = modelFinal.forward(tensor(test.x), 6);
  const accFinal = binaryAcc(predFinal, tensor(test.y.map(y => y)));
  console.log(`  TRM Final Accuracy: ${(accFinal * 100).toFixed(1)}%`);

  // TRM Original
  console.log('  Training TRM Original...');
  const modelOrig = createTinyTRM(2, 1, 64, 6);
  const optOrig = new Adam(modelOrig.parameters(), 0.02);

  for (let ep = 0; ep < 30; ep++) {
    shuffle(train.x, train.y);
    for (let i = 0; i < train.x.length; i += 32) {
      const bx = train.x.slice(i, i + 32);
      const by = train.y.slice(i, i + 32);
      if (bx.length === 0) continue;

      optOrig.zeroGrad();
      const out = modelOrig.forward(tensor(bx), 6);
      const l = loss.forward(out, tensor(by));
      l.backward();
      optOrig.step();
    }
  }

  const predOrig = modelOrig.forward(tensor(test.x), 6);
  const accOrig = binaryAcc(predOrig, tensor(test.y.map(y => y)));
  console.log(`  TRM Original Accuracy: ${(accOrig * 100).toFixed(1)}%`);

  return { final: accFinal, original: accOrig };
}

async function main() {
  console.log('='.repeat(60));
  console.log('TRM FINAL - COMPREHENSIVE TEST SUITE');
  console.log('='.repeat(60));

  const results: { name: string; acc: number; target: number }[] = [];

  results.push({ name: 'XOR (augmented)', acc: await testXOR(), target: 1.0 });
  results.push({ name: 'Noisy XOR', acc: await testNoisyXOR(), target: 0.95 });
  results.push({ name: 'Multi-class', acc: await testMultiClass(), target: 0.95 });
  results.push({ name: 'Arithmetic (a+b)', acc: await testArithmetic(), target: 0.90 });
  results.push({ name: 'Complex (a+b*c)', acc: await testComplexArithmetic(), target: 0.85 });

  const comparison = await compareWithOriginal();

  console.log('\n' + '='.repeat(60));
  console.log('FINAL SUMMARY');
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
  console.log(`TRM Final vs Original: ${(comparison.final * 100).toFixed(1)}% vs ${(comparison.original * 100).toFixed(1)}%`);
  console.log('='.repeat(60));

  if (passed >= 4) {
    console.log('\nSUCCESS! TRM Final is performing excellently!');
  } else {
    console.log('\nNeeds improvement. Analyzing failure cases...');
  }
}

main().catch(console.error);
