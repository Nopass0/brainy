/**
 * TRM Ultra Quick Test
 */

import {
  tensor,
  TRMUltra,
  createTinyTRMUltra,
  createReasoningTRMUltra,
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

async function testXOR() {
  console.log('\n=== XOR Test ===');

  const model = createTinyTRMUltra(2, 1, 32, 4);
  const opt = new Adam(model.parameters(), 0.05);
  const loss = new MSELoss();

  const xData = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yData = [[0], [1], [1], [0]];

  for (let ep = 0; ep < 500; ep++) {
    for (let i = 0; i < 4; i++) {
      opt.zeroGrad();
      const out = model.forward(tensor([[xData[i][0], xData[i][1]]]), 4);
      const l = loss.forward(out, tensor([[yData[i][0]]]));
      l.backward();
      opt.step();
    }

    if ((ep + 1) % 100 === 0) {
      const pred = model.forward(tensor(xData), 4);
      const acc = binaryAcc(pred, tensor(yData.map(y => y)));
      console.log(`  Epoch ${ep + 1}: Acc = ${(acc * 100).toFixed(0)}%`);
    }
  }

  const pred = model.forward(tensor(xData), 4);
  console.log('  Predictions:');
  for (let i = 0; i < 4; i++) {
    console.log(`    ${xData[i][0]} XOR ${xData[i][1]} = ${pred.data[i].toFixed(3)} (exp: ${yData[i][0]})`);
  }

  return binaryAcc(pred, tensor(yData.map(y => y)));
}

async function testNoisyXOR() {
  console.log('\n=== Noisy XOR Test ===');

  const model = createTinyTRMUltra(2, 1, 64, 6);
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

  const train = genData(300);
  const test = genData(100);

  for (let ep = 0; ep < 50; ep++) {
    shuffle(train.x, train.y);
    let totalLoss = 0;
    for (let i = 0; i < train.x.length; i += 16) {
      const bx = train.x.slice(i, i + 16);
      const by = train.y.slice(i, i + 16);
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
      console.log(`  Epoch ${ep + 1}: Loss = ${(totalLoss / 19).toFixed(4)}, Acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const pred = model.forward(tensor(test.x), 6);
  return binaryAcc(pred, tensor(test.y.map(y => y)));
}

async function testMultiClass() {
  console.log('\n=== Multi-class (3 Gaussians) ===');

  const model = new TRMUltra({
    inputDim: 2,
    hiddenDim: 64,
    outputDim: 3,
    numRecursions: 6,
    useMemory: false,
  });

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
      const acc = classAcc(pred, tensor(test.y));
      console.log(`  Epoch ${ep + 1}: Loss = ${(totalLoss / 10).toFixed(4)}, Acc = ${(acc * 100).toFixed(1)}%`);
    }
  }

  const pred = model.forward(tensor(test.x), 6);
  return classAcc(pred, tensor(test.y));
}

async function testArithmetic() {
  console.log('\n=== Arithmetic (a + b) ===');

  const model = createTinyTRMUltra(2, 1, 64, 6);
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

  const train = genData(500);
  const test = genData(100);

  for (let ep = 0; ep < 60; ep++) {
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

    if ((ep + 1) % 15 === 0) {
      const pred = model.forward(tensor(test.x), 6);
      let c = 0;
      for (let i = 0; i < 100; i++) {
        if (Math.abs(pred.data[i] - test.y[i][0]) < 0.15) c++;
      }
      console.log(`  Epoch ${ep + 1}: Loss = ${(totalLoss / 16).toFixed(4)}, Acc = ${c}%`);
    }
  }

  const pred = model.forward(tensor(test.x), 6);
  let c = 0;
  for (let i = 0; i < 100; i++) {
    if (Math.abs(pred.data[i] - test.y[i][0]) < 0.15) c++;
  }
  return c / 100;
}

async function main() {
  console.log('=' .repeat(50));
  console.log('TRM ULTRA TEST SUITE');
  console.log('='.repeat(50));

  const results: { name: string; acc: number; target: number }[] = [];

  results.push({ name: 'XOR', acc: await testXOR(), target: 1.0 });
  results.push({ name: 'Noisy XOR', acc: await testNoisyXOR(), target: 0.95 });
  results.push({ name: 'Multi-class', acc: await testMultiClass(), target: 0.95 });
  results.push({ name: 'Arithmetic', acc: await testArithmetic(), target: 0.90 });

  console.log('\n' + '='.repeat(50));
  console.log('SUMMARY');
  console.log('='.repeat(50));

  let passed = 0;
  for (const r of results) {
    const status = r.acc >= r.target ? 'PASS' : 'FAIL';
    const icon = status === 'PASS' ? '[+]' : '[-]';
    console.log(`${icon} ${r.name.padEnd(15)} ${(r.acc * 100).toFixed(1).padStart(5)}% / ${(r.target * 100).toFixed(0)}%  ${status}`);
    if (r.acc >= r.target) passed++;
  }

  const avgAcc = results.reduce((s, r) => s + r.acc, 0) / results.length;
  console.log('-'.repeat(50));
  console.log(`Passed: ${passed}/${results.length}, Avg: ${(avgAcc * 100).toFixed(1)}%`);
  console.log('='.repeat(50));
}

main().catch(console.error);
