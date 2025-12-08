/**
 * Minimal XOR Test - Debug why XOR doesn't work
 */

import {
  tensor,
  Linear,
  Sequential,
  GELU,
  ReLU,
  Sigmoid,
  Tanh,
  Adam,
  SGD,
  MSELoss,
  Module,
} from '../src';
import { Tensor } from '../src/core/tensor';

// Simple MLP (no TRM)
class SimpleMLP extends Module {
  private fc1: Linear;
  private fc2: Linear;
  private fc3: Linear;
  private act: ReLU;

  constructor(inputDim: number, hiddenDim: number, outputDim: number) {
    super();
    this.fc1 = new Linear(inputDim, hiddenDim);
    this.fc2 = new Linear(hiddenDim, hiddenDim);
    this.fc3 = new Linear(hiddenDim, outputDim);
    this.act = new ReLU();

    this.registerModule('fc1', this.fc1);
    this.registerModule('fc2', this.fc2);
    this.registerModule('fc3', this.fc3);
  }

  forward(x: Tensor): Tensor {
    let h = this.fc1.forward(x);
    h = this.act.forward(h);
    h = this.fc2.forward(h);
    h = this.act.forward(h);
    return this.fc3.forward(h);
  }
}

async function testSimpleMLP() {
  console.log('=== Simple MLP on XOR ===\n');

  const model = new SimpleMLP(2, 32, 1);
  const opt = new Adam(model.parameters(), 0.1);
  const loss = new MSELoss();

  const xData = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yData = [[0], [1], [1], [0]];

  for (let epoch = 0; epoch < 1000; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < 4; i++) {
      opt.zeroGrad();
      const x = tensor([[xData[i][0], xData[i][1]]]);
      const y = tensor([[yData[i][0]]]);

      const out = model.forward(x);
      const l = loss.forward(out, y);
      l.backward();
      opt.step();

      totalLoss += l.item();
    }

    if ((epoch + 1) % 200 === 0) {
      // Test
      let correct = 0;
      for (let i = 0; i < 4; i++) {
        const x = tensor([[xData[i][0], xData[i][1]]]);
        const pred = model.forward(x);
        const p = pred.data[0] > 0.5 ? 1 : 0;
        if (p === yData[i][0]) correct++;
      }
      console.log(`Epoch ${epoch + 1}: Loss = ${(totalLoss / 4).toFixed(4)}, Correct = ${correct}/4`);
    }
  }

  console.log('\nFinal predictions:');
  for (let i = 0; i < 4; i++) {
    const x = tensor([[xData[i][0], xData[i][1]]]);
    const pred = model.forward(x);
    console.log(`  ${xData[i][0]} XOR ${xData[i][1]} = ${pred.data[0].toFixed(3)} (expected: ${yData[i][0]})`);
  }
}

async function testWithSGD() {
  console.log('\n=== Simple MLP with SGD ===\n');

  const model = new SimpleMLP(2, 32, 1);
  const opt = new SGD(model.parameters(), 0.5); // Higher LR for SGD
  const loss = new MSELoss();

  const xData = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yData = [[0], [1], [1], [0]];

  for (let epoch = 0; epoch < 2000; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < 4; i++) {
      opt.zeroGrad();
      const x = tensor([[xData[i][0], xData[i][1]]]);
      const y = tensor([[yData[i][0]]]);

      const out = model.forward(x);
      const l = loss.forward(out, y);
      l.backward();
      opt.step();

      totalLoss += l.item();
    }

    if ((epoch + 1) % 400 === 0) {
      let correct = 0;
      for (let i = 0; i < 4; i++) {
        const x = tensor([[xData[i][0], xData[i][1]]]);
        const pred = model.forward(x);
        const p = pred.data[0] > 0.5 ? 1 : 0;
        if (p === yData[i][0]) correct++;
      }
      console.log(`Epoch ${epoch + 1}: Loss = ${(totalLoss / 4).toFixed(4)}, Correct = ${correct}/4`);
    }
  }

  console.log('\nFinal predictions:');
  for (let i = 0; i < 4; i++) {
    const x = tensor([[xData[i][0], xData[i][1]]]);
    const pred = model.forward(x);
    console.log(`  ${xData[i][0]} XOR ${xData[i][1]} = ${pred.data[0].toFixed(3)} (expected: ${yData[i][0]})`);
  }
}

async function testBatchTraining() {
  console.log('\n=== Simple MLP - Batch Training ===\n');

  const model = new SimpleMLP(2, 32, 1);
  const opt = new Adam(model.parameters(), 0.1);
  const loss = new MSELoss();

  const xData = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const yData = tensor([[0], [1], [1], [0]]);

  for (let epoch = 0; epoch < 500; epoch++) {
    opt.zeroGrad();
    const out = model.forward(xData);
    const l = loss.forward(out, yData);
    l.backward();
    opt.step();

    if ((epoch + 1) % 100 === 0) {
      const pred = model.forward(xData);
      let correct = 0;
      for (let i = 0; i < 4; i++) {
        const p = pred.data[i] > 0.5 ? 1 : 0;
        if (p === yData.data[i]) correct++;
      }
      console.log(`Epoch ${epoch + 1}: Loss = ${l.item().toFixed(4)}, Correct = ${correct}/4`);
    }
  }

  console.log('\nFinal predictions:');
  const pred = model.forward(xData);
  for (let i = 0; i < 4; i++) {
    console.log(`  ${xData.data[i * 2]} XOR ${xData.data[i * 2 + 1]} = ${pred.data[i].toFixed(3)} (expected: ${yData.data[i]})`);
  }
}

async function main() {
  console.log('=' .repeat(50));
  console.log('MINIMAL XOR DEBUG TEST');
  console.log('='.repeat(50));

  await testSimpleMLP();
  await testWithSGD();
  await testBatchTraining();

  console.log('\n' + '='.repeat(50));
  console.log('Test complete');
  console.log('='.repeat(50));
}

main().catch(console.error);
