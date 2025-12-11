/**
 * Simple XOR test - verify basic neural network works
 */

import { Tensor, tensor, zeros, Sequential, Linear, ReLU, Sigmoid, Adam } from '../src';

function testSimpleMLP() {
  console.log('=== Simple MLP XOR Test ===\n');

  // Simple 2-layer MLP
  const model = new Sequential(
    new Linear(2, 8),
    new ReLU(),
    new Linear(8, 8),
    new ReLU(),
    new Linear(8, 1),
    new Sigmoid()
  );

  const optimizer = new Adam(model.parameters(), 0.1);

  // XOR data
  const xData = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
  const yData = new Float32Array([0, 1, 1, 0]);

  const x = new Tensor(xData, [4, 2], { requiresGrad: true });
  const y = new Tensor(yData, [4, 1]);

  console.log('Training simple MLP on XOR...\n');

  for (let epoch = 0; epoch < 500; epoch++) {
    const pred = model.forward(x);
    const loss = pred.sub(y).pow(2).mean();

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 100 === 0) {
      const preds = model.forward(x);
      let correct = 0;
      for (let i = 0; i < 4; i++) {
        const p = preds.data[i] > 0.5 ? 1 : 0;
        const t = yData[i];
        if (p === t) correct++;
      }
      console.log(`Epoch ${epoch + 1}: loss=${loss.data[0].toFixed(4)}, acc=${(correct / 4 * 100).toFixed(0)}%`);
      console.log(`  Predictions: [${Array.from(preds.data).map(v => v.toFixed(2)).join(', ')}]`);
      console.log(`  Targets:     [${Array.from(yData).join(', ')}]`);
    }
  }

  // Final test
  console.log('\nFinal predictions:');
  const finalPred = model.forward(x);
  for (let i = 0; i < 4; i++) {
    const a = Math.round(xData[i * 2]);
    const b = Math.round(xData[i * 2 + 1]);
    const pred = finalPred.data[i];
    const expected = yData[i];
    console.log(`  ${a} XOR ${b} = ${pred.toFixed(3)} (expected: ${expected})`);
  }
}

testSimpleMLP();
