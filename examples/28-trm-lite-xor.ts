/**
 * Quick TRM-Lite XOR test
 */

import { Tensor, createTinyTRMLite, Adam } from '../src';

function main() {
  console.log('=== TRM-Lite XOR Test ===\n');

  // Small model with sigmoid output, 1 recursion (like MLP)
  const model = createTinyTRMLite(2, 1, 16, 1, true);
  const optimizer = new Adam(model.parameters(), 0.05);

  // XOR data
  const xData = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
  const yData = new Float32Array([0, 1, 1, 0]);

  const x = new Tensor(xData, [4, 2], { requiresGrad: true });
  const y = new Tensor(yData, [4, 1]);

  console.log('Training TRM-Lite on XOR (4 recursions)...\n');

  for (let epoch = 0; epoch < 300; epoch++) {
    const pred = model.forward(x);
    const loss = pred.sub(y).pow(2).mean();

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 50 === 0) {
      const preds = model.forward(x);
      let correct = 0;
      for (let i = 0; i < 4; i++) {
        if ((preds.data[i] > 0.5 ? 1 : 0) === yData[i]) correct++;
      }
      console.log(`Epoch ${epoch + 1}: loss=${loss.data[0].toFixed(4)}, acc=${(correct / 4 * 100).toFixed(0)}%`);
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
    const correct = (pred > 0.5 ? 1 : 0) === expected;
    console.log(`  ${a} XOR ${b} = ${pred.toFixed(3)} (expected: ${expected}) ${correct ? '✓' : '✗'}`);
  }
}

main();
