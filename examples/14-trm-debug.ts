/**
 * Debug TRM v2 - найдём и исправим проблему
 */

import { tensor, TRMv2, createTinyTRMv2, Adam, MSELoss } from '../src';
import { TRM, createTinyTRM } from '../src';

async function debugXOR() {
  console.log('=== DEBUG: XOR with TRM v1 vs TRM v2 ===\n');

  // XOR data
  const xData = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yData = [[0], [1], [1], [0]];

  // Test TRM v1 (original)
  console.log('Testing TRM v1 (original):');
  const modelV1 = createTinyTRM(2, 1, 32, 4);
  const optimizerV1 = new Adam(modelV1.parameters(), 0.1);
  const loss = new MSELoss();

  for (let epoch = 0; epoch < 300; epoch++) {
    for (let i = 0; i < 4; i++) {
      optimizerV1.zeroGrad();
      const out = modelV1.forward(tensor([[xData[i][0], xData[i][1]]]));
      const l = loss.forward(out, tensor([[yData[i][0]]]));
      l.backward();
      optimizerV1.step();
    }

    if ((epoch + 1) % 100 === 0) {
      const pred = modelV1.forward(tensor(xData));
      let correct = 0;
      for (let i = 0; i < 4; i++) {
        if ((pred.data[i] > 0.5 ? 1 : 0) === yData[i][0]) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Accuracy = ${(correct / 4 * 100).toFixed(0)}%`);
    }
  }

  // Final TRM v1
  const predV1 = modelV1.forward(tensor(xData));
  console.log('  TRM v1 Predictions:');
  for (let i = 0; i < 4; i++) {
    console.log(`    ${xData[i][0]} XOR ${xData[i][1]} = ${predV1.data[i].toFixed(3)} (expected: ${yData[i][0]})`);
  }

  // Test TRM v2
  console.log('\nTesting TRM v2 (new):');
  const modelV2 = createTinyTRMv2(2, 1, 32, 4);
  const optimizerV2 = new Adam(modelV2.parameters(), 0.1);

  for (let epoch = 0; epoch < 300; epoch++) {
    for (let i = 0; i < 4; i++) {
      optimizerV2.zeroGrad();
      const out = modelV2.forward(tensor([[xData[i][0], xData[i][1]]]), 4);
      const l = loss.forward(out, tensor([[yData[i][0]]]));
      l.backward();
      optimizerV2.step();
    }

    if ((epoch + 1) % 100 === 0) {
      const pred = modelV2.forward(tensor(xData), 4);
      let correct = 0;
      for (let i = 0; i < 4; i++) {
        if ((pred.data[i] > 0.5 ? 1 : 0) === yData[i][0]) correct++;
      }
      console.log(`  Epoch ${epoch + 1}: Accuracy = ${(correct / 4 * 100).toFixed(0)}%`);
    }
  }

  // Final TRM v2
  const predV2 = modelV2.forward(tensor(xData), 4);
  console.log('  TRM v2 Predictions:');
  for (let i = 0; i < 4; i++) {
    console.log(`    ${xData[i][0]} XOR ${xData[i][1]} = ${predV2.data[i].toFixed(3)} (expected: ${yData[i][0]})`);
  }
}

debugXOR().catch(console.error);
