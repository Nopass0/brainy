/**
 * TRM-Lite Quick Test
 */

import {
  tensor,
  TRMLite,
  createTinyTRMLite,
  createStandardTRMLite,
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

async function testXOR() {
  console.log('=== TRM-Lite XOR Test ===\n');

  const model = createTinyTRMLite(2, 1, 8, 4, true);
  const optimizer = new Adam(model.parameters(), 0.1);
  const criterion = new MSELoss();

  const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const Y = tensor([[0], [1], [1], [0]]);

  for (let epoch = 0; epoch < 1000; epoch++) {
    const predictions = model.forward(X, 4);
    const loss = criterion.forward(predictions, Y);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 200 === 0) {
      const acc = binaryAcc(predictions, Y);
      console.log(`Epoch ${epoch + 1}: loss = ${loss.item().toFixed(6)}, acc = ${(acc * 100).toFixed(0)}%`);
    }
  }

  const finalPred = model.forward(X, 4);
  console.log('\nPredictions:');
  for (let i = 0; i < 4; i++) {
    const a = X.data[i * 2], b = X.data[i * 2 + 1];
    console.log(`  ${a} XOR ${b} = ${finalPred.data[i].toFixed(3)} (expected: ${Y.data[i]})`);
  }

  return binaryAcc(finalPred, Y);
}

async function testArithmetic() {
  console.log('\n=== TRM-Lite Arithmetic Test (a + b) ===\n');

  const model = createStandardTRMLite(2, 1, 32, 6);
  const optimizer = new Adam(model.parameters(), 0.02);
  const criterion = new MSELoss();

  // Generate data
  const trainX: number[][] = [], trainY: number[][] = [];
  for (let i = 0; i < 500; i++) {
    const a = Math.random() * 2 - 1;
    const b = Math.random() * 2 - 1;
    trainX.push([a, b]);
    trainY.push([a + b]);
  }

  const testX: number[][] = [], testY: number[][] = [];
  for (let i = 0; i < 100; i++) {
    const a = Math.random() * 2 - 1;
    const b = Math.random() * 2 - 1;
    testX.push([a, b]);
    testY.push([a + b]);
  }

  const X = tensor(trainX);
  const Y = tensor(trainY);

  for (let epoch = 0; epoch < 80; epoch++) {
    const predictions = model.forward(X, 6);
    const loss = criterion.forward(predictions, Y);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 20 === 0) {
      const testPred = model.forward(tensor(testX), 6);
      let correct = 0;
      for (let i = 0; i < 100; i++) {
        if (Math.abs(testPred.data[i] - testY[i][0]) < 0.15) correct++;
      }
      console.log(`Epoch ${epoch + 1}: loss = ${loss.item().toFixed(4)}, test acc = ${correct}%`);
    }
  }

  const testPred = model.forward(tensor(testX), 6);
  let correct = 0;
  for (let i = 0; i < 100; i++) {
    if (Math.abs(testPred.data[i] - testY[i][0]) < 0.15) correct++;
  }
  return correct / 100;
}

async function main() {
  console.log('='.repeat(50));
  console.log('TRM-Lite TEST SUITE');
  console.log('='.repeat(50));

  const xorAcc = await testXOR();
  const arithAcc = await testArithmetic();

  console.log('\n' + '='.repeat(50));
  console.log('SUMMARY');
  console.log('='.repeat(50));
  console.log(`XOR:        ${(xorAcc * 100).toFixed(1)}% (target: 100%)`);
  console.log(`Arithmetic: ${(arithAcc * 100).toFixed(1)}% (target: 90%)`);
  console.log('='.repeat(50));
}

main().catch(console.error);
