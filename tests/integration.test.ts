/**
 * @fileoverview Integration tests for complete workflows
 * @description End-to-end tests for common ML tasks
 */

import { describe, test, expect } from 'bun:test';
import {
  tensor,
  zeros,
  ones,
  randn,
  Sequential,
  Linear,
  ReLU,
  Sigmoid,
  Tanh,
  GELU,
  Softmax,
  MSELoss,
  CrossEntropyLoss,
  BCELoss,
  Adam,
  SGD,
  AdamW,
  Conv2d,
  MaxPool2d,
  Flatten,
  BatchNorm1d,
  Dropout,
  Embedding,
  LayerNorm,
} from '../src';

describe('Complete Training Workflows', () => {
  test('XOR problem with MLP', () => {
    // XOR data
    const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const Y = tensor([[0], [1], [1], [0]]);

    // Model
    const model = new Sequential(
      new Linear(2, 16),
      new ReLU(),
      new Linear(16, 8),
      new ReLU(),
      new Linear(8, 1),
      new Sigmoid()
    );

    const optimizer = new Adam(model.parameters(), 0.1);
    const criterion = new MSELoss();

    // Train
    let lastLoss = Infinity;
    for (let epoch = 0; epoch < 200; epoch++) {
      const pred = model.forward(X);
      const loss = criterion.forward(pred, Y);

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();

      lastLoss = loss.item();
    }

    // Test accuracy
    const predictions = model.forward(X);
    let correct = 0;
    for (let i = 0; i < 4; i++) {
      const pred = predictions.data[i] > 0.5 ? 1 : 0;
      const target = Y.data[i];
      if (pred === target) correct++;
    }

    expect(correct).toBeGreaterThanOrEqual(3);
    expect(lastLoss).toBeLessThan(0.5);
  });

  test('Linear regression', () => {
    // Generate linear data: y = 2x + 1 + noise
    const X = tensor([[1], [2], [3], [4], [5]]);
    const Y = tensor([[3], [5], [7], [9], [11]]);

    const model = new Linear(1, 1);
    const optimizer = new SGD(model.parameters(), 0.05);
    const criterion = new MSELoss();

    // Train
    for (let i = 0; i < 100; i++) {
      const pred = model.forward(X);
      const loss = criterion.forward(pred, Y);

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
    }

    // Test
    const predictions = model.forward(tensor([[6]]));
    expect(predictions.data[0]).toBeCloseTo(13, 0); // Expected: 2*6 + 1 = 13
  });

  test('Multi-class classification', () => {
    // 3-class classification
    const X = randn([12, 10]);
    const Y = tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);

    const model = new Sequential(
      new Linear(10, 8),
      new ReLU(),
      new Linear(8, 3)
    );

    const optimizer = new Adam(model.parameters(), 0.05);
    const criterion = new CrossEntropyLoss();

    // Train
    for (let i = 0; i < 50; i++) {
      const logits = model.forward(X);
      const loss = criterion.forward(logits, Y);

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
    }

    // Just verify it runs without error
    const output = model.forward(X);
    expect(output.shape).toEqual([12, 3]);
  });

  test('Binary classification with BCE', () => {
    const X = tensor([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]]);
    const Y = tensor([[0], [1], [0], [1]]);

    const model = new Sequential(
      new Linear(2, 4),
      new ReLU(),
      new Linear(4, 1),
      new Sigmoid()
    );

    const optimizer = new Adam(model.parameters(), 0.1);
    const criterion = new BCELoss();

    for (let i = 0; i < 100; i++) {
      const pred = model.forward(X);
      const loss = criterion.forward(pred, Y);

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
    }

    // Test predictions
    const pred = model.forward(X);
    expect(pred.shape).toEqual([4, 1]);
  });
});

describe('Layer Operations', () => {
  test('Conv2d forward pass', () => {
    const conv = new Conv2d(3, 16, 3, 1, 1);
    const input = randn([2, 3, 8, 8]); // batch=2, channels=3, 8x8

    const output = conv.forward(input);
    expect(output.shape).toEqual([2, 16, 8, 8]);
  });

  test('MaxPool2d reduces spatial dimensions', () => {
    const pool = new MaxPool2d(2, 2);
    const input = randn([2, 3, 8, 8]);

    const output = pool.forward(input);
    expect(output.shape).toEqual([2, 3, 4, 4]);
  });

  test('BatchNorm1d normalizes features', () => {
    const bn = new BatchNorm1d(10);
    const input = randn([4, 10]);

    const output = bn.forward(input);
    expect(output.shape).toEqual([4, 10]);
  });

  test('Dropout in train vs eval mode', () => {
    const dropout = new Dropout(0.5);
    const input = ones([100]);

    // Train mode - should drop some values
    dropout.train();
    const trainOutput = dropout.forward(input);

    // Eval mode - should pass through
    dropout.eval();
    const evalOutput = dropout.forward(input);

    expect(evalOutput.data.every(v => v === 1)).toBe(true);
  });

  test('Embedding lookup', () => {
    const emb = new Embedding(100, 32);
    const indices = tensor([[1, 5, 10], [2, 3, 4]]);

    const output = emb.forward(indices);
    expect(output.shape).toEqual([2, 3, 32]);
  });

  test('LayerNorm normalizes', () => {
    const ln = new LayerNorm(16);
    const input = randn([4, 16]);

    const output = ln.forward(input);
    expect(output.shape).toEqual([4, 16]);
  });

  test('Flatten layer', () => {
    const flatten = new Flatten();
    const input = randn([2, 3, 4, 5]);

    const output = flatten.forward(input);
    expect(output.shape).toEqual([2, 60]); // 3*4*5 = 60
  });
});

describe('Activation Functions', () => {
  const testInput = tensor([[-2, -1, 0, 1, 2]]);

  test('ReLU', () => {
    const relu = new ReLU();
    const output = relu.forward(testInput);
    expect(output.data[0]).toBe(0);
    expect(output.data[2]).toBe(0);
    expect(output.data[4]).toBe(2);
  });

  test('Sigmoid', () => {
    const sigmoid = new Sigmoid();
    const output = sigmoid.forward(testInput);
    expect(output.data[2]).toBeCloseTo(0.5, 5);
  });

  test('Tanh', () => {
    const tanh = new Tanh();
    const output = tanh.forward(testInput);
    expect(output.data[2]).toBeCloseTo(0, 5);
  });

  test('GELU', () => {
    const gelu = new GELU();
    const output = gelu.forward(testInput);
    expect(output.data[2]).toBeCloseTo(0, 5);
  });

  test('Softmax sums to 1', () => {
    const softmax = new Softmax(-1);
    const output = softmax.forward(testInput);
    const sum = Array.from(output.data).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });
});

describe('Optimizers', () => {
  test('SGD with momentum', () => {
    const model = new Linear(5, 3);
    const optimizer = new SGD(model.parameters(), 0.01, 0.9);

    const input = randn([2, 5]);
    const target = randn([2, 3]);
    const criterion = new MSELoss();

    const pred = model.forward(input);
    const loss = criterion.forward(pred, target);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    // Just verify it runs
    expect(loss.item()).toBeGreaterThanOrEqual(0);
  });

  test('AdamW with weight decay', () => {
    const model = new Linear(5, 3);
    const optimizer = new AdamW(model.parameters(), 0.01, [0.9, 0.999], 0.01);

    const input = randn([2, 5]);
    const target = randn([2, 3]);
    const criterion = new MSELoss();

    const pred = model.forward(input);
    const loss = criterion.forward(pred, target);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    expect(loss.item()).toBeGreaterThanOrEqual(0);
  });
});

describe('Model Serialization', () => {
  test('stateDict() and loadStateDict()', () => {
    const model1 = new Sequential(
      new Linear(4, 8),
      new ReLU(),
      new Linear(8, 2)
    );

    // Get state
    const state = model1.stateDict();
    expect(state.size).toBeGreaterThan(0);

    // Create new model and load state
    const model2 = new Sequential(
      new Linear(4, 8),
      new ReLU(),
      new Linear(8, 2)
    );

    model2.loadStateDict(state);

    // Verify weights match
    const input = randn([1, 4]);
    const out1 = model1.forward(input);
    const out2 = model2.forward(input);

    for (let i = 0; i < out1.size; i++) {
      expect(out1.data[i]).toBeCloseTo(out2.data[i], 5);
    }
  });

  test('numParameters() count', () => {
    const model = new Sequential(
      new Linear(10, 5, true),  // 10*5 + 5 = 55
      new Linear(5, 2, true)   // 5*2 + 2 = 12
    );

    expect(model.numParameters()).toBe(55 + 12);
  });
});

describe('Broadcasting and Shape Operations', () => {
  test('scalar broadcast', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const result = a.add(10);
    expect(result.data[0]).toBe(11);
    expect(result.data[3]).toBe(14);
  });

  test('1D to 2D broadcast', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = tensor([10, 20]);
    const result = a.add(b);
    expect(result.data[0]).toBe(11);
    expect(result.data[1]).toBe(22);
  });

  test('reshape preserves data', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = a.reshape(3, 2);
    expect(b.shape).toEqual([3, 2]);
    expect(b.data[0]).toBe(1);
    expect(b.data[5]).toBe(6);
  });

  test('transpose swaps dimensions', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = a.T;
    expect(b.shape).toEqual([3, 2]);
  });

  test('squeeze removes size-1 dims', () => {
    const a = tensor([[[1, 2, 3]]]);
    expect(a.shape).toEqual([1, 1, 3]);
    const b = a.squeeze();
    expect(b.shape).toEqual([3]);
  });

  test('unsqueeze adds dimension', () => {
    const a = tensor([1, 2, 3]);
    const b = a.unsqueeze(0);
    expect(b.shape).toEqual([1, 3]);
  });
});
