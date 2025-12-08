/**
 * @fileoverview Tests for Neural Network modules
 */

import { describe, test, expect } from 'bun:test';
import {
  Tensor,
  tensor,
  zeros,
  ones,
  randn,
  Sequential,
  Linear,
  Conv2d,
  ReLU,
  Sigmoid,
  Tanh,
  Softmax,
  GELU,
  Dropout,
  BatchNorm1d,
  LayerNorm,
  Flatten,
  Embedding,
  MSELoss,
  CrossEntropyLoss,
  BCELoss,
  Adam,
  SGD,
} from '../src';

describe('Linear Layer', () => {
  test('Linear forward pass shape', () => {
    const linear = new Linear(10, 5);
    const x = randn([3, 10]);
    const y = linear.forward(x);
    expect(y.shape).toEqual([3, 5]);
  });

  test('Linear with bias=false works', () => {
    const linear = new Linear(10, 5, false);
    const x = randn([2, 10]);
    const y = linear.forward(x);
    expect(y.shape).toEqual([2, 5]);
  });

  test('Linear has parameters', () => {
    const linear = new Linear(10, 5);
    const params = linear.parameters();
    expect(params).toBeDefined();
  });
});

describe('Activation Functions', () => {
  test('ReLU zeros negative values', () => {
    const relu = new ReLU();
    const x = tensor([[-1, 0, 1, 2]]);
    const y = relu.forward(x);
    expect(Array.from(y.data)).toEqual([0, 0, 1, 2]);
  });

  test('Sigmoid output range [0, 1]', () => {
    const sigmoid = new Sigmoid();
    const x = tensor([[-100, 0, 100]]);
    const y = sigmoid.forward(x);
    expect(y.data[0]).toBeCloseTo(0, 2);
    expect(y.data[1]).toBeCloseTo(0.5, 5);
    expect(y.data[2]).toBeCloseTo(1, 2);
  });

  test('Tanh output range [-1, 1]', () => {
    const tanh = new Tanh();
    const x = tensor([[-100, 0, 100]]);
    const y = tanh.forward(x);
    expect(y.data[0]).toBeCloseTo(-1, 2);
    expect(y.data[1]).toBeCloseTo(0, 5);
    expect(y.data[2]).toBeCloseTo(1, 2);
  });

  test('Softmax outputs sum to 1', () => {
    const softmax = new Softmax(-1);
    const x = tensor([[1, 2, 3]]);
    const y = softmax.forward(x);
    const sum = y.data.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  test('GELU smooth activation', () => {
    const gelu = new GELU();
    const x = tensor([[0]]);
    const y = gelu.forward(x);
    expect(y.data[0]).toBeCloseTo(0, 5);
  });
});

describe('Sequential', () => {
  test('Sequential forward pass', () => {
    const model = new Sequential(
      new Linear(10, 5),
      new ReLU(),
      new Linear(5, 2)
    );
    const x = randn([3, 10]);
    const y = model.forward(x);
    expect(y.shape).toEqual([3, 2]);
  });

  test('Sequential parameters collection', () => {
    const model = new Sequential(
      new Linear(4, 3),
      new Linear(3, 2)
    );
    const params = model.parameters();
    expect(params).toBeDefined();
  });
});

describe('Dropout', () => {
  test('Dropout in eval mode passes through', () => {
    const dropout = new Dropout(0.5);
    dropout.eval();
    const x = ones([10]);
    const y = dropout.forward(x);
    expect(Array.from(y.data)).toEqual(Array.from(x.data));
  });

  test('Dropout(0) does nothing', () => {
    const dropout = new Dropout(0);
    const x = ones([10]);
    const y = dropout.forward(x);
    expect(Array.from(y.data)).toEqual(Array.from(x.data));
  });
});

describe('BatchNorm1d', () => {
  test('BatchNorm1d output shape', () => {
    const bn = new BatchNorm1d(5);
    const x = randn([3, 5]);
    const y = bn.forward(x);
    expect(y.shape).toEqual([3, 5]);
  });
});

describe('LayerNorm', () => {
  test('LayerNorm normalizes features', () => {
    const ln = new LayerNorm(4);
    const x = tensor([[1, 2, 3, 4]]);
    const y = ln.forward(x);
    const mean = y.data.reduce((a, b) => a + b, 0) / y.data.length;
    expect(mean).toBeCloseTo(0, 4);
  });
});

describe('Embedding', () => {
  test('Embedding lookup', () => {
    const emb = new Embedding(10, 4);
    const indices = tensor([[0, 1, 2]]);
    const y = emb.forward(indices);
    expect(y.shape).toEqual([1, 3, 4]);
  });
});

describe('Flatten', () => {
  test('Flatten reshapes tensor', () => {
    const flatten = new Flatten();
    const x = randn([2, 3, 4]);
    const y = flatten.forward(x);
    expect(y.shape).toEqual([2, 12]);
  });
});

describe('Loss Functions', () => {
  test('MSELoss computes squared error', () => {
    const loss = new MSELoss();
    const pred = tensor([[1, 2, 3]]);
    const target = tensor([[1, 2, 3]]);
    const l = loss.forward(pred, target);
    expect(l.item()).toBe(0);
  });

  test('MSELoss with difference', () => {
    const loss = new MSELoss();
    const pred = tensor([[0, 0]]);
    const target = tensor([[1, 1]]);
    const l = loss.forward(pred, target);
    expect(l.item()).toBe(1);  // Mean of (1^2 + 1^2) / 2 = 1
  });

  test('BCELoss output range', () => {
    const loss = new BCELoss();
    const pred = tensor([[0.5]]);
    const target = tensor([[1]]);
    const l = loss.forward(pred, target);
    expect(l.item()).toBeGreaterThan(0);
  });
});

describe('Optimizers', () => {
  test('SGD step runs without error', () => {
    const model = new Sequential(new Linear(2, 1));
    const opt = new SGD(model.parameters(), 0.1);

    const x = tensor([[1, 1]], { requiresGrad: true });
    const y = model.forward(x);
    const loss = y.pow(2).mean();

    opt.zeroGrad();
    loss.backward();
    opt.step();
    // Should complete without error
    expect(true).toBe(true);
  });

  test('Adam step runs without error', () => {
    const model = new Sequential(new Linear(2, 1));
    const opt = new Adam(model.parameters(), 0.01);

    const x = tensor([[1, 1]], { requiresGrad: true });
    const y = model.forward(x);
    const loss = y.pow(2).mean();

    opt.zeroGrad();
    loss.backward();
    opt.step();
    // Should complete without error
    expect(true).toBe(true);
  });

  test('Adam zeroGrad runs without error', () => {
    const model = new Sequential(new Linear(2, 1));
    const opt = new Adam(model.parameters(), 0.01);

    const x = tensor([[1, 1]], { requiresGrad: true });
    const y = model.forward(x);
    const loss = y.pow(2).mean();

    loss.backward();
    opt.zeroGrad();
    // Should complete without error
    expect(true).toBe(true);
  });
});

describe('Model Training', () => {
  test('Simple model learns XOR', () => {
    const model = new Sequential(
      new Linear(2, 8),
      new ReLU(),
      new Linear(8, 1),
      new Sigmoid()
    );
    const opt = new Adam(model.parameters(), 0.1);
    const loss = new MSELoss();

    // XOR data
    const inputs = [
      tensor([[0, 0]]),
      tensor([[0, 1]]),
      tensor([[1, 0]]),
      tensor([[1, 1]]),
    ];
    const targets = [
      tensor([[0]]),
      tensor([[1]]),
      tensor([[1]]),
      tensor([[0]]),
    ];

    // Train
    for (let epoch = 0; epoch < 100; epoch++) {
      for (let i = 0; i < 4; i++) {
        const pred = model.forward(inputs[i]);
        const l = loss.forward(pred, targets[i]);
        opt.zeroGrad();
        l.backward();
        opt.step();
      }
    }

    // Test
    let correct = 0;
    for (let i = 0; i < 4; i++) {
      const pred = model.forward(inputs[i]);
      const expected = targets[i].data[0];
      if ((pred.data[0] > 0.5 ? 1 : 0) === expected) correct++;
    }

    expect(correct).toBeGreaterThanOrEqual(3);  // At least 3/4 correct
  });
});
