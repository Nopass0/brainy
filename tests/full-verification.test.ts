/**
 * @fileoverview Full verification test suite
 * @description Comprehensive tests to verify all library components work correctly
 */

import { describe, test, expect } from 'bun:test';
import {
  // Core
  tensor,
  zeros,
  ones,
  rand,
  randn,
  eye,
  linspace,
  arange,
  full,
  cat,
  stack,
  scalar,
  Tensor,
  DType,

  // NN Layers
  Module,
  Sequential,
  Parameter,
  Linear,
  Conv2d,
  MaxPool2d,
  AvgPool2d,
  BatchNorm1d,
  BatchNorm2d,
  LayerNorm,
  Dropout,
  Embedding,
  Flatten,

  // Activations
  ReLU,
  LeakyReLU,
  GELU,
  SiLU,
  Sigmoid,
  Tanh,
  Softmax,
  LogSoftmax,

  // Loss functions
  MSELoss,
  L1Loss,
  CrossEntropyLoss,
  BCELoss,
  BCEWithLogitsLoss,
  NLLLoss,

  // Optimizers
  SGD,
  Adam,
  AdamW,
  RMSprop,

  // LR Schedulers
  StepLR,
  CosineAnnealingLR,

  // Data
  Dataset,
  TensorDataset,
  DataLoader,

  // Functional
  F,

  // Utils
  saveModel,
  loadModel,
  manualSeed,
} from '../src';

describe('Full Library Verification', () => {

  describe('1. Tensor Creation', () => {
    test('tensor() from nested array', () => {
      const t = tensor([[1, 2, 3], [4, 5, 6]]);
      expect(t.shape).toEqual([2, 3]);
      expect(t.size).toBe(6);
      expect(t.ndim).toBe(2);
      expect(t.data[0]).toBe(1);
      expect(t.data[5]).toBe(6);
    });

    test('zeros() creates zero tensor', () => {
      const t = zeros([3, 4, 5]);
      expect(t.shape).toEqual([3, 4, 5]);
      expect(t.size).toBe(60);
      expect(Array.from(t.data).every(v => v === 0)).toBe(true);
    });

    test('ones() creates ones tensor', () => {
      const t = ones([2, 3]);
      expect(Array.from(t.data).every(v => v === 1)).toBe(true);
    });

    test('rand() creates uniform random', () => {
      const t = rand([1000]);
      const min = Math.min(...t.data);
      const max = Math.max(...t.data);
      expect(min).toBeGreaterThanOrEqual(0);
      expect(max).toBeLessThan(1);
    });

    test('randn() creates normal random', () => {
      const t = randn([10000]);
      const mean = Array.from(t.data).reduce((a, b) => a + b, 0) / t.size;
      expect(Math.abs(mean)).toBeLessThan(0.1);
    });

    test('eye() creates identity matrix', () => {
      const t = eye(4);
      expect(t.shape).toEqual([4, 4]);
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          expect(t.get(i, j)).toBe(i === j ? 1 : 0);
        }
      }
    });

    test('linspace() creates evenly spaced values', () => {
      const t = linspace(0, 10, 11);
      expect(t.shape).toEqual([11]);
      expect(t.data[0]).toBe(0);
      expect(t.data[5]).toBe(5);
      expect(t.data[10]).toBe(10);
    });

    test('arange() creates range', () => {
      const t = arange(0, 5);
      expect(Array.from(t.data)).toEqual([0, 1, 2, 3, 4]);
    });

    test('full() creates filled tensor', () => {
      const t = full([2, 2], 7);
      expect(Array.from(t.data).every(v => v === 7)).toBe(true);
    });
  });

  describe('2. Tensor Operations', () => {
    test('add() element-wise', () => {
      const a = tensor([[1, 2], [3, 4]]);
      const b = tensor([[5, 6], [7, 8]]);
      const c = a.add(b);
      expect(Array.from(c.data)).toEqual([6, 8, 10, 12]);
    });

    test('sub() element-wise', () => {
      const a = tensor([10, 20, 30]);
      const b = tensor([1, 2, 3]);
      const c = a.sub(b);
      expect(Array.from(c.data)).toEqual([9, 18, 27]);
    });

    test('mul() element-wise', () => {
      const a = tensor([1, 2, 3]);
      const c = a.mul(2);
      expect(Array.from(c.data)).toEqual([2, 4, 6]);
    });

    test('div() element-wise', () => {
      const a = tensor([10, 20, 30]);
      const c = a.div(10);
      expect(Array.from(c.data)).toEqual([1, 2, 3]);
    });

    test('matmul() matrix multiplication', () => {
      const a = tensor([[1, 2], [3, 4]]);
      const b = tensor([[5, 6], [7, 8]]);
      const c = a.matmul(b);
      expect(c.shape).toEqual([2, 2]);
      expect(c.get(0, 0)).toBe(19);  // 1*5 + 2*7
      expect(c.get(0, 1)).toBe(22);  // 1*6 + 2*8
      expect(c.get(1, 0)).toBe(43);  // 3*5 + 4*7
      expect(c.get(1, 1)).toBe(50);  // 3*6 + 4*8
    });

    test('pow() power', () => {
      const a = tensor([2, 3, 4]);
      const c = a.pow(2);
      expect(Array.from(c.data)).toEqual([4, 9, 16]);
    });

    test('sqrt() square root', () => {
      const a = tensor([4, 9, 16]);
      const c = a.sqrt();
      expect(Array.from(c.data)).toEqual([2, 3, 4]);
    });

    test('exp() exponential', () => {
      const a = tensor([0, 1]);
      const c = a.exp();
      expect(c.data[0]).toBeCloseTo(1, 5);
      expect(c.data[1]).toBeCloseTo(Math.E, 5);
    });

    test('log() logarithm', () => {
      const a = tensor([1, Math.E, Math.E * Math.E]);
      const c = a.log();
      expect(c.data[0]).toBeCloseTo(0, 5);
      expect(c.data[1]).toBeCloseTo(1, 5);
      expect(c.data[2]).toBeCloseTo(2, 5);
    });

    test('sum() reduction', () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]]);
      expect(a.sum().item()).toBe(21);

      const rowSum = a.sum(1);
      expect(Array.from(rowSum.data)).toEqual([6, 15]);

      const colSum = a.sum(0);
      expect(Array.from(colSum.data)).toEqual([5, 7, 9]);
    });

    test('mean() reduction', () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]]);
      expect(a.mean().item()).toBe(3.5);
    });

    test('max() and min()', () => {
      const a = tensor([[1, 5, 3], [2, 4, 6]]);
      expect(a.max().values.item()).toBe(6);
      expect(a.min().values.item()).toBe(1);
    });

    test('argmax() and argmin()', () => {
      const a = tensor([1, 5, 2, 4, 3]);
      expect(a.argmax().item()).toBe(1);
      expect(a.argmin().item()).toBe(0);
    });

    test('reshape()', () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]]);
      const b = a.reshape(3, 2);
      expect(b.shape).toEqual([3, 2]);
      expect(b.data[0]).toBe(1);
      expect(b.data[5]).toBe(6);
    });

    test('transpose()', () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]]);
      const b = a.T;
      expect(b.shape).toEqual([3, 2]);
      expect(b.get(0, 0)).toBe(1);
      expect(b.get(0, 1)).toBe(4);
    });

    test('squeeze() and unsqueeze()', () => {
      const a = tensor([[[1, 2, 3]]]);
      expect(a.shape).toEqual([1, 1, 3]);

      const b = a.squeeze();
      expect(b.shape).toEqual([3]);

      const c = b.unsqueeze(0);
      expect(c.shape).toEqual([1, 3]);
    });

    test('flatten()', () => {
      const a = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const b = a.flatten();
      expect(b.shape).toEqual([8]);
    });
  });

  describe('3. Broadcasting', () => {
    test('scalar broadcast', () => {
      const a = tensor([[1, 2], [3, 4]]);
      const b = a.add(10);
      expect(Array.from(b.data)).toEqual([11, 12, 13, 14]);
    });

    test('1D to 2D broadcast', () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]]);
      const b = tensor([10, 20, 30]);
      const c = a.add(b);
      expect(Array.from(c.data)).toEqual([11, 22, 33, 14, 25, 36]);
    });

    test('column broadcast', () => {
      const a = tensor([[1, 2], [3, 4]]);
      const b = tensor([[10], [20]]);
      const c = a.add(b);
      expect(Array.from(c.data)).toEqual([11, 12, 23, 24]);
    });
  });

  describe('4. Autograd', () => {
    test('simple gradient x^2', () => {
      const x = tensor([[3]], { requiresGrad: true });
      const y = x.pow(2);  // y = x^2
      y.backward();
      expect(x.grad!.item()).toBeCloseTo(6, 5);  // dy/dx = 2x = 6
    });

    test('chain rule', () => {
      const x = tensor([[2]], { requiresGrad: true });
      const y = x.pow(2).add(x.mul(3)).add(1);  // y = x^2 + 3x + 1
      y.backward();
      expect(x.grad!.item()).toBeCloseTo(7, 5);  // dy/dx = 2x + 3 = 7
    });

    test('matmul gradient', () => {
      const a = tensor([[1, 2], [3, 4]], { requiresGrad: true });
      const b = tensor([[1, 0], [0, 1]]);
      const c = a.matmul(b);
      const loss = c.sum();
      loss.backward();
      expect(a.grad).not.toBeNull();
    });
  });

  describe('5. Neural Network Layers', () => {
    test('Linear layer', () => {
      const linear = new Linear(10, 5);
      const x = randn([3, 10]);
      const y = linear.forward(x);
      expect(y.shape).toEqual([3, 5]);
      expect(linear.numParameters()).toBe(10 * 5 + 5);
    });

    test('Conv2d layer', () => {
      const conv = new Conv2d(3, 16, 3, 1, 1);
      const x = randn([2, 3, 8, 8]);
      const y = conv.forward(x);
      expect(y.shape).toEqual([2, 16, 8, 8]);
    });

    test('MaxPool2d layer', () => {
      const pool = new MaxPool2d(2);
      const x = randn([2, 3, 8, 8]);
      const y = pool.forward(x);
      expect(y.shape).toEqual([2, 3, 4, 4]);
    });

    test('BatchNorm1d layer', () => {
      const bn = new BatchNorm1d(10);
      const x = randn([4, 10]);
      const y = bn.forward(x);
      expect(y.shape).toEqual([4, 10]);
    });

    test('LayerNorm layer', () => {
      const ln = new LayerNorm(16);
      const x = randn([4, 16]);
      const y = ln.forward(x);
      expect(y.shape).toEqual([4, 16]);
    });

    test('Dropout layer', () => {
      const dropout = new Dropout(0.5);
      const x = ones([100]);

      dropout.eval();
      const yEval = dropout.forward(x);
      expect(Array.from(yEval.data).every(v => v === 1)).toBe(true);
    });

    test('Embedding layer', () => {
      const emb = new Embedding(100, 32);
      const idx = tensor([[1, 5, 10]]);
      const y = emb.forward(idx);
      expect(y.shape).toEqual([1, 3, 32]);
    });

    test('Sequential container', () => {
      const model = new Sequential(
        new Linear(10, 20),
        new ReLU(),
        new Linear(20, 5)
      );
      const x = randn([3, 10]);
      const y = model.forward(x);
      expect(y.shape).toEqual([3, 5]);
    });
  });

  describe('6. Activation Functions', () => {
    test('ReLU', () => {
      const relu = new ReLU();
      const x = tensor([[-1, 0, 1, 2]]);
      const y = relu.forward(x);
      expect(Array.from(y.data)).toEqual([0, 0, 1, 2]);
    });

    test('Sigmoid', () => {
      const sigmoid = new Sigmoid();
      const x = tensor([[0]]);
      const y = sigmoid.forward(x);
      expect(y.data[0]).toBeCloseTo(0.5, 5);
    });

    test('Tanh', () => {
      const tanh = new Tanh();
      const x = tensor([[0]]);
      const y = tanh.forward(x);
      expect(y.data[0]).toBeCloseTo(0, 5);
    });

    test('Softmax', () => {
      const softmax = new Softmax(-1);
      const x = tensor([[1, 2, 3]]);
      const y = softmax.forward(x);
      const sum = Array.from(y.data).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);
    });

    test('GELU', () => {
      const gelu = new GELU();
      const x = tensor([[0, 1, -1]]);
      const y = gelu.forward(x);
      expect(y.data[0]).toBeCloseTo(0, 4);
    });
  });

  describe('7. Loss Functions', () => {
    test('MSELoss', () => {
      const loss = new MSELoss();
      const pred = tensor([[1, 2, 3]]);
      const target = tensor([[1, 2, 3]]);
      expect(loss.forward(pred, target).item()).toBe(0);

      const pred2 = tensor([[0, 0]]);
      const target2 = tensor([[1, 1]]);
      expect(loss.forward(pred2, target2).item()).toBe(1);
    });

    test('L1Loss', () => {
      const loss = new L1Loss();
      const pred = tensor([[0, 0]]);
      const target = tensor([[1, 2]]);
      expect(loss.forward(pred, target).item()).toBe(1.5);
    });

    test('BCELoss', () => {
      const loss = new BCELoss();
      const pred = tensor([[0.5]]);
      const target = tensor([[1]]);
      expect(loss.forward(pred, target).item()).toBeGreaterThan(0);
    });
  });

  describe('8. Optimizers', () => {
    test('SGD optimizer', () => {
      const model = new Linear(5, 3);
      const optimizer = new SGD(model.parameters(), 0.1);

      const x = randn([2, 5]);
      const y = model.forward(x);
      const loss = y.pow(2).mean();

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();

      expect(loss.item()).toBeGreaterThanOrEqual(0);
    });

    test('Adam optimizer', () => {
      const model = new Linear(5, 3);
      const optimizer = new Adam(model.parameters(), 0.01);

      const x = randn([2, 5]);
      const y = model.forward(x);
      const loss = y.pow(2).mean();

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();

      expect(loss.item()).toBeGreaterThanOrEqual(0);
    });
  });

  describe('9. Complete Training Workflow', () => {
    test('Linear regression converges', () => {
      // y = 2x + 3
      const X = tensor([[1], [2], [3], [4], [5]]);
      const Y = tensor([[5], [7], [9], [11], [13]]);

      const model = new Linear(1, 1);
      const optimizer = new Adam(model.parameters(), 0.1);
      const criterion = new MSELoss();

      let lastLoss = Infinity;
      for (let i = 0; i < 200; i++) {
        const pred = model.forward(X);
        const loss = criterion.forward(pred, Y);

        optimizer.zeroGrad();
        loss.backward();
        optimizer.step();

        lastLoss = loss.item();
      }

      expect(lastLoss).toBeLessThan(1);

      // Test prediction
      const testX = tensor([[10]]);
      const pred = model.forward(testX);
      // Allow some tolerance since this is approximate
      expect(Math.abs(pred.data[0] - 23)).toBeLessThan(3); // 2*10 + 3 = 23 ± 3
    });

    test('Binary classification', () => {
      const X = tensor([[0.1], [0.2], [0.8], [0.9]]);
      const Y = tensor([[0], [0], [1], [1]]);

      const model = new Sequential(
        new Linear(1, 4),
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

      const pred = model.forward(X);
      expect(pred.data[0]).toBeLessThan(0.5);
      expect(pred.data[3]).toBeGreaterThan(0.5);
    });
  });

  describe('10. Concatenation and Stacking', () => {
    test('cat() along axis 0', () => {
      const a = tensor([[1, 2]]);
      const b = tensor([[3, 4]]);
      const c = cat([a, b], 0);
      expect(c.shape).toEqual([2, 2]);
      expect(Array.from(c.data)).toEqual([1, 2, 3, 4]);
    });

    test('cat() along axis 1', () => {
      const a = tensor([[1], [2]]);
      const b = tensor([[3], [4]]);
      const c = cat([a, b], 1);
      expect(c.shape).toEqual([2, 2]);
    });

    test('stack()', () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = stack([a, b], 0);
      expect(c.shape).toEqual([2, 3]);
    });
  });

  describe('11. Model Serialization', () => {
    test('stateDict and loadStateDict', () => {
      const model1 = new Sequential(
        new Linear(4, 8),
        new ReLU(),
        new Linear(8, 2)
      );

      const state = model1.stateDict();

      const model2 = new Sequential(
        new Linear(4, 8),
        new ReLU(),
        new Linear(8, 2)
      );

      model2.loadStateDict(state);

      const x = randn([1, 4]);
      const y1 = model1.forward(x);
      const y2 = model2.forward(x);

      for (let i = 0; i < y1.size; i++) {
        expect(y1.data[i]).toBeCloseTo(y2.data[i], 5);
      }
    });
  });

  describe('12. Functional API', () => {
    test('F.relu', () => {
      const x = tensor([[-1, 0, 1]]);
      const y = F.relu(x);
      expect(Array.from(y.data)).toEqual([0, 0, 1]);
    });

    test('F.sigmoid', () => {
      const x = tensor([[0]]);
      const y = F.sigmoid(x);
      expect(y.data[0]).toBeCloseTo(0.5, 5);
    });

    test('F.softmax', () => {
      const x = tensor([[1, 2, 3]]);
      const y = F.softmax(x, -1);
      const sum = Array.from(y.data).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);
    });
  });
});
