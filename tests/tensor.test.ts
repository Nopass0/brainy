/**
 * @fileoverview Tests for Tensor operations
 */

import { describe, test, expect } from 'bun:test';
import {
  Tensor,
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
} from '../src';

describe('Tensor Creation', () => {
  test('tensor() creates tensor from array', () => {
    const t = tensor([[1, 2], [3, 4]]);
    expect(t.shape).toEqual([2, 2]);
    expect(t.data[0]).toBe(1);
    expect(t.data[3]).toBe(4);
  });

  test('zeros() creates zero tensor', () => {
    const t = zeros([2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.data.every(v => v === 0)).toBe(true);
  });

  test('ones() creates ones tensor', () => {
    const t = ones([3, 2]);
    expect(t.shape).toEqual([3, 2]);
    expect(t.data.every(v => v === 1)).toBe(true);
  });

  test('rand() creates random tensor in [0, 1)', () => {
    const t = rand([100]);
    expect(t.shape).toEqual([100]);
    expect(t.data.every(v => v >= 0 && v < 1)).toBe(true);
  });

  test('randn() creates normal distributed tensor', () => {
    const t = randn([1000]);
    const mean = t.data.reduce((a, b) => a + b, 0) / t.data.length;
    expect(Math.abs(mean)).toBeLessThan(0.2); // Should be close to 0
  });

  test('eye() creates identity matrix', () => {
    const t = eye(3);
    expect(t.shape).toEqual([3, 3]);
    expect(t.data[0]).toBe(1); // [0,0]
    expect(t.data[4]).toBe(1); // [1,1]
    expect(t.data[8]).toBe(1); // [2,2]
    expect(t.data[1]).toBe(0); // [0,1]
  });

  test('linspace() creates evenly spaced values', () => {
    const t = linspace(0, 10, 5);
    expect(t.shape).toEqual([5]);
    expect(t.data[0]).toBe(0);
    expect(t.data[4]).toBe(10);
    expect(t.data[2]).toBeCloseTo(5, 5);
  });

  test('arange() creates range of values', () => {
    const t = arange(0, 5, 1);
    expect(t.shape).toEqual([5]);
    expect(Array.from(t.data)).toEqual([0, 1, 2, 3, 4]);
  });

  test('full() creates tensor filled with value', () => {
    const t = full([2, 2], 5);
    expect(t.data.every(v => v === 5)).toBe(true);
  });

  test('scalar() creates tensor with single value', () => {
    const t = scalar(42);
    expect(t.data[0]).toBe(42);
    expect(t.size).toBe(1);
  });
});

describe('Tensor Operations', () => {
  test('add() element-wise addition', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = tensor([[5, 6], [7, 8]]);
    const c = a.add(b);
    expect(Array.from(c.data)).toEqual([6, 8, 10, 12]);
  });

  test('sub() element-wise subtraction', () => {
    const a = tensor([[5, 6], [7, 8]]);
    const b = tensor([[1, 2], [3, 4]]);
    const c = a.sub(b);
    expect(Array.from(c.data)).toEqual([4, 4, 4, 4]);
  });

  test('mul() element-wise multiplication', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = tensor([[2, 2], [2, 2]]);
    const c = a.mul(b);
    expect(Array.from(c.data)).toEqual([2, 4, 6, 8]);
  });

  test('div() element-wise division', () => {
    const a = tensor([[4, 6], [8, 10]]);
    const b = tensor([[2, 2], [2, 2]]);
    const c = a.div(b);
    expect(Array.from(c.data)).toEqual([2, 3, 4, 5]);
  });

  test('pow() element-wise power', () => {
    const a = tensor([[2, 3], [4, 5]]);
    const b = a.pow(2);
    expect(Array.from(b.data)).toEqual([4, 9, 16, 25]);
  });

  test('matmul() matrix multiplication', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = tensor([[5, 6], [7, 8]]);
    const c = a.matmul(b);
    expect(c.shape).toEqual([2, 2]);
    expect(c.data[0]).toBe(19);  // 1*5 + 2*7
    expect(c.data[3]).toBe(50);  // 3*6 + 4*8
  });

  test('transpose() swaps dimensions', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = a.transpose(0, 1);
    expect(b.shape).toEqual([3, 2]);
  });

  test('view returns tensor data', () => {
    const a = tensor([[1, 2], [3, 4]]);
    expect(a.size).toBe(4);
    expect(Array.from(a.data)).toEqual([1, 2, 3, 4]);
  });

  test('sum() reduces tensor', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = a.sum();
    expect(b.item()).toBe(10);
  });

  test('mean() computes average', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = a.mean();
    expect(b.item()).toBe(2.5);
  });

  test('max() finds maximum value', () => {
    const a = tensor([[1, 5], [3, 2]]);
    const maxVal = Math.max(...a.data);
    expect(maxVal).toBe(5);
  });

  test('min() finds minimum value', () => {
    const a = tensor([[1, 5], [3, 2]]);
    const minVal = Math.min(...a.data);
    expect(minVal).toBe(1);
  });

  test('exp() computes exponential', () => {
    const a = tensor([0, 1]);
    const b = a.exp();
    expect(b.data[0]).toBeCloseTo(1, 5);
    expect(b.data[1]).toBeCloseTo(Math.E, 5);
  });

  test('log() computes logarithm', () => {
    const a = tensor([1, Math.E]);
    const b = a.log();
    expect(b.data[0]).toBeCloseTo(0, 5);
    expect(b.data[1]).toBeCloseTo(1, 5);
  });

  test('abs() computes absolute value', () => {
    const a = tensor([-1, 2, -3]);
    const b = a.abs();
    expect(Array.from(b.data)).toEqual([1, 2, 3]);
  });

  test('neg() negates values', () => {
    const a = tensor([1, -2, 3]);
    const b = a.neg();
    expect(Array.from(b.data)).toEqual([-1, 2, -3]);
  });
});

describe('Tensor Concatenation', () => {
  test('cat() concatenates along axis 0', () => {
    const a = tensor([[1, 2]]);
    const b = tensor([[3, 4]]);
    const c = cat([a, b], 0);
    expect(c.shape).toEqual([2, 2]);
  });

  test('cat() concatenates along axis 1', () => {
    const a = tensor([[1], [2]]);
    const b = tensor([[3], [4]]);
    const c = cat([a, b], 1);
    expect(c.shape).toEqual([2, 2]);
  });

  test('stack() stacks tensors', () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const c = stack([a, b], 0);
    expect(c.shape).toEqual([2, 2]);
  });
});

describe('Broadcasting', () => {
  test('scalar broadcast with tensor', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = scalar(2);
    const c = a.add(b);
    expect(Array.from(c.data)).toEqual([3, 4, 5, 6]);
  });

  test('1D broadcast with 2D', () => {
    const a = tensor([[1, 2], [3, 4]]);
    const b = tensor([10, 20]);
    const c = a.add(b);
    expect(Array.from(c.data)).toEqual([11, 22, 13, 24]);
  });
});

describe('Autograd', () => {
  test('backward() computes gradients', () => {
    const x = tensor([[2]], { requiresGrad: true });
    const y = x.mul(x);  // y = x^2
    y.backward();
    expect(x.grad).not.toBeNull();
    expect(x.grad!.data[0]).toBeCloseTo(4, 5);  // dy/dx = 2x = 4
  });

  test('chain rule through multiple ops', () => {
    const x = tensor([[3]], { requiresGrad: true });
    const y = x.mul(x).add(x);  // y = x^2 + x
    y.backward();
    expect(x.grad!.data[0]).toBeCloseTo(7, 5);  // dy/dx = 2x + 1 = 7
  });
});
