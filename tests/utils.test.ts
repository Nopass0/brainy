/**
 * @fileoverview Tests for utility functions
 */

import { describe, test, expect } from 'bun:test';
import {
  manualSeed,
  getRng,
  random,
  randint,
  shuffle,
  choice,
  computeSize,
  computeStrides,
  broadcastShapes,
  shapesEqual,
  inferShape,
  flattenArray,
  Sequential,
  Linear,
  ReLU,
  summary,
  visualize,
  OnlineLearner,
  ContinualLearner,
  MetaLearner,
  Tensor,
  tensor,
  randn,
} from '../src';

describe('Random Utils', () => {
  test('manualSeed makes random reproducible', () => {
    manualSeed(42);
    const a = random();
    manualSeed(42);
    const b = random();
    expect(a).toBe(b);
  });

  test('random() returns value in [0, 1)', () => {
    for (let i = 0; i < 100; i++) {
      const r = random();
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThan(1);
    }
  });

  test('randint returns integer in range', () => {
    for (let i = 0; i < 100; i++) {
      const r = randint(0, 10);
      expect(Number.isInteger(r)).toBe(true);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(10);  // May be inclusive upper bound
    }
  });

  test('shuffle permutes array', () => {
    const arr = [1, 2, 3, 4, 5];
    const original = [...arr];
    shuffle(arr);

    // Same elements
    expect(arr.sort()).toEqual(original.sort());
  });

  test('choice selects from array', () => {
    const arr = [1, 2, 3];
    for (let i = 0; i < 100; i++) {
      const c = choice(arr);
      expect(arr).toContain(c);
    }
  });
});

describe('Shape Utils', () => {
  test('computeSize calculates total elements', () => {
    expect(computeSize([2, 3, 4])).toBe(24);
    expect(computeSize([10])).toBe(10);
    expect(computeSize([])).toBe(1);
  });

  test('computeStrides returns correct strides', () => {
    const strides = computeStrides([2, 3, 4]);
    expect(strides).toEqual([12, 4, 1]);
  });

  test('broadcastShapes handles same shapes', () => {
    const result = broadcastShapes([2, 3], [2, 3]);
    expect(result).toEqual([2, 3]);
  });

  test('broadcastShapes handles different dims', () => {
    const result = broadcastShapes([1, 3], [2, 1]);
    expect(result).toEqual([2, 3]);
  });

  test('broadcastShapes handles scalar', () => {
    const result = broadcastShapes([2, 3], []);
    expect(result).toEqual([2, 3]);
  });

  test('shapesEqual compares correctly', () => {
    expect(shapesEqual([2, 3], [2, 3])).toBe(true);
    expect(shapesEqual([2, 3], [3, 2])).toBe(false);
    expect(shapesEqual([1], [1, 1])).toBe(false);
  });

  test('inferShape from nested array', () => {
    expect(inferShape([[1, 2], [3, 4]])).toEqual([2, 2]);
    expect(inferShape([1, 2, 3])).toEqual([3]);
    expect(inferShape(5)).toEqual([]);
  });

  test('flattenArray flattens nested', () => {
    const result = flattenArray([[1, 2], [3, 4]]);
    expect(result).toEqual([1, 2, 3, 4]);
  });
});

describe('Model Visualization', () => {
  test('summary returns string', () => {
    const model = new Sequential(
      new Linear(10, 5),
      new ReLU(),
      new Linear(5, 2)
    );
    const s = summary(model, [1, 10]);
    expect(typeof s).toBe('string');
    expect(s).toContain('Linear');
    expect(s).toContain('ReLU');
  });

  test('visualize returns string', () => {
    const model = new Sequential(
      new Linear(4, 2),
      new ReLU()
    );
    const v = visualize(model);
    expect(typeof v).toBe('string');
  });
});

describe('Online Learning', () => {
  test('OnlineLearner predict returns tensor', () => {
    const model = new Sequential(
      new Linear(4, 2)
    );
    const learner = new OnlineLearner(model);
    const x = randn([1, 4]);
    const pred = learner.predict(x);
    expect(pred.shape).toEqual([1, 2]);
  });

  test('OnlineLearner addExample updates buffer', () => {
    const model = new Sequential(new Linear(2, 1));
    const learner = new OnlineLearner(model, { bufferSize: 10 });

    for (let i = 0; i < 5; i++) {
      learner.addExample(
        randn([1, 2], { requiresGrad: true }),
        randn([1, 1])
      );
    }

    const stats = learner.getStats();
    expect(stats.stepCount).toBe(5);
    expect(stats.bufferSize).toBe(5);
  });

  test('OnlineLearner reset clears state', () => {
    const model = new Sequential(new Linear(2, 1));
    const learner = new OnlineLearner(model);

    learner.addExample(randn([1, 2], { requiresGrad: true }), randn([1, 1]));
    learner.reset();

    const stats = learner.getStats();
    expect(stats.stepCount).toBe(0);
    expect(stats.bufferSize).toBe(0);
  });

  test('OnlineLearner getModel returns model', () => {
    const model = new Sequential(new Linear(2, 1));
    const learner = new OnlineLearner(model);
    expect(learner.getModel()).toBe(model);
  });
});

describe('Continual Learning', () => {
  test('ContinualLearner trainOnTask returns losses', () => {
    const model = new Sequential(
      new Linear(4, 2),
      new ReLU(),
      new Linear(2, 1)
    );
    const learner = new ContinualLearner(model, 0.01, 10, 0.3);

    const data = [
      { x: randn([1, 4], { requiresGrad: true }), y: randn([1, 1]) },
      { x: randn([1, 4], { requiresGrad: true }), y: randn([1, 1]) },
    ];

    const losses = learner.trainOnTask('task1', data, 2);
    expect(losses.length).toBe(2);
  });

  test('ContinualLearner evaluate returns task scores', () => {
    const model = new Sequential(new Linear(2, 1));
    const learner = new ContinualLearner(model);

    const tasks = new Map([
      ['task1', [{ x: randn([1, 2]), y: randn([1, 1]) }]],
    ]);

    const results = learner.evaluate(tasks);
    expect(results.has('task1')).toBe(true);
  });
});

describe('Meta Learning', () => {
  test('MetaLearner adapt runs without error', () => {
    const model = new Sequential(new Linear(2, 1));
    const learner = new MetaLearner(model, 0.001, 0.1, 5);

    const supportX = randn([3, 2], { requiresGrad: true });
    const supportY = randn([3, 1]);

    learner.adapt(supportX, supportY);
    expect(true).toBe(true);  // Should complete without error
  });

  test('MetaLearner predictAfterAdapt returns tensor', () => {
    const model = new Sequential(new Linear(2, 1));
    const learner = new MetaLearner(model);

    const supportX = randn([2, 2], { requiresGrad: true });
    const supportY = randn([2, 1]);
    const queryX = randn([1, 2]);

    const pred = learner.predictAfterAdapt(supportX, supportY, queryX);
    expect(pred.shape).toEqual([1, 1]);
  });
});
