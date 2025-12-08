/**
 * @fileoverview Tests for model architectures
 */

import { describe, test, expect } from 'bun:test';
import {
  Tensor,
  tensor,
  randn,
  GPT,
  createSmallGPT,
  TRM,
  createTinyTRM,
  TRMClassifier,
  MultimodalFewShot,
  createSmallMultimodal,
} from '../src';

describe('GPT Model', () => {
  test('GPT forward pass shape', () => {
    const gpt = createSmallGPT();
    const inputIds = tensor([[1, 2, 3, 4]]);
    const output = gpt.forward(inputIds);
    expect(output.shape[0]).toBe(1);
    expect(output.shape[1]).toBe(4);
  });

  test('GPT parameters exist', () => {
    const gpt = createSmallGPT();
    const params = gpt.parameters();
    expect(params).toBeDefined();
  });

  test('GPT state dict exists', () => {
    const gpt = createSmallGPT();
    const stateDict = gpt.stateDict();
    expect(stateDict).toBeDefined();
  });
});

describe('TRM Model', () => {
  test('TRM forward pass', () => {
    const trm = createTinyTRM(4, 2, 32, 4);
    const x = randn([1, 4]);
    const output = trm.forward(x);
    expect(output.shape).toEqual([1, 2]);
  });

  test('TRM with different recursion steps', () => {
    const trm = createTinyTRM(4, 2, 32, 4);
    const x = randn([1, 4]);

    const out1 = trm.forward(x, 1);
    const out4 = trm.forward(x, 4);

    // Different number of recursions may give different outputs
    expect(out1.shape).toEqual([1, 2]);
    expect(out4.shape).toEqual([1, 2]);
  });

  test('TRM forwardWithHistory tracks intermediates', () => {
    const trm = createTinyTRM(4, 2, 32, 4);
    const x = randn([1, 4]);
    const { output, intermediates } = trm.forwardWithHistory(x);

    expect(intermediates.length).toBe(5);  // Initial + 4 steps
    expect(output.shape).toEqual([1, 2]);
  });

  test('TRM adaptive returns output', () => {
    const trm = createTinyTRM(4, 2, 32, 4);
    const x = randn([1, 4]);
    const { output, steps } = trm.forwardAdaptive(x, 16, 0.0001);

    expect(steps).toBeLessThanOrEqual(16);
    expect(output.shape).toEqual([1, 2]);
  });

  test('TRMClassifier few-shot prediction', () => {
    const classifier = new TRMClassifier(4, 32, 3, 4);

    const supportX = tensor([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
    ]);
    const supportY = [0, 1, 2];

    const queryX = tensor([
      [0.9, 0.1, 0, 0],
      [0.1, 0.9, 0, 0],
    ]);

    const predictions = classifier.fewShotPredict(supportX, supportY, queryX);
    expect(predictions.length).toBe(2);
  });
});

describe('Multimodal Model', () => {
  test('Multimodal encode image', () => {
    const model = createSmallMultimodal();
    const image = randn([1, 64]);  // 8x8 flattened
    const embedding = model.encode({ image });
    expect(embedding.shape).toEqual([1, 64]);
  });

  test('Multimodal encode sequence', () => {
    const model = createSmallMultimodal();
    const sequence = randn([1, 16]);
    const embedding = model.encode({ sequence });
    expect(embedding.shape).toEqual([1, 64]);
  });

  test('Multimodal encode combined', () => {
    const model = createSmallMultimodal();
    const image = randn([1, 64]);
    const sequence = randn([1, 16]);
    const embedding = model.encode({ image, sequence });
    expect(embedding.shape).toEqual([1, 64]);
  });

  test('Multimodal classify output', () => {
    const model = createSmallMultimodal();
    const input = { sequence: randn([1, 16]) };
    const logits = model.classify(input);
    expect(logits.shape).toEqual([1, 10]);  // 10 classes default
  });

  test('Multimodal regress output', () => {
    const model = createSmallMultimodal();
    const input = { sequence: randn([1, 16]) };
    const value = model.regress(input);
    expect(value.shape).toEqual([1, 1]);
  });
});
