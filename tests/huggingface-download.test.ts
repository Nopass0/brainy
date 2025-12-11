/**
 * @fileoverview Test actual model download from Hugging Face
 * @description Tests that verify model weights can be downloaded and parsed
 */

import { describe, test, expect } from 'bun:test';
import { HuggingFaceHub } from '../src';

describe('Hugging Face Model Download', () => {
  const hub = new HuggingFaceHub();

  test('can get model info for Qwen', async () => {
    const info = await hub.getModelInfo('Qwen/Qwen2.5-0.5B');
    expect(info).toBeDefined();
    expect(info.id || info.modelId).toBeDefined();
  });

  test('can list files in model repository', async () => {
    const files = await hub.listFiles('Qwen/Qwen2.5-0.5B');
    expect(Array.isArray(files)).toBe(true);
    expect(files.length).toBeGreaterThan(0);
    expect(files).toContain('config.json');
    expect(files).toContain('model.safetensors');
  });

  test('can download and parse config.json', async () => {
    const config = await hub.downloadConfig('Qwen/Qwen2.5-0.5B');
    expect(config).toBeDefined();
    expect(config.hidden_size).toBe(896);
    expect(config.vocab_size).toBe(151936);
    expect(config.num_hidden_layers).toBe(24);
    expect(config.num_attention_heads).toBe(14);
  });

  test('can download tokenizer files', async () => {
    const tokenizer = await hub.downloadTokenizer('Qwen/Qwen2.5-0.5B');
    // At least one of vocab, config, or merges should be present
    const hasTokenizerData = tokenizer.vocab || tokenizer.config || tokenizer.merges;
    expect(hasTokenizerData).toBeTruthy();
  });

  // Note: Skipping full weights download in tests due to size (494MB)
  // Uncomment to test full download:
  // test('can download model weights', async () => {
  //   const weights = await hub.downloadWeights('Qwen/Qwen2.5-0.5B');
  //   expect(weights.size).toBeGreaterThan(0);
  // }, 300000); // 5 minute timeout
});

describe('Small Model Download Test', () => {
  const hub = new HuggingFaceHub();

  test('can get info for a small model', async () => {
    // Use a smaller model for testing
    const info = await hub.getModelInfo('gpt2');
    expect(info).toBeDefined();
  });

  test('can download gpt2 config', async () => {
    const config = await hub.downloadConfig('gpt2');
    expect(config).toBeDefined();
    expect(config.vocab_size).toBeDefined();
    expect(config.n_embd).toBeDefined();
  });
});
