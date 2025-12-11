/**
 * @fileoverview Пример загрузки и использования модели Qwen 3 с GPU/Hybrid
 * @description Демонстрация интеграции с Hugging Face Hub для загрузки Qwen3-0.5B
 * с поддержкой GPU ускорения и гибридного режима
 *
 * Example of loading and using Qwen 3 model with GPU/Hybrid
 * Demonstrates Hugging Face Hub integration for Qwen3-0.5B download
 * with GPU acceleration and hybrid mode support
 */

import {
  HuggingFaceHub,
  Tensor,
  tensor,
  zeros,
  Linear,
  LayerNorm,
  Embedding,
  Module,
  Sequential,
  GELU,
  Softmax,
  // GPU/Hybrid imports
  isWebGPUSupported,
  getHybridEngine,
  disposeHybridEngine,
  DeviceManager,
  createDevice,
  GPUBackend,
  isGPUBackendAvailable,
} from '../src';

/**
 * Конфигурация модели Qwen3
 * Qwen3 model configuration
 */
interface Qwen3Config {
  vocabSize: number;
  hiddenSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  intermediateSize: number;
  maxPositionEmbeddings: number;
  rmsNormEps: number;
}

/**
 * RMSNorm - нормализация используемая в Qwen3
 * RMSNorm - normalization used in Qwen3
 */
class RMSNorm extends Module {
  weight: Tensor;
  eps: number;

  constructor(hiddenSize: number, eps: number = 1e-6) {
    super();
    this.eps = eps;
    this.weight = zeros([hiddenSize], { requiresGrad: true });
    // Initialize to ones
    for (let i = 0; i < hiddenSize; i++) {
      (this.weight.data as Float32Array)[i] = 1.0;
    }
  }

  forward(input: Tensor): Tensor {
    // RMS = sqrt(mean(x^2) + eps)
    const squared = input.mul(input);
    const meanSquared = squared.mean(-1, true);
    const rms = meanSquared.add(this.eps).sqrt();
    const normalized = input.div(rms);
    return normalized.mul(this.weight);
  }
}

/**
 * Простая реализация Qwen3 для демонстрации с поддержкой GPU
 * Simple Qwen3 implementation for demonstration with GPU support
 */
class Qwen3Demo extends Module {
  config: Qwen3Config;
  embedTokens: Embedding;
  norm: RMSNorm;
  lmHead: Linear;
  private useGPU: boolean = false;

  constructor(config: Qwen3Config) {
    super();
    this.config = config;

    // Token embeddings
    this.embedTokens = new Embedding(config.vocabSize, config.hiddenSize);
    this.registerModule('embed_tokens', this.embedTokens);

    // Final layer norm
    this.norm = new RMSNorm(config.hiddenSize, config.rmsNormEps);
    this.registerModule('norm', this.norm);

    // LM head (ties with embeddings in full implementation)
    this.lmHead = new Linear(config.hiddenSize, config.vocabSize, false);
    this.registerModule('lm_head', this.lmHead);
  }

  /**
   * Включить GPU режим
   * Enable GPU mode
   */
  setGPU(enabled: boolean): void {
    this.useGPU = enabled;
  }

  forward(inputIds: Tensor): Tensor {
    // Get embeddings
    let hidden = this.embedTokens.forward(inputIds);

    // Apply normalization
    hidden = this.norm.forward(hidden);

    // Project to vocabulary
    const logits = this.lmHead.forward(hidden);

    return logits;
  }

  /**
   * Генерация текста
   * Text generation
   */
  generate(inputIds: number[], maxNewTokens: number = 10): number[] {
    const result = [...inputIds];

    for (let i = 0; i < maxNewTokens; i++) {
      const inputTensor = tensor([result]);
      const logits = this.forward(inputTensor);

      // Get last token logits
      const lastLogits = logits.getRow(0).getRow(result.length - 1);

      // Greedy decoding - select argmax
      let maxIdx = 0;
      let maxVal = lastLogits.data[0];
      for (let j = 1; j < this.config.vocabSize; j++) {
        if (lastLogits.data[j] > maxVal) {
          maxVal = lastLogits.data[j];
          maxIdx = j;
        }
      }

      result.push(maxIdx);
    }

    return result.slice(inputIds.length);
  }
}

/**
 * Инициализация GPU/Hybrid режима
 * Initialize GPU/Hybrid mode
 */
async function initializeCompute(): Promise<{
  mode: 'gpu' | 'hybrid' | 'cpu';
  gpuInfo?: string;
}> {
  console.log('\n--- Инициализация вычислений / Initializing compute ---');

  // Check WebGPU support
  const gpuSupported = await isWebGPUSupported();
  console.log(`WebGPU поддержка / WebGPU support: ${gpuSupported ? 'Да/Yes' : 'Нет/No'}`);

  if (gpuSupported) {
    try {
      // Try to create GPU device
      const device = await createDevice('gpu');
      console.log('GPU устройство создано / GPU device created');

      // Initialize hybrid engine
      const hybridEngine = getHybridEngine({
        gpuThreshold: 512,    // Use GPU for tensors > 512 elements
        gpuPriority: 0.8,     // 80% preference for GPU
        autoBalance: true,    // Auto-balance based on performance
        profiling: true,      // Enable profiling
      });

      await hybridEngine.initialize();
      console.log('Гибридный движок инициализирован / Hybrid engine initialized');

      // Get GPU info if available
      let gpuInfo = 'GPU активен / GPU active';
      if (isGPUBackendAvailable()) {
        gpuInfo = 'GPU Backend доступен / GPU Backend available';
      }

      return { mode: 'hybrid', gpuInfo };
    } catch (error) {
      console.warn('GPU инициализация не удалась, используем CPU');
      console.warn('GPU initialization failed, using CPU');
      console.warn(error);
    }
  }

  // Fallback to CPU
  console.log('Используем CPU режим / Using CPU mode');
  return { mode: 'cpu' };
}

/**
 * Основная функция демонстрации
 * Main demonstration function
 */
async function main() {
  console.log('='.repeat(60));
  console.log('Qwen3 Model Loading Example / Пример загрузки модели Qwen3');
  console.log('С поддержкой GPU/Hybrid / With GPU/Hybrid support');
  console.log('='.repeat(60));

  // Initialize compute mode (GPU/Hybrid/CPU)
  const computeInfo = await initializeCompute();
  console.log(`\nРежим вычислений / Compute mode: ${computeInfo.mode.toUpperCase()}`);
  if (computeInfo.gpuInfo) {
    console.log(`GPU Info: ${computeInfo.gpuInfo}`);
  }

  const hub = new HuggingFaceHub();

  // Qwen3-0.5B - самая маленькая версия Qwen3
  // Qwen3-0.5B - smallest version of Qwen3
  const modelId = 'Qwen/Qwen2.5-0.5B';

  console.log(`\nЗагрузка информации о модели / Loading model info: ${modelId}`);

  try {
    // Get model info
    const modelInfo = await hub.getModelInfo(modelId);
    console.log('\nИнформация о модели / Model info:');
    console.log(`  - ID: ${modelInfo.id || modelId}`);
    console.log(`  - Pipeline: ${modelInfo.pipeline_tag || 'text-generation'}`);
    console.log(`  - Downloads: ${modelInfo.downloads || 'N/A'}`);

    // List files
    const files = await hub.listFiles(modelId);
    console.log('\nФайлы модели / Model files:');
    files.slice(0, 10).forEach(f => console.log(`  - ${f}`));
    if (files.length > 10) console.log(`  ... and ${files.length - 10} more`);

    // Download config
    console.log('\nЗагрузка конфигурации / Downloading config...');
    const config = await hub.downloadConfig(modelId);

    if (config) {
      console.log('\nКонфигурация модели / Model configuration:');
      console.log(`  - Hidden size: ${config.hidden_size}`);
      console.log(`  - Vocab size: ${config.vocab_size}`);
      console.log(`  - Num layers: ${config.num_hidden_layers}`);
      console.log(`  - Num heads: ${config.num_attention_heads}`);
      console.log(`  - Intermediate size: ${config.intermediate_size}`);
    }

    // Create demo model with small config for testing
    console.log('\nСоздание демо-модели / Creating demo model...');
    const demoConfig: Qwen3Config = {
      vocabSize: 1000, // Small vocab for demo
      hiddenSize: 64,
      numHiddenLayers: 2,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      intermediateSize: 256,
      maxPositionEmbeddings: 128,
      rmsNormEps: 1e-6,
    };

    const model = new Qwen3Demo(demoConfig);
    model.setGPU(computeInfo.mode !== 'cpu');
    console.log(`  - Параметры / Parameters: ${model.numParameters()}`);
    console.log(`  - Режим GPU / GPU mode: ${computeInfo.mode !== 'cpu' ? 'Да/Yes' : 'Нет/No'}`);

    // Test forward pass with timing
    console.log('\nТестирование forward pass / Testing forward pass...');
    const testInput = tensor([[1, 2, 3, 4, 5]]);

    // Warm-up run
    model.forward(testInput);

    // Timed run
    const startTime = performance.now();
    const iterations = 10;
    for (let i = 0; i < iterations; i++) {
      model.forward(testInput);
    }
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;

    const output = model.forward(testInput);
    console.log(`  - Input shape: [${testInput.shape}]`);
    console.log(`  - Output shape: [${output.shape}]`);
    console.log(`  - Среднее время / Avg time: ${avgTime.toFixed(2)}ms`);

    // Test generation with timing
    console.log('\nТестирование генерации / Testing generation...');
    const genStart = performance.now();
    const generated = model.generate([1, 2, 3], 5);
    const genEnd = performance.now();

    console.log(`  - Input tokens: [1, 2, 3]`);
    console.log(`  - Generated tokens: [${generated}]`);
    console.log(`  - Время генерации / Generation time: ${(genEnd - genStart).toFixed(2)}ms`);

    // Performance comparison info
    console.log('\n--- Информация о производительности / Performance Info ---');
    console.log(`  - Compute mode: ${computeInfo.mode}`);
    if (computeInfo.mode === 'hybrid' || computeInfo.mode === 'gpu') {
      console.log('  - Тензоры > 512 элементов обрабатываются на GPU');
      console.log('  - Tensors > 512 elements processed on GPU');
    }

    console.log('\n' + '='.repeat(60));
    console.log('Демонстрация завершена! / Demo complete!');
    console.log('='.repeat(60));

    // Instructions for full model loading
    console.log('\nДля загрузки полных весов используйте:');
    console.log('For loading full weights use:');
    console.log(`
const weights = await hub.downloadWeights('${modelId}');
console.log('Loaded weights:', weights.size);
`);

    // GPU setup instructions
    console.log('Для включения GPU установите:');
    console.log('To enable GPU install:');
    console.log(`
# Для Bun / For Bun:
bun add bun-webgpu

# Для Node.js / For Node.js:
npm install webgpu
`);

  } catch (error) {
    console.error('Ошибка / Error:', error);

    console.log('\nПримечание / Note:');
    console.log('Для загрузки моделей Qwen может потребоваться HF_TOKEN');
    console.log('Qwen model download may require HF_TOKEN');
    console.log('Set: export HF_TOKEN=your_token_here');
  } finally {
    // Cleanup
    try {
      disposeHybridEngine();
    } catch (e) {
      // Ignore cleanup errors
    }
  }
}

main().catch(console.error);
