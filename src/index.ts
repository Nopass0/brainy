/**
 * @fileoverview Brainy - Быстрый AI/ML фреймворк для Bun
 * @description Главный экспорт всех модулей фреймворка
 * @author Brainy Team
 * @license MIT
 */

// ============================================
// CORE - Тензоры и Autograd
// ============================================
export {
  Tensor,
  // Фабричные функции
  tensor,
  scalar,
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
  sumToShape,
  // Autograd
  noGrad,
  noGradContext,
  isNoGradEnabled,
} from './core/tensor';
export type { TensorOptions } from './core/tensor';

export { DType } from './core/dtype';
export type { Shape, TypedArray, NestedArray } from './core/dtype';

export { GradNode, GradContext, backward, topologicalSort } from './core/autograd';
export type { GradientFunction } from './core/autograd';

export {
  computeSize,
  computeStrides,
  broadcastShapes,
  shapesEqual,
  inferShape,
  flattenArray,
} from './core/shape';

// ============================================
// NN - Нейронные сети
// ============================================
export {
  Module,
  Sequential,
  Parameter,
  Param,
} from './nn/module';

export {
  // Слои
  Linear,
  Conv2d,
  MaxPool2d,
  AvgPool2d,
  Dropout,
  BatchNorm1d,
  BatchNorm2d,
  Flatten,
  Embedding,
  LayerNorm,
} from './nn/layers';

export {
  // Активации
  ReLU,
  LeakyReLU,
  ELU,
  SELU,
  Sigmoid,
  Tanh,
  Softmax,
  LogSoftmax,
  GELU,
  SiLU,
  Swish,
  PReLU,
  Hardswish,
  Mish,
} from './nn/activations';

export {
  // Функции потерь
  MSELoss,
  L1Loss,
  SmoothL1Loss,
  BCELoss,
  BCEWithLogitsLoss,
  CrossEntropyLoss,
  NLLLoss,
  HingeLoss,
  CosineEmbeddingLoss,
  KLDivLoss,
} from './nn/loss';

export {
  // Инициализация
  xavierUniform,
  xavierNormal,
  kaimingUniform,
  kaimingNormal,
  uniform,
  normal,
  constant,
  zeros_,
  ones_,
  eye_,
  orthogonal,
  calculateFanInFanOut,
  calculateGain,
} from './nn/init';

// ============================================
// NN - Transformer
// ============================================
export {
  ScaledDotProductAttention,
  MultiHeadAttention,
  FeedForward,
  TransformerEncoderBlock,
  TransformerDecoderBlock,
  PositionalEncoding,
  LearnedPositionalEmbedding,
  TransformerEncoder,
  TransformerDecoder,
  RotaryEmbedding,
  createCausalMask,
  createPaddingMask,
} from './nn/transformer';
export type { TransformerConfig } from './nn/transformer';

// ============================================
// OPTIM - Оптимизаторы
// ============================================
export {
  Optimizer,
  SGD,
  Adam,
  AdamW,
  RMSprop,
  Adagrad,
  // LR Schedulers
  LRScheduler,
  StepLR,
  ExponentialLR,
  CosineAnnealingLR,
  ReduceLROnPlateau,
} from './optim/optimizer';

// ============================================
// DATA - Работа с данными
// ============================================
export {
  Dataset,
  TensorDataset,
  ArrayDataset,
  DataLoader,
  trainTestSplit,
  ConcatDataset,
} from './data/dataloader';
export type { Batch } from './data/dataloader';

// ============================================
// FUNCTIONAL - Функциональный API
// ============================================
import * as F from './functional/functional';
export { F };

export {
  relu,
  leakyRelu,
  sigmoid,
  tanh,
  softmax,
  logSoftmax,
  gelu,
  silu,
  swish,
  linear,
  dropout,
  mseLoss,
  bceLoss,
  bceWithLogitsLoss,
  crossEntropyLoss,
  nllLoss,
  l1Loss,
} from './functional/functional';

// ============================================
// COMPUTE - GPU и многопоточность
// ============================================
export {
  DeviceType,
  DeviceManager,
  createDevice,
  getDevice,
  isWebGPUSupported,
  getCPUCores,
} from './compute/device';
export type { DeviceConfig, GPUInfo, CPUInfo, PerformanceStats } from './compute/device';

export { GPUBackend, createGPUBackend, isGPUBackendAvailable } from './compute/gpu';

export { WorkerPool, WorkerOp, getWorkerPool, terminateWorkerPool } from './compute/cpu-workers';

export {
  HybridEngine,
  getHybridEngine,
  disposeHybridEngine,
  asyncOps,
} from './compute/hybrid';
export type { HybridConfig } from './compute/hybrid';

// ============================================
// TEXT - Токенизаторы
// ============================================
export {
  Tokenizer,
  BPETokenizer,
  WordPieceTokenizer,
  CharTokenizer,
  SpecialTokens,
  loadTokenizer,
} from './text/tokenizer';
export type { TokenizerConfig, TokenizerOutput, BatchEncoding } from './text/tokenizer';

// ============================================
// MODELS - Готовые модели
// ============================================
export {
  GPT,
  createSmallGPT,
  createMediumGPT,
  createLargeGPT,
} from './models/gpt';
export type { GPTConfig, GenerationConfig } from './models/gpt';

export {
  VAE,
  ConvVAE,
  createMNISTVAE,
  createCIFARVAE,
} from './models/vae';
export type { VAEConfig, VAEOutput } from './models/vae';

export {
  TRM,
  TRMSeq2Seq,
  TRMClassifier,
  createTinyTRM,
  createReasoningTRM,
  createEnhancedTRM,
  createPonderingTRM,
  createMathTRM,
  createSequenceTRM,
  initTRMGPU,
  isTRMGPUAvailable,
  getTRMGPUBackend,
} from './models/trm';
export type { TRMConfig } from './models/trm';

// TRM v2 - Advanced version with attention, memory, MoE
export {
  TRMv2,
  TRMv2Classifier,
  TRMv2Seq2Seq,
  createTinyTRMv2,
  createReasoningTRMv2,
  createTextTRMv2,
} from './models/trm-v2';
export type { TRMv2Config } from './models/trm-v2';

// TRM Ultra - Stable high-performance version
export {
  TRMUltra,
  TRMUltraClassifier,
  createTinyTRMUltra,
  createReasoningTRMUltra,
} from './models/trm-ultra';
export type { TRMUltraConfig } from './models/trm-ultra';

// TRM Final - Ultimate optimized version
export {
  TRMFinal,
  TRMFinalClassifier,
  createTinyTRMFinal,
  createStandardTRMFinal,
  createReasoningTRMFinal,
} from './models/trm-final';
export type { TRMFinalConfig } from './models/trm-final';

// TRM-X - Extreme Performance TRM
export {
  TRMX,
  TRMXClassifier,
  createTinyTRMX,
  createStandardTRMX,
  createReasoningTRMX,
} from './models/trm-x';
export type { TRMXConfig } from './models/trm-x';

// TRM-Lite - Simple and stable TRM
export {
  TRMLite,
  createTinyTRMLite,
  createStandardTRMLite,
} from './models/trm-lite';
export type { TRMLiteConfig } from './models/trm-lite';

// TRM-Pro - Production-ready TRM with 98%+ accuracy
export {
  TRMPro,
  TRMProClassifier,
  createTinyTRMPro,
  createStandardTRMPro,
  createReasoningTRMPro,
  createBinaryTRMPro,
} from './models/trm-pro';
export type { TRMProConfig } from './models/trm-pro';

// TRM-Supreme - Ultimate TRM with residual scaling
export {
  TRMSupreme,
  createTinyTRMSupreme,
  createStandardTRMSupreme,
  createReasoningTRMSupreme,
  createBinaryTRMSupreme,
} from './models/trm-supreme';
export type { TRMSupremeConfig } from './models/trm-supreme';

export {
  MultimodalFewShot,
  createSmallMultimodal,
  createMediumMultimodal,
} from './models/multimodal';
export type { MultimodalConfig } from './models/multimodal';

// ============================================
// UTILS - Утилиты
// ============================================
export {
  saveModel,
  loadModel,
  saveStateDict,
  loadStateDict,
  serializeStateDict,
  deserializeStateDict,
  exportForInference,
} from './utils/serialize';

export {
  manualSeed,
  getRng,
  resetRng,
  random,
  randint,
  randn as randomRandn,
  choice,
  shuffle,
} from './utils/random';

// Чекпоинты и сжатие
export {
  CheckpointManager,
  CompressionFormat,
  saveCheckpoint,
  loadCheckpoint,
  exportModel,
} from './utils/checkpoint';
export type { CheckpointMetadata, Checkpoint, SaveOptions } from './utils/checkpoint';

// Квантизация
export {
  Quantizer,
  QuantizedTensor,
  QuantizedLinear,
  QuantizedModule,
  QuantizationMode,
  QuantizationBits,
  FakeQuantize,
  dynamicQuantize,
  prepareQAT,
  convertQAT,
  getModelSize,
} from './utils/quantization';
export type { QuantizationConfig, QuantParams } from './utils/quantization';

// Fine-tuning
export {
  FineTuneTrainer,
  FineTuneStrategy,
  LoRALayer,
  AdapterLayer,
  PrefixTuningLayer,
  createLoRAModel,
  fineTune,
} from './utils/finetune';
export type { FineTuneConfig } from './utils/finetune';

// Визуализация моделей
export {
  visualize,
  visualizeVertical,
  visualizeFlow,
  printModel,
  summary,
  extendModuleWithVisualize,
} from './utils/visualize';
export type { VisualizeOptions } from './utils/visualize';

// Онлайн/Real-time обучение
export {
  OnlineLearner,
  ContinualLearner,
  MetaLearner,
  SelfTrainer,
} from './utils/online-learning';
export type { OnlineLearningConfig } from './utils/online-learning';

// ============================================
// DATA - Расширенные возможности
// ============================================
export {
  StreamingDataset,
  HuggingFaceDataset,
  SequenceDataset,
  NameDataset,
  DataGenerator,
  loadJson,
  loadJsonl,
  loadCsv,
  createHuggingFaceLoader,
} from './data/dataloader';
export type { StreamConfig, HuggingFaceConfig } from './data/dataloader';

// ============================================
// RL - Reinforcement Learning
// ============================================
export {
  ReplayBuffer,
  DQNAgent,
  PolicyGradientAgent,
  ActorCriticAgent,
  CartPoleEnv,
  GridWorldEnv,
  trainDQN,
} from './rl/index';
export type {
  Experience,
  DQNConfig,
  PolicyGradientConfig,
  ActorCriticConfig,
} from './rl/index';

// ============================================
// VERSION
// ============================================
export const VERSION = '2.1.0';
export const FRAMEWORK_NAME = 'Brainy';
