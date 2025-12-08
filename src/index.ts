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

// ============================================
// VERSION
// ============================================
export const VERSION = '1.0.0';
export const FRAMEWORK_NAME = 'Brainy';
